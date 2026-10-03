"""Open the IQMS web app in an emulated iPhone (Safari/WebKit engine).

Setup (once):
    pip install playwright
    playwright install webkit

Usage:
    python scripts/iphone.py
    python scripts/iphone.py --color-scheme light --device "iPhone 13"
    python scripts/iphone.py --url https://other-tunnel.ngrok-free.dev --no-pause
    python scripts/iphone.py --list-devices
"""
import argparse

from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://wildfowl-agenda-curve.ngrok-free.dev"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--device", default="iPhone 15 Pro")
    parser.add_argument("--color-scheme", choices=["dark", "light"], default="dark")
    parser.add_argument("--locale", default="fr-FR")
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--wait-ms", type=int, default=3000, help="time to let live data load before the screenshot")
    parser.add_argument("--screenshot", default="iqms_iphone.png", help="output path, empty string to skip")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-pause", action="store_true", help="close right after the screenshot instead of opening the Inspector")
    parser.add_argument("--list-devices", action="store_true", help="print available iPhone/iPad profiles and exit")
    args = parser.parse_args()

    with sync_playwright() as p:
        if args.list_devices:
            for name in p.devices:
                if "iPhone" in name or "iPad" in name:
                    print(name)
            return

        if args.device not in p.devices:
            parser.error(f"unknown device {args.device!r}; run with --list-devices")

        browser = p.webkit.launch(headless=args.headless)
        context = browser.new_context(
            **p.devices[args.device],
            locale=args.locale,
            timezone_id=args.timezone,
            color_scheme=args.color_scheme,
            # Skips ngrok's "Visit Site" interstitial
            extra_http_headers={"ngrok-skip-browser-warning": "1"},
        )
        page = context.new_page()
        page.goto(args.url)
        page.wait_for_timeout(args.wait_ms)

        if args.screenshot:
            page.screenshot(path=args.screenshot, full_page=True)
            print(f"Screenshot saved to {args.screenshot}")

        if not args.no_pause and not args.headless:
            page.pause()  # keeps the window open + opens the Inspector

        browser.close()


if __name__ == "__main__":
    main()
