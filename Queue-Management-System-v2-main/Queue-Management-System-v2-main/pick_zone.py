"""
Zone picker — click polygon points on a live frame, press Enter to confirm.
Prints coordinates ready to paste into config2.yml as mask_zones.

Controls:
  Left-click      add a point
  Right-click     undo last point
  Enter / Space   confirm zone and print coordinates
  R               reset / start over
  Q / Escape      quit
"""

import cv2
import numpy as np
import yaml

CONFIG   = './config.yml'
CONFIG2  = './config2.yml'
WIN      = 'Pick Zone  [left-click=add  right-click=undo  Enter=confirm  R=reset  Q=quit]'


def load_stream_url():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    username   = cfg.get('username', '')
    password   = cfg.get('password', '')
    ip_address = cfg.get('ip_address', '')

    if not ip_address:
        return 'live_test_video.mp4'
    if ip_address.startswith('http'):
        return ip_address
    if any(ip_address.lower().endswith(ext) for ext in ['.mp4', '.avi', '.m4v']):
        return ip_address
    if username and password:
        return f'rtsp://{username}:{password}@{ip_address}'
    return f'rtsp://{ip_address}'


def grab_frame(url):
    cap = cv2.VideoCapture(url)
    for _ in range(5):          # skip a few frames so camera warms up
        cap.grab()
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from: {url}")
    return cv2.resize(frame, (640, 480))


def draw_state(base, points, hover=None):
    img = base.copy()
    pts = points + ([hover] if hover else [])

    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (0, 255, 255), 2, cv2.LINE_AA)
        if len(points) >= 3:
            cv2.line(img, pts[-1], pts[0], (0, 255, 255), 1, cv2.LINE_AA)

    for i, p in enumerate(points):
        cv2.circle(img, p, 5, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    if len(points) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(pts[:len(points)], dtype=np.int32)], (0, 0, 0))
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    hint = "Left-click to add points | Right-click to undo | Enter=confirm | R=reset | Q=quit"
    cv2.putText(img, hint, (6, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    url = load_stream_url()
    print(f"Connecting to: {url}")
    base = grab_frame(url)
    print("Frame grabbed. Draw your zone in the window.")

    points = []
    hover  = None

    def on_mouse(event, x, y, flags, _):
        nonlocal hover
        hover = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 600)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        cv2.imshow(WIN, draw_state(base, points, hover))
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 32):          # Enter or Space → confirm
            if len(points) < 3:
                print("Need at least 3 points — keep clicking.")
                continue
            break

        elif key in (ord('r'), ord('R')):
            points.clear()

        elif key in (ord('q'), ord('Q'), 27):  # Q or Escape → quit without saving
            cv2.destroyAllWindows()
            print("Cancelled.")
            return

    cv2.destroyAllWindows()

    # ── Print result ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Zone confirmed!  Paste this into config2.yml:\n")
    print("mask_zones:")
    print("  -")
    for x, y in points:
        print(f"    - [{x}, {y}]")
    print()
    print("Points as a flat list (for reference):")
    print([list(p) for p in points])
    print("=" * 60)


if __name__ == '__main__':
    main()
