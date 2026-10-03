"""
Reproduction script for UnicodeEncodeError on cp1252 Windows consoles.
Tests printing strings found in ensemble_predict.py, db_logger.py, and queue_management_v2.py
against cp1252 text streams.
"""
import io
import sys

SAMPLE_LOG_STRINGS = [
    "[Wait] Browsing gap = 15 min (5 buckets) \u2014 shifting entrance \u2192 checkout...",
    "  {'Time':<10} {'Prophet':>10} {'LSTM':>10} {'XGBoost':>10} {'Ensemble':>10} {'\u2192Checkout':>12}",
    "  (\u2192Checkout = entrance shifted 15 min forward for browsing gap)",
    "[DB] Connected to PostgreSQL \u2713",
    "[Backtest] Models loaded \u2713",
    "[COUNTER] Line crossing mode | line=p1\u2192p2 | direction=entering",
    "\u2717 Pipeline failed with ValueError",
]

def main():
    print("[*] Testing Unicode console encoding under cp1252...")
    cp1252_stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")

    reproduced_count = 0
    for idx, s in enumerate(SAMPLE_LOG_STRINGS, 1):
        try:
            cp1252_stream.write(s + "\n")
            cp1252_stream.flush()
            print(f"  String #{idx}: OK under cp1252 (Unexpected)")
        except UnicodeEncodeError as exc:
            reproduced_count += 1
            char_repr = ascii(exc.object[exc.start:exc.end])
            print(f"  String #{idx}: REPRODUCED UnicodeEncodeError: {exc.encoding} cannot encode {char_repr}")

    print("\n[Result Summary]")
    print(f"  Total tested: {len(SAMPLE_LOG_STRINGS)} | UnicodeEncodeErrors: {reproduced_count}")
    if reproduced_count > 0:
        print("  => REPRODUCTION SUCCESSFUL: Non-ASCII characters cause terminal/log crash in cp1252 environments.")
    else:
        print("  => No errors observed.")

if __name__ == "__main__":
    main()
