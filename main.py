"""
main.py — Command-Line Runner
==============================
Use this if you don't want the Streamlit UI.
Runs the full recognition + attendance marking loop from terminal.

Usage:
    python main.py              # Start recognition
    python main.py --encode     # Re-generate encodings only
    python main.py --summary    # Print today's summary
    python main.py --query      # Interactive NLP query mode
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from utils.face_encoder       import generate_encodings
from utils.face_recognizer    import run_recognition
from utils.nlp_engine         import generate_summary, answer_query


def main():
    parser = argparse.ArgumentParser(
        description="Face Attendance System — CLI"
    )
    parser.add_argument("--encode",  action="store_true",
                        help="Generate face encodings from dataset")
    parser.add_argument("--summary", action="store_true",
                        help="Print today's attendance summary")
    parser.add_argument("--query",   action="store_true",
                        help="Interactive NLP query mode")
    parser.add_argument("--camera",  type=int, default=0,
                        help="Camera index (default: 0)")

    args = parser.parse_args()

    # ── Encode mode ───────────────────────────────────────────
    if args.encode:
        print("\n🔄 Generating face encodings…\n")
        try:
            enc = generate_encodings()
            print(f"\n✅ Encoded {len(enc)} person(s).")
        except Exception as e:
            print(f"❌ Error: {e}")
        return

    # ── Summary mode ──────────────────────────────────────────
    if args.summary:
        print()
        print(generate_summary())
        return

    # ── Interactive NLP mode ──────────────────────────────────
    if args.query:
        print("\n🤖 NLP Query Mode  (type 'exit' to quit)\n")
        while True:
            try:
                q = input("❓ Your question: ").strip()
                if q.lower() in ("exit", "quit", "q"):
                    break
                if q:
                    print(f"\n💬 {answer_query(q)}\n")
            except (KeyboardInterrupt, EOFError):
                break
        return

    # ── Default: Run recognition ──────────────────────────────
    print("\n🎥 Starting Face Recognition Attendance System…")
    print("   Press 'Q' in the camera window to stop.\n")
    run_recognition(camera_index=args.camera)


if __name__ == "__main__":
    main()
