"""Run the VisionControl desktop application."""

from app.controller import VisionControlApp


def main():
    try:
        app = VisionControlApp()
        app.start()
        print("[SYSTEM] Camera connected. Model loaded. Gesture engine ready.")
        app.run()
    except Exception as error:
        print(f"[STARTUP ERROR] {error}")


if __name__ == "__main__":
    main()
