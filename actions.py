import time
import pyautogui
from collections import deque


# ======================================
# SETTINGS
# ======================================

COOLDOWN = 1.5

last_action_time = 0
last_gesture = None

last_action = "WAITING..."
action_status = "READY"

# Latest 5 actions
action_history = deque(maxlen=5)


# ======================================
# PERFORM GESTURE ACTION
# ======================================

def perform_gesture_action(gesture):

    global last_action_time
    global last_gesture
    global last_action
    global action_status

    gesture = str(gesture).upper().strip()

    current_time = time.time()


    # ----------------------------------
    # Ignore invalid states
    # ----------------------------------

    if gesture in [
        "NO HAND",
        "UNKNOWN"
    ]:
        return


    # ----------------------------------
    # Don't repeat same gesture
    # ----------------------------------

    if gesture == last_gesture:
        return


    # ----------------------------------
    # Cooldown
    # ----------------------------------

    if current_time - last_action_time < COOLDOWN:
        return


    last_gesture = gesture
    last_action_time = current_time

    action_status = "EXECUTING"


    print("\n================================")
    print(f"[GESTURE] {gesture}")


    # ==================================
    # 👍 VOLUME UP
    # ==================================

    if gesture in [
        "THUMBS_UP",
        "THUMBS UP",
        "THUMB_UP"
    ]:

        pyautogui.press("volumeup")

        last_action = "VOLUME UP"


    # ==================================
    # 👎 VOLUME DOWN
    # ==================================

    elif gesture in [
        "THUMBS_DOWN",
        "THUMBS DOWN",
        "THUMB_DOWN",

        "THUMBS_UP_INVERTED",
        "THUMBS UP INVERTED",

        "THUMB_UP_INVERTED",
        "THUMB UP INVERTED"
    ]:

        pyautogui.press("volumedown")

        last_action = "VOLUME DOWN"


    # ==================================
    # ✊ PLAY / PAUSE
    # ==================================

    elif gesture in [
        "FIST",
        "CLOSED_FIST"
    ]:

        pyautogui.press("playpause")

        last_action = "PLAY / PAUSE"


    # ==================================
    # ✌️ SCREENSHOT
    # ==================================

    elif gesture in [
        "PEACE",
        "V_SIGN",
        "VICTORY"
    ]:

        screenshot = pyautogui.screenshot()

        filename = (
            f"screenshot_"
            f"{int(time.time())}.png"
        )

        screenshot.save(filename)

        last_action = "SCREENSHOT"

        print(f"[SAVED] {filename}")


    # ==================================
    # ☝️ NEXT TRACK
    # ==================================

    elif gesture in [
        "POINT",
        "POINTING"
    ]:

        pyautogui.press("nexttrack")

        last_action = "NEXT TRACK"


    # ==================================
    # 🖐️ STOP MEDIA
    # ==================================

    elif gesture in [
        "OPEN_PALM",
        "OPEN PALM",
        "PALM"
    ]:

        pyautogui.press("stop")

        last_action = "STOP MEDIA"


    # ==================================
    # 🤘 PREVIOUS TRACK
    # ==================================

    elif gesture in [
        "ROCK"
    ]:

        pyautogui.press("prevtrack")

        last_action = "PREVIOUS TRACK"


    # ==================================
    # NO ACTION
    # ==================================

    else:

        last_action = "NO ACTION"

        print(
            f"[ACTION] No mapping for {gesture}"
        )


    # ==================================
    # SAVE HISTORY
    # ==================================

    if last_action != "NO ACTION":

        timestamp = time.strftime("%H:%M:%S")

        history_item = (
            f"{timestamp} - {last_action}"
        )

        action_history.appendleft(
            history_item
        )


    # ==================================
    # STATUS
    # ==================================

    action_status = "EXECUTED"


    print(
        f"[ACTION] {last_action}"
    )

    print("================================\n")


# ======================================
# RESET
# ======================================

def reset_action_state():

    global last_gesture
    global last_action_time
    global last_action
    global action_status

    last_gesture = None

    last_action_time = 0

    last_action = "WAITING..."

    action_status = "READY"

    action_history.clear()