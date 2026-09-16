import pyautogui


# ======================================
# SETTINGS
# ======================================

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SMOOTHING = 0.25


# Current mouse position
current_x = SCREEN_WIDTH // 2
current_y = SCREEN_HEIGHT // 2


# ======================================
# MOVE MOUSE
# ======================================

def move_mouse(hand_x, hand_y):

    global current_x
    global current_y

    # Convert camera coordinates
    # to screen coordinates

    target_x = (
        hand_x / FRAME_WIDTH
    ) * SCREEN_WIDTH

    target_y = (
        hand_y / FRAME_HEIGHT
    ) * SCREEN_HEIGHT


    # Smooth movement

    current_x = (
        current_x
        + (target_x - current_x)
        * SMOOTHING
    )

    current_y = (
        current_y
        + (target_y - current_y)
        * SMOOTHING
    )


    # Keep inside screen

    current_x = max(
        0,
        min(
            SCREEN_WIDTH - 1,
            current_x
        )
    )

    current_y = max(
        0,
        min(
            SCREEN_HEIGHT - 1,
            current_y
        )
    )


    pyautogui.moveTo(
        int(current_x),
        int(current_y),
        duration=0
    )


# ======================================
# MOUSE CLICK
# ======================================

def mouse_click():

    pyautogui.click()


# ======================================
# DOUBLE CLICK
# ======================================

def mouse_double_click():

    pyautogui.doubleClick()


# ======================================
# RIGHT CLICK
# ======================================

def mouse_right_click():

    pyautogui.rightClick()


# ======================================
# RESET POSITION
# ======================================

def reset_mouse():

    global current_x
    global current_y

    current_x = SCREEN_WIDTH // 2
    current_y = SCREEN_HEIGHT // 2

    pyautogui.moveTo(
        current_x,
        current_y
    )