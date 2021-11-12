"""
code to solve exe games
"""

import logging
import sys
import win32gui
import pyautogui

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

def get_active_window():
    """
    get the currently active window

    returns
    -------
    string :
        name of the currently active window
    """
    active_window_name = None
    window = win32gui.GetForegroundWindow()
    print(type(window))
    print(window)
    active_window_name = win32gui.GetWindowText(window)
    return active_window_name

def get_named_window(window_name):
    """
    gets the handle of named window

    returns
    -------
    int :
        value of window
    """
    window = win32gui.FindWindow(None, window_name)
    return window

def get_window_image(window_handle):
    """
    gets an image of a window given the handle
    returns
    image object :
        image of window
    """
    win32gui.SetForegroundWindow(window_handle)
    x, y, x1, y1 = win32gui.GetClientRect(window_handle)
    x, y = win32gui.ClientToScreen(window_handle, (x, y))
    x1, y1 = win32gui.ClientToScreen(window_handle, (x1-x, y1-y))
    im = pyautogui.screenshot(region=(x, y, x1, y1))
    return im

# print(f"Active window: {get_active_window()}")
WINDOW_TO_FIND = "Tents"
# WINDOW_TO_FIND = "code.py - python-game-solver - Visual Studio Code"
# WINDOW_TO_FIND = get_active_window()
WINDOW = get_named_window(WINDOW_TO_FIND)
print(f"Window {WINDOW_TO_FIND} has id {WINDOW}")

SCREENSHOT = get_window_image(WINDOW)
SCREENSHOT.show()