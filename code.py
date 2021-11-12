"""
code to solve exe games
"""

import logging
import sys
from types import NoneType
from numpy.core.numeric import cross
import win32gui
import pyautogui
import cv2 as cv
import numpy as np

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
    -------
    image object :
        image of window
    """
    win32gui.SetForegroundWindow(window_handle)
    x, y, x1, y1 = win32gui.GetClientRect(window_handle)
    x, y = win32gui.ClientToScreen(window_handle, (x, y))
    x1, y1 = win32gui.ClientToScreen(window_handle, (x1-x, y1-y))
    im = pyautogui.screenshot(region=(x, y, x1, y1))
    return im

def get_line_cross(rho1, theta1, rho2, theta2):
    """
    calculates whether two line cross given their rho and theta

    returns
    -------
        dict of x.y if they cross, false if not
    """
    ct1 = np.cos(theta1)
    st1 = np.sin(theta1)
    ct2 = np.cos(theta2)
    st2 = np.sin(theta2)
    d = ct1*st2-st1*ct2
    if (d != 0):
        return (int(((st2*rho1-st1*rho2)/d).item()), int(((-ct2*rho1+ct1*rho2)/d).item()))
    else:
        return False

# print(f"Active window: {get_active_window()}")
WINDOW_TO_FIND = "Tents"
# WINDOW_TO_FIND = "code.py - python-game-solver - Visual Studio Code"
# WINDOW_TO_FIND = get_active_window()
WINDOW = get_named_window(WINDOW_TO_FIND)
print(f"Window {WINDOW_TO_FIND} has id {WINDOW}")

SCREENSHOT = get_window_image(WINDOW)
OPEN_CV_IMAGE = np.array(SCREENSHOT)
COLOUR_CORRECTED = cv.cvtColor(OPEN_CV_IMAGE, cv.COLOR_BGR2RGB)
GREY = cv.cvtColor(COLOUR_CORRECTED, cv.COLOR_BGR2GRAY)
EDGES = cv.Canny(GREY,90,150,apertureSize = 3)
KERNEL = np.ones((3,3),np.uint8)
EDGES = cv.dilate(EDGES,KERNEL,iterations = 1)
KERNEL = np.ones((5,5),np.uint8)
EDGES = cv.erode(EDGES,KERNEL,iterations = 1)

LINES = cv.HoughLines(EDGES,1,np.pi/180,150)

if LINES is None:
    print('No LINES were found')
    exit()

print('number of Hough lines:', len(LINES))

HORIZ_LINES = 0
VERT_LINES = 0

for line in LINES:
    rho,theta = line[0]
    if (theta < np.pi/4 and theta > -np.pi/4) or (theta < 5*np.pi/4 and theta > 3*np.pi/4):
        VERT_LINES += 1
    else:
        HORIZ_LINES += 1
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv.line(COLOUR_CORRECTED,(x1,y1),(x2,y2),(0,0,255),2)

print(f"number of rows: {HORIZ_LINES - 1}")
print(f"number of cols: {VERT_LINES - 1}")

CORNERS = []

for i in range(len(LINES)):
    for line in LINES[i+1:]:
        rho1,theta1 = LINES[i][0]
        rho2,theta2 = line[0]
        if np.abs(theta1 - theta2) > np.pi/4:
            # at least 45 deg difference
            crossover = get_line_cross(rho1, theta1, rho2, theta2)
            if not LINES is False:
                CORNERS.append(crossover)

CORNERS.sort(key = lambda tup: tup[0])
CORNERS.sort(key = lambda tup: tup[1])
CORNERS = np.array_split(CORNERS, HORIZ_LINES)
CORNERS = CORNERS[:-1]
for i in range(len(CORNERS)):
    CORNERS[i] = CORNERS[i][:-1]

COLUMN_WIDTH = CORNERS[0][1][0] - CORNERS[0][0][0]
ROW_HEIGHT = CORNERS[1][0][1] - CORNERS[0][0][1]
TREE_POS = (int(COLUMN_WIDTH*0.5), int(ROW_HEIGHT*0.4))

for row in CORNERS:
    for col in row:
        cv.circle(COLOUR_CORRECTED, col+TREE_POS, radius = 5, color = (255, 0, 0), thickness = -1)

print(f"col: {COLUMN_WIDTH}, row: {ROW_HEIGHT}")



cv.imshow("image window", COLOUR_CORRECTED)
k = cv.waitKey(0)
