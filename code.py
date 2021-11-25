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
import copy
import math
import win32api
import win32con
import pywinauto
from time import sleep

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

def get_lines(image_to_process):
    """
    calculates where lines are

    returns
    -------
    something with line data in
    """
    EDGES = cv.Canny(image_to_process,90,150,apertureSize = 3)
    KERNEL = np.ones((3,3),np.uint8)
    EDGES = cv.dilate(EDGES,KERNEL,iterations = 1)
    KERNEL = np.ones((5,5),np.uint8)
    EDGES = cv.erode(EDGES,KERNEL,iterations = 1)

    LINES = cv.HoughLines(EDGES,1,np.pi/180,150)
    return LINES

def get_corners(line_data_object):
    """
    get corners where lines cross

    return
    -------
    something with corner data in
    """
    corners = []
    for i in range(len(line_data_object)):
        for line in line_data_object[i+1:]:
            rho1,theta1 = line_data_object[i][0]
            rho2,theta2 = line[0]
            if np.abs(theta1 - theta2) > np.pi/4:
                # at least 45 deg difference
                crossover = get_line_cross(rho1, theta1, rho2, theta2)
                if not line_data_object is False:
                    corners.append(crossover)

    corners.sort(key = lambda tup: tup[0])
    corners.sort(key = lambda tup: tup[1])
    corners = np.array_split(corners, HORIZ_LINES)
    corners = corners[:-1]
    for i in range(len(corners)):
        corners[i] = corners[i][:-1]

    return corners

def resize_image(img_to_resize, scaler):
    """
    resizes image using given scaler

    returns
    resized image
    """
    width = int(img_to_resize.shape[1] * scaler)
    height = int(img_to_resize.shape[0] * scaler)
    dim = (width, height)
    resized_image = cv.resize(img_to_resize, dim, interpolation = cv.INTER_AREA)
    return resized_image

def get_col(grid, col):
    col_list = []
    for x in grid[:-1]:
        col_list.append(x[0][col])
    return col_list

def check_0(grid):
    """
    clears rows/cols with 0 tents
    """
    for i,x in enumerate(grid[:-1]):
        if x[1] == "0":
            for j,y in enumerate(x[0]):
                if (y == " "):
                    grid[i][0][j] = "."

    for i,x in enumerate(grid[-1][0]):
        if x == "0":
            for j,y in enumerate(get_col(grid,i)):
                if (y == " "):
                    grid[j][0][i] = "."

def check_lonely(grid, swamp=False):
    """
    checks for cells with no tree next to them
    """
    for i,x in enumerate(grid[:-1]):
        for j,y in enumerate(x[0]):
            # check if tree
            if (y == "T" or y == "C" or y == "x" or y == "X"):
                continue
            has_tree = False
            # check left
            if (j>0):
                if (x[0][j-1] == "T"):
                    has_tree = True
            # check right
            if (j<(GAME_COLS-1)):
                if (x[0][j+1] == "T"):
                    has_tree = True
            #check above
            if (i>0):
                if (grid[i-1][0][j] == "T"):
                    has_tree = True
            #check below
            if (i<(GAME_ROWS-1)):
                if (grid[i+1][0][j] == "T"):
                    has_tree = True
            if (has_tree):
                if (swamp and grid[i][0][j] == " "):
                    check_swamp(grid, i, j)
            else:
                grid[i][0][j] = "."

def check_chopped(grid):
    """
    checks if tree can only have 1 tent next to it

    returns
    -------
    bool if valid
    """
    for i,x in enumerate(grid[:-1]):
        for j,y in enumerate(x[0]):
            if (not grid[i][0][j] == "T"):
                continue                
            tent_space = ""
            solitary_tent = True
            # check above
            if (i > 0):
                cell = grid[i-1][0][j]
                if (cell == "x" or cell == " "):
                    if (tent_space == ""):
                        tent_space = (i-1,j)
                    else:
                        solitary_tent = False
            # check below
            if (i < (GAME_ROWS - 1)):
                cell = grid[i+1][0][j]
                if (cell == "x" or cell == " "):
                    if (tent_space == ""):
                        tent_space = (i+1,j)
                    else:
                        solitary_tent = False
            # check left
            if (j > 0):
                cell = grid[i][0][j-1]
                if (cell == "x" or cell == " "):
                    if (tent_space == ""):
                        tent_space = (i,j-1)
                    else:
                        solitary_tent = False
            # check right
            if (j < (GAME_COLS - 1)):
                cell = grid[i][0][j+1]
                if (cell == "x" or cell == " "):
                    if (tent_space == ""):
                        tent_space = (i,j+1)
                    else:
                        solitary_tent = False
            if (tent_space == ""):
                print("Oh no, tree placed with no tent!", i, j)
                return False
            elif (solitary_tent == True):
                # only one tent next to tree
                grid[i][0][j] = "C"
                cell_before = grid[tent_space[0]][0][tent_space[1]]
                grid[tent_space[0]][0][tent_space[1]] = "X"
                if (not cell_before == "x"):
                    clear_tent(grid, tent_space[0], tent_space[1])
    return True

def chop_tree(grid, row, col):
    """
    if a tent has only one tree next to it then it chops the tree
    """
    tree_space = ""
    solitary_tree = True
    # check above
    if (row > 0):
        if (grid[row-1][0][col] == "T"):
            if (tree_space == ""):
                tree_space = (row-1,col)
            else:
                solitary_tree = False
    # check below
    if (row < (GAME_ROWS - 1)):
        if (grid[row+1][0][col] == "T"):
            if (tree_space == ""):
                tree_space = (row+1,col)
            else:
                solitary_tree = False
    # check left
    if (col > 0):
        if (grid[row][0][col-1] == "T"):
            if (tree_space == ""):
                tree_space = (row,col-1)
            else:
                solitary_tree = False
    # check right
    if (col < (GAME_COLS - 1)):
        if (grid[row][0][col+1] == "T"):
            if (tree_space == ""):
                tree_space = (row,col+1)
            else:
                solitary_tree = False
    if (tree_space == ""):
        print("Oh no, tent placed with no tree!")
    elif (solitary_tree == True):
        # only one tree next to tent
        grid[tree_space[0]][0][tree_space[1]] = "C"
        grid[row][0][col] = "X"

def clear_tent(grid, row, col):
    """
    clears the space around a new tent and chops it's tree if possible
    """
    # above left
    if (row > 0 and col > 0):
        if (grid[row-1][0][col-1] == " "):
            grid[row-1][0][col-1] = "."
    # above
    if (row > 0):
        if (grid[row-1][0][col] == " "):
            grid[row-1][0][col] = "."
    # above right
    if (row > 0 and col < (GAME_COLS-1)):
        if (grid[row-1][0][col+1] == " "):
            grid[row-1][0][col+1] = "."
    # left
    if (col > 0):
        if (grid[row][0][col-1] == " "):
            grid[row][0][col-1] = "."
    # right
    if (col < (GAME_COLS-1)):
        if (grid[row][0][col+1] == " "):
            grid[row][0][col+1] = "."
    # below left
    if (row < (GAME_ROWS-1) and col > 0):
        if (grid[row+1][0][col-1] == " "):
            grid[row+1][0][col-1] = "."
    # below
    if (row < (GAME_ROWS-1)):
        if (grid[row+1][0][col] == " "):
            grid[row+1][0][col] = "."
    # below right
    if (row < (GAME_ROWS-1) and col < (GAME_COLS-1)):
        if (grid[row+1][0][col+1] == " "):
            grid[row+1][0][col+1] = "."
    if (not grid[row][0][col] == "X"):
        chop_tree(grid, row, col)

def get_spaces(arr):
    """
    gets a list of spaces from a given row/col

    returns
    pair of array of spaces and tents placed
    """
    # count groups of spaces
    spaces = []
    group = 0
    start = 0
    in_group = False
    tents_placed = 0
    for j,y in enumerate(arr):
        if (y == "x" or y == "X"):
            tents_placed += 1
        elif (y == " "):
            if (not in_group):
                start = j
            in_group = True
            group += 1
        else:
            if (group > 0):
                spaces.append((group, start))
                group = 0
            in_group = False
    if (in_group):
        spaces.append((group, start))
    return (spaces, tents_placed)

def spaces_to_places(lst):
    """
    calculates possible placements from spaces list

    returns
    -------
    max placements possible
    """
    plcs = sum(map(lambda x : math.ceil(x[0]/2), lst))
    return plcs

def check_solved(grid):
    """
    checks for rows/cols where tents = open spaces/groups
    """
    for i,x in enumerate(grid[:-1]):
        spaces, tents_placed = get_spaces(x[0])
        possible_placements = spaces_to_places(spaces)
        tents_to_place = int(x[1]) - tents_placed
        # print(spaces)
        # print(possible_placements, tents_to_place)
        if (possible_placements == tents_to_place):
            # print(i)
            for j,y in enumerate(spaces):
                if (y[0]%2 == 1):
                    for k in range(math.ceil(y[0]/2)):
                        # print(spaces[j][1], k)
                        grid[i][0][spaces[j][1]+k*2] = "x"
                        clear_tent(grid, i, spaces[j][1]+k*2)
    
    for i,x in enumerate(grid[-1][0]):
        spaces, tents_placed = get_spaces(get_col(grid, i))
        possible_placements = spaces_to_places(spaces)
        tents_to_place = int(x) - tents_placed
        # print(possible_placements, tents_to_place)
        if (possible_placements == tents_to_place):
            for j,y in enumerate(spaces):
                if (y[0]%2 == 1):
                    for k in range(math.ceil(y[0]/2)):
                        grid[spaces[j][1]+k*2][0][i] = "x"
                        clear_tent(grid, spaces[j][1]+k*2, i)
        
def check_done(grid):
    """
    finds rows/cols that have enough tents and clears them
    """
    for i,x in enumerate(grid[:-1]):
        tents_placed = 0
        for j,y in enumerate(x[0]):
            if (y == "x" or y == "X"):
                tents_placed += 1
        # print(tents_placed, x[1])
        if (tents_placed == int(x[1])):
            for j,y in enumerate(x[0]):
                if (y == " "):
                    grid[i][0][j] = "."
    
    for i,x in enumerate(grid[-1][0]):
        tents_placed = 0
        for j,y in enumerate(get_col(grid, i)):
            if (y == "x" or y == "X"):
                tents_placed += 1
        if (tents_placed == int(x)):
            for j,y in enumerate(get_col(grid, i)):
                if (y == " "):
                    grid[j][0][i] = "."

def check_neighbours(grid):
    """
    checks if it has to be one of 2 and then eliminates neighbours
    """
    for i,x in enumerate(grid[:-1]):
        spaces, tents_placed = get_spaces(x[0])
        tents_to_place = int(x[1]) - tents_placed
        for j,y in enumerate(spaces):
            if (y[0] > 1 and spaces_to_places(spaces) == tents_to_place):
                # 2 spaces next to each other
                # check each in group
                for k in range(int(y[0])-1):
                    # check above first
                    if (i > 0):
                        if (grid[i-1][0][spaces[j][1]+k] == " "):
                            grid[i-1][0][spaces[j][1]+k] = "."
                    # check above second
                    if (i > 0):
                        if (grid[i-1][0][spaces[j][1]+k+1] == " "):
                            grid[i-1][0][spaces[j][1]+k+1] = "."
                    # check below first
                    if (i < (GAME_ROWS -1)):
                        if (grid[i+1][0][spaces[j][1]+k] == " "):
                            grid[i+1][0][spaces[j][1]+k] = "."
                    # check above second
                    if (i < (GAME_ROWS -1)):
                        if (grid[i+1][0][spaces[j][1]+k+1] == " "):
                            grid[i+1][0][spaces[j][1]+k+1] = "."
            if ((not y == spaces[-1]) and y[0]%2 == 1 and spaces[j+1][0]%2 == 1 and spaces_to_places(spaces) - 1 == tents_to_place):
                if (y[1]+y[0]-1+2 == spaces[j+1][1]):
                    #check above inbetween
                    if (i > 0):
                        if (grid[i-1][0][spaces[j][1]+spaces[j][0]-1+1] == " "):
                            grid[i-1][0][spaces[j][1]+spaces[j][0]-1+1] = "."
                    #check below inbetween
                    if (i < (GAME_ROWS -1)):
                        if (grid[i+1][0][spaces[j][1]+spaces[j][0]-1+1] == " "):
                            grid[i+1][0][spaces[j][1]+spaces[j][0]-1+1] = "."

    for i,x in enumerate(grid[-1][0]):
        spaces, tents_placed = get_spaces(get_col(grid, i))
        tents_to_place = int(x) - tents_placed
        for j,y in enumerate(spaces):
            if (y[0] > 1 and spaces_to_places(spaces) == tents_to_place):
                # 2 spaces next to each other
                for k in range(int(y[0])-1):
                    # check left first
                    if (i > 0):
                        if (grid[spaces[j][1]+k][0][i-1] == " "):
                            grid[spaces[j][1]+k][0][i-1] = "."
                    # check left second
                    if (i > 0):
                        if (grid[spaces[j][1]+k+1][0][i-1] == " "):
                            grid[spaces[j][1]+k+1][0][i-1] = "."
                    # check right first
                    if (i < (GAME_COLS -1)):
                        if (grid[spaces[j][1]+k][0][i+1] == " "):
                            grid[spaces[j][1]+k][0][i+1] = "."
                    # check right second
                    if (i < (GAME_COLS -1)):
                        if (grid[spaces[j][1]+k+1][0][i+1] == " "):
                            grid[spaces[j][1]+k+1][0][i+1] = "."
            if ((not y == spaces[-1]) and y[0]%2 == 1 and spaces[j+1][0]%2 == 1 and spaces_to_places(spaces) - 1 == tents_to_place):
                if (y[1]+y[0]-1+2 == spaces[j+1][1]):
                    # check left inbetween
                    if (i > 0):
                        if (grid[spaces[j][1]+spaces[j][0]-1+1][0][i-1] == " "):
                            grid[spaces[j][1]+spaces[j][0]-1+1][0][i-1] = "."
                    # check right inbetween
                    if (i < (GAME_COLS - 1)):
                        if (grid[spaces[j][1]+spaces[j][0]-1+1][0][i+1] == " "):
                            grid[spaces[j][1]+spaces[j][0]-1+1][0][i+1] = "."

def click(col, row, right=False):
    """
    clicks at the given x,y on the screen
    """
    x = int((col+0.5)*COLUMN_WIDTH+TOP_CORNER[0])/SCALER
    y = int((row+0.5)*ROW_HEIGHT+TOP_CORNER[1])/SCALER

    window_x, window_y, x1, y1 = win32gui.GetClientRect(WINDOW)
    window_x, window_y = win32gui.ClientToScreen(WINDOW, (window_x, window_y))
    if (right):
        pyautogui.click(window_x+x, window_y+y, button="right")
    else:
        pyautogui.click(window_x+x, window_y+y)

def click_grid(grid, right=False):
    """
    clicks in the right places on the grid
    """
    for i,x in enumerate(grid[:-1]):
        for j,y in enumerate(x[0]):
            if (y == "x" or y == "X"):
                click(j, i)
            elif y == ".":
                if (right):
                    click(j, i, True)

def check_finished(grid):
    """
    check if finished

    returns
    -------
    true or false
    """
    for x in grid[:-1]:
        for y in x[0]:
            if (y == " "):
                return False
    return True

def check_lonely_straight_tree(grid, extra = False):
    """
    checks if tree only has one place for tent
    or if the tent can only be in 1 row or col and then if the row only needs one more then clears the rest of the row
    """
    for i,x in enumerate(grid[:-1]):
        for j,y in enumerate(x[0]):
            if False:
                print("new")
                print(y, "y before check")
                print(grid[i][0][j], "grid before check")
                print(x)
                print(grid[i])
            if (y == "T"):
                if False:
                    print(y, "y after check")
                    print(grid[i][0][j], "grid after check")
                spaces = []
                # check above
                if (i > 0):
                    cell = grid[i-1][0][j]
                    if (cell == "x"):
                        continue
                    elif (cell == " "):
                        # if extra:
                        #     print(grid[i][0][j])
                        spaces.append((i-1,j))
                # check below
                if (i < (GAME_ROWS - 1)):
                    cell = grid[i+1][0][j]
                    if (cell == "x"):
                        continue
                    elif (cell == " "):
                        # if extra:
                        #     print(grid[i][0][j])
                        spaces.append((i+1,j))
                # check left
                if (j > 0):
                    cell = grid[i][0][j-1]
                    if (cell == "x"):
                        continue
                    elif (cell == " "):
                        # if extra:
                        #     print(grid[i][0][j])
                        spaces.append((i,j-1))
                # check right
                if (j < (GAME_COLS - 1)):
                    cell = grid[i][0][j+1]
                    if (cell == "x"):
                        continue
                    elif (cell == " "):
                        # if extra:
                        #     print(grid[i][0][j])
                        spaces.append((i,j+1))
                if (len(spaces) == 0):
                    print("Oh no, tree has no space for a tent!")
                elif (len(spaces) == 1):
                    print("lonely")
                    spaces = spaces[0]
                    grid[spaces[0]][0][spaces[1]] = "x"
                    clear_tent(grid, spaces[0], spaces[1])
                elif (len(spaces) == 2):
                    if (extra):
                        if (spaces[0][0] == spaces[1][0]):
                            # print(i, j)
                            # # spaces in same row
                            # print("same row")
                            # print(spaces)
                            # print(spaces[0][0], spaces[0][1]+1)
                            # print(grid[spaces[0][0]][0][spaces[0][1]+1])
                            # count tents in row
                            tents = 0
                            for z in grid[spaces[0][0]][0]:
                                if (z == "x" or z == "X"):
                                    tents += 1
                            # print(tents, grid[spaces[0][0]][1])
                            # print(tents == (int(grid[spaces[0][0]][1]) -1))
                            if (tents == (int(grid[spaces[0][0]][1]) -1)):
                                for k,z in enumerate(grid[spaces[0][0]][0]):
                                    if (not k == spaces[0][1] and not k == spaces[1][1]):
                                        if (grid[spaces[0][0]][0][k] == " "):
                                            grid[spaces[0][0]][0][k] = "."
                        elif (spaces[0][1] == spaces[1][1]):
                            # print(i, j)
                            # # spaces in same col
                            # print("same col")
                            # print(spaces)
                            # print(spaces[0][0]+1, spaces[0][1])
                            # print(grid[spaces[0][0]+1][0][spaces[0][1]])
                            # count tents in col
                            tents = 0
                            for z in get_col(grid, spaces[0][1]):
                                if (z == "x" or z == "X"):
                                    tents += 1
                            # print(tents, grid[-1][0][spaces[0][1]])
                            # print(tents == (int(grid[-1][0][spaces[0][1]]) - 1))
                            if (tents == (int(grid[-1][0][spaces[0][1]]) - 1)):
                                for k,z in enumerate(get_col(grid, spaces[0][1])):
                                    if (not k == spaces[0][0] and not k == spaces[1][0]):
                                        if (grid[k][0][spaces[0][1]] == " "):
                                            grid[k][0][spaces[0][1]] = "."

def check_swamp(grid, row, col):
    """
    checks if placing a tent would "swamp" trees such that they can't get enough tents
    """
    temp_grid = copy.deepcopy(grid)
    temp_grid[row][0][col] = "x"
    clear_tent(temp_grid, row, col)
    # print_grid(temp_grid)
    if (not check_chopped(temp_grid)):
        print("Reverting. Must be clear.")
        grid[row][0][col] = "."

def print_grid(grid):
    for x in grid:
        print(x)
    print("")


WINDOW_TO_FIND = "Tents"
WINDOW = get_named_window(WINDOW_TO_FIND)

PUZZLES_TO_SOLVE = 1
SOLVE_STATS = {"puzzles": 0,
                "loops": 0,
                "swamps": 0,
                "straights": 0,
                "max": 0,
                "min": 99,
                "average_loops": 0}

for loop in range(PUZZLES_TO_SOLVE):
    SCREENSHOT = get_window_image(WINDOW)
    OPEN_CV_IMAGE = np.array(SCREENSHOT)

    COLOUR_CORRECTED_IMAGE = cv.cvtColor(OPEN_CV_IMAGE, cv.COLOR_BGR2RGB)
    GREY_IMAGE = cv.cvtColor(COLOUR_CORRECTED_IMAGE, cv.COLOR_BGR2GRAY)

    LINES = get_lines(GREY_IMAGE)

    if LINES is None:
        print('No LINES were found')
        exit()

    # print('number of Hough lines:', len(LINES))

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
        y1 = int(y0 + 5000*(a))
        x1 = int(x0 + 5000*(-b))
        x2 = int(x0 - 5000*(-b))
        y2 = int(y0 - 5000*(a))

        cv.line(COLOUR_CORRECTED_IMAGE,(x1,y1),(x2,y2),(0,0,255),2)

    GAME_ROWS = HORIZ_LINES - 1
    GAME_COLS = VERT_LINES - 1
    print(f"number of rows: {GAME_ROWS}")
    print(f"number of cols: {GAME_COLS}")

    CORNERS = get_corners(LINES)

    TOP_CORNER = CORNERS[0][0]
    COLUMN_WIDTH = CORNERS[0][1][0] - CORNERS[0][0][0]
    ROW_HEIGHT = CORNERS[1][0][1] - CORNERS[0][0][1]
    print(f"col: {COLUMN_WIDTH}, row: {ROW_HEIGHT}")

    if (COLUMN_WIDTH < 48):
        print('grid maybe too small')
        # exit()

    SCALER = 60/COLUMN_WIDTH
    GREY_IMAGE = resize_image(GREY_IMAGE, SCALER)
    COLOUR_CORRECTED_IMAGE = resize_image(COLOUR_CORRECTED_IMAGE, SCALER)
    for x in LINES:
        x[0][0] *= SCALER
        x[0][1] *= SCALER

    for x in CORNERS:
        for y in x:
            y[0] *= SCALER
            y[1] *= SCALER

    COLUMN_WIDTH = 60
    ROW_HEIGHT = 60
    TREE_POS = (int(COLUMN_WIDTH*0.5), int(ROW_HEIGHT*0.4))
    TOP_CORNER[0] *= SCALER
    TOP_CORNER[1] *= SCALER

    # game_grid = list([[[""] * game_cols, ""] * game_rows])

    # game_grid = [["" for _ in range(game_cols)] for _ in range(game_rows)]

    game_grid = []

    for x in range(GAME_ROWS):
        game_grid.append([[" "] * GAME_COLS, ""])

    game_grid.append([[""] * GAME_COLS, ""])

    # print(game_grid)
    # game_grid[0][0][0] = "T"
    # print(game_grid)
    # exit()

    for row in CORNERS:
        for col in row:
            middle_pos = col+TREE_POS
            blue_in_cell = COLOUR_CORRECTED_IMAGE[middle_pos[1]][middle_pos[0]][0]
            green_in_cell = COLOUR_CORRECTED_IMAGE[middle_pos[1]][middle_pos[0]][1]
            red_in_cell = COLOUR_CORRECTED_IMAGE[middle_pos[1]][middle_pos[0]][2]
            if blue_in_cell < 50 and red_in_cell < 50:
                cv.circle(COLOUR_CORRECTED_IMAGE, middle_pos, radius = 5, color = (255, 0, 0), thickness = -1)
                cell_row = int((col[1]-TOP_CORNER[1])/ROW_HEIGHT)
                cell_col = int((col[0]-TOP_CORNER[0])/COLUMN_WIDTH)
                game_grid[cell_row][0][cell_col] = "T"
            else:
                cv.circle(COLOUR_CORRECTED_IMAGE, middle_pos, radius = 5, color = (0, 255, 0), thickness = -1)

    # exit()
    # GREY = cv.cvtColor(COLOUR_CORRECTED, cv.COLOR_BGR2GRAY)
    # cv.imwrite("./output.png", COLOUR_CORRECTED)
    # cv.imshow("image window", COLOUR_CORRECTED_IMAGE)
    # cv.waitKey(0)

    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))

    model = cv.ml.KNearest_create()
    model.train(samples,cv.ml.ROW_SAMPLE,responses)

    thresh = cv.adaptiveThreshold(GREY_IMAGE,255,1,1,11,2)

    contours,hierarchy = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv.contourArea(cnt)>50:
            [x,y,w,h] = cv.boundingRect(cnt)
            if  h==22:
                cv.rectangle(COLOUR_CORRECTED_IMAGE,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                # retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                string = str(int((results[0][0])))
                cv.putText(COLOUR_CORRECTED_IMAGE,string,(x,y+h),0,1,(0,255,0))

                cell_row = int((y+h/2-TOP_CORNER[1])/ROW_HEIGHT)
                cell_col = int((x+w/2-TOP_CORNER[0])/COLUMN_WIDTH)
                
                if (cell_row > (GAME_ROWS - 1)):
                    # print("bottom number")
                    game_grid[cell_row][0][cell_col] = string
                elif (cell_col > (GAME_COLS -1)):
                    # print("side number")
                    game_grid[cell_row][1] = string
                else:
                    cv.putText(COLOUR_CORRECTED_IMAGE,string,(x,y+h),0,1,(0,0,255))
                    cv.imshow("image window", COLOUR_CORRECTED_IMAGE)
                    cv.waitKey(0)

    # print(game_grid)
    # cv.imshow("image window", COLOUR_CORRECTED_IMAGE)
    # cv.waitKey(0)

    working_grid = copy.deepcopy(game_grid)
    # check_0(working_grid)
    # check_lonely(working_grid)

    solve_loops = 0
    show_working = False
    while True:
        solve_loops += 1
        start_grid = copy.deepcopy(working_grid)
        check_lonely(working_grid)
        check_solved(working_grid)
        check_done(working_grid)
        check_neighbours(working_grid)
        check_lonely_straight_tree(working_grid)
        check_chopped(working_grid)

        if (check_finished(working_grid)):
            print(f"Done! It took {solve_loops} loops to solve.")
            SOLVE_STATS["puzzles"] += 1
            SOLVE_STATS["loops"] += solve_loops
            if (SOLVE_STATS["max"] < solve_loops):
                SOLVE_STATS["max"] = solve_loops
            if (SOLVE_STATS["min"] > solve_loops):
                SOLVE_STATS["min"] = solve_loops
            break
        if (start_grid == working_grid):
            print(f"No change after {solve_loops} loops. Trying swamp.")
            SOLVE_STATS["swamps"] += 1
            check_lonely(working_grid, True)
            if (start_grid == working_grid):
                print(f"No change after swamp. Trying straight.")
                SOLVE_STATS["straights"] += 1
                check_lonely_straight_tree(working_grid, True)
                if (start_grid == working_grid):
                    print(f"No change after straight.")
                    show_working = True
                    break

    # check_lonely(working_grid)
    # check_solved(working_grid)
    # check_done(working_grid)
    # check_neighbours(working_grid)
    # check_lonely_tree(working_grid)
    # chop_tree(working_grid, 7, 7)
    # check_lonely(working_grid)
    # check_chopped(working_grid)
    # clear_tent(working_grid, 12, 0)
    # clear_tent(working_grid, 2, 8)
    # print_grid(working_grid)
    # check_lonely(working_grid, True)

    click_grid(working_grid, show_working)
    if (show_working):
        print_grid(working_grid)
        break

    # sleep(1)
    # pyautogui.press('n')
    # sleep(4)

SOLVE_STATS["average_loops"] = SOLVE_STATS["loops"]/SOLVE_STATS["puzzles"]

check_lonely_straight_tree(working_grid, True)
print_grid(working_grid)
print(SOLVE_STATS)
