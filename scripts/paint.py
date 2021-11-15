#! /usr/bin/env python
import os
import time
import sys
import rospy
import numpy as np
from tqdm import tqdm

from paint_utils import *

q = np.array([0.704020578925, 0.710172716916,0.00244101361829,0.00194372088834])


TABLE_Z = -0.102

CANVAS_POSITION = (0,.5,TABLE_Z)
CANVAS_WIDTH = .2
CANVAS_HEIGHT = .2

WATER_POSITION = (-.3,.6,TABLE_Z)
RAG_POSTITION = (-.3,.3,TABLE_Z)

PALLETTE_POSITION = (.3,.5,TABLE_Z)
PAINT_DIFFERENCE = 0.03976

HOVER_FACTOR = 0.1

curr_position = None

# def global_to_canvas_coordinates(x,y,z):
#     x_new = x + CANVAS_POSITION[0]/2
#     y_new = y - CANVAS_POSITION[1]
#     z_new = z
#     return x_new, y_new, z_new

def canvas_to_global_coordinates(x,y,z):
    x_new = (x -.5) * CANVAS_WIDTH + CANVAS_POSITION[0]
    y_new = y*CANVAS_HEIGHT + CANVAS_POSITION[1]
    z_new = z
    return x_new, y_new, z_new

def _move(x, y, z, timeout=20, method='linear', step_size=.2):
    '''
    Move to given x, y, z in global coordinates
    kargs:
        method 'linear'|'curved'|'direct'
    '''

    global curr_position
    if curr_position is None:
        curr_position = [x, y, z]

    # Calculate how many
    dist = ((x-curr_position[0])**2 + (y-curr_position[1])**2 + (z-curr_position[2])**2)**(0.5)
    n_steps = max(2, int(dist//step_size))

    if method == 'linear':
        x_s = np.linspace(curr_position[0], x, n_steps)
        y_s = np.linspace(curr_position[1], y, n_steps)
        z_s = np.linspace(curr_position[2], z, n_steps)

        for i in range(1,n_steps):
            pos = inverse_kinematics([x_s[i], y_s[i], z_s[i]], q)

            try:
                move(pos, timeout=timeout)
            except Exception as e:
                print("error moving robot: ", e)
    elif method == 'curved':
        # TODO
        pass
    else:
        # Direct
        pos = inverse_kinematics([x, y, z], q)
        move(pos, timeout=timeout)

    curr_position = [x, y, z]

def hover_above(x,y,z, method='linear'):
    _move(x,y,z+HOVER_FACTOR, method=method)
    rate = rospy.Rate(100)
    rate.sleep()

def move_to(x,y,z, method='linear'):
    _move(x,y,z, method=method)

def dip_brush_in_water():
    hover_above(WATER_POSITION[0],WATER_POSITION[1],WATER_POSITION[2])
    move_to(WATER_POSITION[0],WATER_POSITION[1],WATER_POSITION[2])
    rate = rospy.Rate(100)
    for i in range(5):
        noise = np.random.randn(2)*0.01
        move_to(WATER_POSITION[0]+noise[0],WATER_POSITION[1]+noise[1],WATER_POSITION[2], method='direct')
        rate.sleep()
    hover_above(WATER_POSITION[0],WATER_POSITION[1],WATER_POSITION[2])

def rub_brush_on_rag():
    hover_above(RAG_POSTITION[0],RAG_POSTITION[1],RAG_POSTITION[2])
    move_to(RAG_POSTITION[0],RAG_POSTITION[1],RAG_POSTITION[2])
    for i in range(5):
        noise = np.random.randn(2)*0.01
        move_to(RAG_POSTITION[0]+noise[0],RAG_POSTITION[1]+noise[1],RAG_POSTITION[2], method='direct')
    hover_above(RAG_POSTITION[0],RAG_POSTITION[1],RAG_POSTITION[2])

def clean_paint_brush():
    dip_brush_in_water()
    rub_brush_on_rag()

def get_paint(paint_index):
    x_offset = PAINT_DIFFERENCE * np.floor(paint_index/6)
    y_offset = PAINT_DIFFERENCE * (paint_index%6)

    x = PALLETTE_POSITION[0] + x_offset
    y = PALLETTE_POSITION[1] + y_offset
    z = PALLETTE_POSITION[2]

    hover_above(x,y,z)
    for i in range(3):
        noise = np.random.randn(2)*0.005
        move_to(x+noise[0],y+noise[1],z, method='direct')
    hover_above(x,y,z)

def paint_path(path):
    """
    args:
        path np.array([2,n]) : x,y coordinates of a path of a brush stroke
    """
    pass

def paint_bezier_curve(p0,p1,p2, step_size=.005):
    p0 = canvas_to_global_coordinates(p0[0], p0[1], TABLE_Z)
    p1 = canvas_to_global_coordinates(p1[0], p1[1], TABLE_Z)
    p2 = canvas_to_global_coordinates(p2[0], p2[1], TABLE_Z)

    stroke_length = ((p1[0]-p0[0])**2 + (p1[1] - p0[1])**2)**.5 \
            + ((p2[0]-p1[0])**2 + (p2[1] - p1[1])**2)**.5
    # print('stroke_length', stroke_length)
    n = max(2, int(stroke_length/step_size))
    # print('n',n)

    hover_above(p0[0], p0[1], TABLE_Z)
    for t in np.linspace(0,1,n):
        x = (1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0]
        y = (1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1]
        move_to(x,y,TABLE_Z, method='direct')
    hover_above(p2[0],p2[1],TABLE_Z)

def set_brush_height():
    # set the robot arm at a location on the canvas and
    # wait for the user to attach the brush

    p = canvas_to_global_coordinates(.5, .5, TABLE_Z)
    hover_above(p[0],p[1],TABLE_Z)
    move_to(p[0],p[1],TABLE_Z, method='direct')

    raw_input('Attach the paint brush now. Press enter to continue:')

    hover_above(p[0],p[1],TABLE_Z)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sawyer Painter')

    # parser.add_argument("file", type=str,
    #     help='Path CSV instructions.')
    args = parser.parse_args()
    # args.file = '/home/peterschaldenbrand/paint/AniPainter/animation_instructions/actions.csv'
    args.file = '/home/peterschaldenbrand/Downloads/actions.csv'

    # Set which robot we talkin to
    # os.environ['ROS_MASTER_URI'] = "http://localhost:11311"
    # os.environ['ROS_HOSTNAME'] = "localhost"

    rospy.init_node("painting")

    good_morning_robot()


    # Initial spot
    _move(0,0.5,TABLE_Z+0.1, timeout=20, method="direct")

    # Allow user a chance to attach the brush
    set_brush_height()

    instructions = np.loadtxt(args.file, delimiter=',')

    curr_color = -1
    for instr in tqdm(instructions[:,:]):
        p0, p1, p2 = instr[0:2], instr[2:4], instr[4:6]
        color = instr[12]
        radius = instr[6]
        if color != curr_color:
            clean_paint_brush()
        get_paint(color)
        paint_bezier_curve(p0, p1, p2)
        curr_color = color

    # for i in range(12):
    #     get_paint(i)
    # get_paint(0)
    # get_paint(5)
    # get_paint(6)
    # get_paint(11)

    # clean_paint_brush()

    # paint_bezier_curve((0,0),(0,0),(0,0))
    # paint_bezier_curve((0,1),(0,1),(0,1)) # top-left
    # paint_bezier_curve((1,0),(1,0),(1,0)) # bottom-right
    # paint_bezier_curve((1,1),(1,1),(1,1))


    _move(0,0.5,TABLE_Z+0.1, timeout=20, method="direct")


    good_night_robot()