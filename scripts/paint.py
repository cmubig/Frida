#! /usr/bin/env python
import os
import time
import sys
import rospy
import numpy as np
from tqdm import tqdm
import scipy.special

from paint_utils import *

q = np.array([0.704020578925, 0.710172716916,0.00244101361829,0.00194372088834])


TABLE_Z = -0.099

CANVAS_POSITION = (0,.5,TABLE_Z)
CANVAS_WIDTH = .2
CANVAS_HEIGHT = .2

WATER_POSITION = (-.3,.6,TABLE_Z)
RAG_POSTITION = (-.3,.3,TABLE_Z)

PALLETTE_POSITION = (.3,.5,TABLE_Z)
PAINT_DIFFERENCE = 0.03976

GET_PAINT_FREQ = 2

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

def display_frida():
    import rospkg
    rospack = rospkg.RosPack()
    # get the file path for rospy_tutorials
    ros_dir = rospack.get_path('paint')
    display_image(os.path.join(str(ros_dir), 'scripts', 'frida.jpg'))

def _move(x, y, z, timeout=20, method='linear', step_size=.2, speed=0.1):
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
                move(pos, timeout=timeout, speed=speed)
            except Exception as e:
                print("error moving robot: ", e)
    elif method == 'curved':
        # TODO
        pass
    else:
        # Direct
        pos = inverse_kinematics([x, y, z], q)
        move(pos, timeout=timeout, speed=speed)

    curr_position = [x, y, z]

def hover_above(x,y,z, method='linear'):
    _move(x,y,z+HOVER_FACTOR, method=method, speed=0.4)
    # rate = rospy.Rate(100)
    # rate.sleep()

def move_to(x,y,z, method='linear', speed=0.05):
    _move(x,y,z, method=method, speed=speed)

def dip_brush_in_water():
    hover_above(WATER_POSITION[0],WATER_POSITION[1],WATER_POSITION[2])
    move_to(WATER_POSITION[0],WATER_POSITION[1],WATER_POSITION[2], speed=0.2)
    rate = rospy.Rate(100)
    for i in range(5):
        noise = np.clip(np.random.randn(2)*0.01, a_min=-.02, a_max=0.02)
        move_to(WATER_POSITION[0]+noise[0],WATER_POSITION[1]+noise[1],WATER_POSITION[2], method='direct')
        rate.sleep()
    hover_above(WATER_POSITION[0],WATER_POSITION[1],WATER_POSITION[2])

def rub_brush_on_rag():
    hover_above(RAG_POSTITION[0],RAG_POSTITION[1],RAG_POSTITION[2])
    move_to(RAG_POSTITION[0],RAG_POSTITION[1],RAG_POSTITION[2], speed=0.2)
    for i in range(5):
        noise = np.clip(np.random.randn(2)*0.02, a_min=-.03, a_max=0.03)
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
    move_to(x,y,z + 0.02, speed=0.2)
    for i in range(3):
        noise = np.clip(np.random.randn(2)*0.0025, a_min=-.005, a_max=0.005)
        move_to(x+noise[0],y+noise[1],z, method='direct')
    move_to(x,y,z + 0.02, speed=0.2)
    hover_above(x,y,z)

def paint_path(path, step_size=.005):
    """
    args:
        path np.array([n,2]) : x,y coordinates of a path of a brush stroke
    """
    # p0 = canvas_to_global_coordinates(path[0,0], path[0,1], TABLE_Z)
    # hover_above(p0[0], p0[1], TABLE_Z)
    # move_to(p0[0], p0[1], TABLE_Z + 0.02, speed=0.2)

    # for i in range(len(path)):
    #     p = canvas_to_global_coordinates(path[i,0], path[i,1], TABLE_Z)
    #     move_to(p[0], p[1], TABLE_Z, method='linear')

    # pn = canvas_to_global_coordinates(path[-1,0], path[-1,1], TABLE_Z)
    # move_to(pn[0], pn[1], TABLE_Z + 0.02, speed=0.2)
    # hover_above(pn[0], pn[1], TABLE_Z)
    ###################################################################
    p0 = canvas_to_global_coordinates(path[0,0], path[0,1], TABLE_Z)
    hover_above(p0[0], p0[1], TABLE_Z)
    move_to(p0[0], p0[1], TABLE_Z + 0.02, speed=0.2)
    p3 = None

    for i in range(1, len(path)-1, 3):
        p1 = canvas_to_global_coordinates(path[i+0,0], path[i+0,1], TABLE_Z)
        p2 = canvas_to_global_coordinates(path[i+1,0], path[i+1,1], TABLE_Z)
        p3 = canvas_to_global_coordinates(path[i+2,0], path[i+2,1], TABLE_Z)

        stroke_length = ((p3[0]-p0[0])**2 + (p3[1] - p0[1])**2)**.5
        n = max(2, int(stroke_length/step_size))
        n=10
        for t in np.linspace(0,1,n):
            x = (1-t)**3 * p0[0] \
                  + 3*(1-t)**2*t*p1[0] \
                  + 3*(1-t)*t**2*p2[0] \
                  + t**3*p3[0]
            y = (1-t)**3 * p0[1] \
                  + 3*(1-t)**2*t*p1[1] \
                  + 3*(1-t)*t**2*p2[1] \
                  + t**3*p3[1]
            move_to(x, y, TABLE_Z, method='direct', speed=0.03)
        p0 = p3

    pn = canvas_to_global_coordinates(path[-1,0], path[-1,1], TABLE_Z)
    move_to(pn[0], pn[1], TABLE_Z + 0.02, speed=0.2)
    hover_above(pn[0], pn[1], TABLE_Z)
    #############################################################

    # p0 = canvas_to_global_coordinates(path[0,0], path[0,1], TABLE_Z)
    # hover_above(p0[0], p0[1], TABLE_Z)
    # move_to(p0[0], p0[1], TABLE_Z + 0.02, speed=0.2)

    # n = len(path) *1.0
    # xs, ys = [], []
    # for t in np.linspace(0.01,0.99,40):
    #     numer = 0
    #     denom = 0
    #     w_i = 1./n
    #     for i in range(len(path)):
    #         binom = scipy.special.binom(n, i)
    #         numer += binom * t**i * (1.-t)**(n-i) * path[i] * w_i
    #         denom += binom * t**i * (1.-t)**(n-i) * w_i

    #         # print('binom', binom)
    #         # print('n', n)
    #         # print('i', i)
    #         # print('t', t)
    #         # print('w_i', w_i)
    #         # print('binom * t**i', binom * t**i)
    #         # print('(1-t)**(n-i)', (1-t)**(n-i))
    #         # print('numer', numer)
    #         # print('denom', denom)
    #     b_t = numer/denom
    #     x = b_t[0]
    #     y = b_t[1]
    #     coords = canvas_to_global_coordinates(x, y, TABLE_Z)
    #     # print('coords', coords, x, y)
    #     move_to(coords[0], coords[1], TABLE_Z, method='direct', speed=0.03)

    # pn = canvas_to_global_coordinates(path[-1,0], path[-1,1], TABLE_Z)
    # move_to(pn[0], pn[1], TABLE_Z + 0.02, speed=0.2)
    # hover_above(pn[0], pn[1], TABLE_Z)

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
    move_to(p0[0], p0[1], TABLE_Z + 0.02, speed=0.2)
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

    parser.add_argument("--file", type=str,
        default='/home/peterschaldenbrand/Downloads/abby_traced_actions.csv',
        help='Path CSV instructions.')

    parser.add_argument('--type', default='full_path', type=str, help='Type of instructions: [full_path | bezier]')

    args = parser.parse_args()
    # args.file = '/home/peterschaldenbrand/paint/AniPainter/animation_instructions/actions.csv'
    # args.file = '/home/peterschaldenbrand/Downloads/actions.csv'

    # Set which robot we talkin to
    # os.environ['ROS_MASTER_URI'] = "http://localhost:11311"
    # os.environ['ROS_HOSTNAME'] = "localhost"

    rospy.init_node("painting")

    good_morning_robot()

    display_frida()


    # Initial spot
    _move(0,0.5,TABLE_Z+0.1, timeout=20, method="direct")

    # Allow user a chance to attach the brush
    set_brush_height()

    instructions = []#np.loadtxt(args.file, delimiter=',')
    with open(args.file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 1:
                instructions.append(np.array([float(s) for s in line.split(',')]))

    curr_color = -1
    since_got_paint = 0
    for instr in tqdm(instructions[:]):
        if args.type == 'full_path':
            # Full path
            path = instr[2:]
            path = np.reshape(path, (len(path)//2, 2))
            color = instr[1]
            radius = instr[0]
            if color != curr_color:
                clean_paint_brush()

            if color != curr_color or since_got_paint == GET_PAINT_FREQ:
                get_paint(color)
                since_got_paint = 0
            since_got_paint += 1

            paint_path(path)
            curr_color = color
        else:
            # Bezier Curve
            p0, p1, p2 = instr[0:2], instr[2:4], instr[4:6]
            color = instr[12]
            radius = instr[6]
            if color != curr_color:
                clean_paint_brush()
            get_paint(color)
            paint_bezier_curve(p0, p1, p2)
            curr_color = color
    clean_paint_brush()

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