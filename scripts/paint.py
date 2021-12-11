#! /usr/bin/env python
import os
import time
import sys
import rospy
import numpy as np
from tqdm import tqdm
import scipy.special

from paint_utils import *
from painter import Painter
from robot import *



def load_instructions(fn):
    '''
    Load instructions into a list of lists
    '''

    instructions = []
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 1:
                instructions.append(np.array([float(s) for s in line.split(',')]))
    return instructions

limb = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sawyer Painter')

    parser.add_argument("--file", type=str,
        default='/home/peterschaldenbrand/Downloads/vangogh.csv',
        help='Path CSV instructions.')

    parser.add_argument('--type', default='cubic_bezier', type=str, help='Type of instructions: [cubic_bezier | bezier]')
    parser.add_argument('--continue_ind', default=0, type=int, help='Instruction to start from. Default 0.')

    args = parser.parse_args()
    # args.file = '/home/peterschaldenbrand/paint/AniPainter/animation_instructions/actions.csv'
    # args.file = '/home/peterschaldenbrand/Downloads/actions.csv'

    # Set which robot we talkin to
    # os.environ['ROS_MASTER_URI'] = "http://localhost:11311"
    # os.environ['ROS_HOSTNAME'] = "localhost"

    painter = Painter(robot="sawyer")

    global limb
    limb = intera_interface.Limb(synchronous_pub=False)


    painter.robot.display_frida()


    instructions = load_instructions(args.file)

    # Allow user a chance to attach the brush
    painter.set_brush_height()


    curr_color = -1
    since_got_paint = 0
    
    for instr in tqdm(instructions[args.continue_ind:]):
        if args.type == 'cubic_bezier':
            # Full path
            path = instr[2:]
            path = np.reshape(path, (len(path)//2, 2))
            color = instr[1]
            radius = instr[0]
            if color != curr_color:
                painter.clean_paint_brush()

            if color != curr_color or since_got_paint == GET_PAINT_FREQ:
                painter.get_paint(color)
                since_got_paint = 0
            since_got_paint += 1

            painter.paint_cubic_bezier(path)
            curr_color = color
        else:
            # Bezier Curve
            p0, p1, p2 = instr[0:2], instr[2:4], instr[4:6]
            color = instr[12]
            radius = instr[6]
            if color != curr_color:
                painter.clean_paint_brush()

            if color != curr_color or since_got_paint == GET_PAINT_FREQ:
                painter.get_paint(color)
                since_got_paint = 0

            painter.paint_quadratic_bezier(p0, p1, p2)
            curr_color = color

    painter.clean_paint_brush()

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


    painter.to_neutral()


    painter.robot.good_night_robot()