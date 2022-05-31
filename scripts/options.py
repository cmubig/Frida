
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import argparse 

# Based off of Jun-Yan Zhu's Cyclegan implementation's options

class Options(object):
    def __init__(self):
        self.initialized = False
        self.opt = {}

        self.INIT_TABLE_Z = 0.1

        # Dimensions of canvas in meters
        # CANVAS_WIDTH  = 0.3047 # 12"
        # CANVAS_HEIGHT = 0.2285 # 9"
        self.CANVAS_WIDTH  = 0.254 -0.005# 10"
        self.CANVAS_HEIGHT = 0.2032 -0.005# 8"


        self.CANVAS_WIDTH_PIX  = None # set these after taking a picture
        self.CANVAS_HEIGHT_PIX = None

        # X,Y of canvas wrt to robot center (global coordinates)
        self.CANVAS_POSITION = (0,.5) 

        """ How many times in a row can you paint with the same color before needing more paint """
        self.GET_PAINT_FREQ = 3

        self.HOVER_FACTOR = 0.1


        # Number of cells to paint in x and y directions
        self.cells_x, self.cells_y = 3, 4

        # Dimensions of the cells in Meters
        #cell_dim = (0.0254, 0.0508) #h/w in meters. 1"x2"
        self.cell_dim_y, self.cell_dim_x = self.CANVAS_HEIGHT / self.cells_y, self.CANVAS_WIDTH / self.cells_x

        # The brush stroke starts halfway down and 20% over from left edge of cell
        self.down = 0.5 * self.cell_dim_y
        self.over = 0.2 * self.cell_dim_x

    def initialize(self, parser):
        parser.add_argument('--use_cache', action='store_true')
        parser.add_argument('--n_colors', default=6, type=int, help='Number of colors of paint to use')
        parser.add_argument("--file", type=str,
            default='/home/peterschaldenbrand/Downloads/david_lynch.csv',
            help='Path CSV instructions.')
        parser.add_argument('--target', type=str, default='/home/frida/Downloads/cutoutjon.jpg')
        parser.add_argument("--cache_dir", type=str,
            default='/tmp', help='Where to store cached files.')
        parser.add_argument("--tensorboard_dir", type=str,
            default='./log', help='Where to write tensorboard log to.')
        parser.add_argument("--global_it", type=int,
            default=0, help='Picking up where it left off.')

        # parser.add_argument('--type', default='cubic_bezier', type=str, help='Type of instructions: [cubic_bezier | bezier]')
        # parser.add_argument('--continue_ind', default=0, type=int, help='Instruction to start from. Default 0.')
        parser.add_argument('--simulate', action='store_true')
        return parser 

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(description="Sawyer Painter")
            parser = self.initialize(parser)

        self.opt = vars(parser.parse_args())


    def __getattr__(self, attr_name):
        return self.opt[attr_name]