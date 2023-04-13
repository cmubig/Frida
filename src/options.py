
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import argparse 
import math

# Based off of Jun-Yan Zhu's Cyclegan implementation's options

class Options(object):
    def __init__(self):
        self.initialized = False
        self.opt = {}

        self.INIT_TABLE_Z = 0.3

        # Dimensions of canvas in meters
        # CANVAS_WIDTH  = 0.3047 # 12"
        # CANVAS_HEIGHT = 0.2285 # 9"
        # self.CANVAS_WIDTH  = 0.254 -0.005# 10"
        # self.CANVAS_HEIGHT = 0.2032 -0.005# 8"
        self.CANVAS_WIDTH  = 0.3556 -0.001# 14"
        self.CANVAS_HEIGHT = 0.2794 -0.001# 11"
        # self.CANVAS_WIDTH  = 0.5080 # 20"
        # self.CANVAS_HEIGHT = 0.4064 # 16"


        self.CANVAS_WIDTH_PIX  = None # set these after taking a picture
        self.CANVAS_HEIGHT_PIX = None

        # X,Y of canvas wrt to robot center (global coordinates)
        # self.CANVAS_POSITION = (0,.5) 
        self.CANVAS_POSITION = (0.34, .29) # 14x11"
        # self.CANVAS_POSITION = (0+0.0762, .5-.04) # 20x16"
        # self.CANVAS_POSITION = (0+0.0762-0.12, .5-.04-0.0635-0.06+.202)# 10x8"

        thresh = 0.001 # 1cm tolerance on overshooting the canvas
        self.Y_CANVAS_MAX = self.CANVAS_POSITION[1] + self.CANVAS_HEIGHT + thresh
        self.Y_CANVAS_MIN = self.CANVAS_POSITION[1] - thresh
        self.X_CANVAS_MAX = self.CANVAS_POSITION[0] + self.CANVAS_WIDTH/2 + thresh
        self.X_CANVAS_MIN = self.CANVAS_POSITION[0] - self.CANVAS_WIDTH/2 - thresh

        """ How many times in a row can you paint with the same color before needing more paint """
        self.GET_PAINT_FREQ = 3

        self.HOVER_FACTOR = 0.1


        # Number of cells to paint in x and y directions
        self.cells_x, self.cells_y = 4, 5

        # Dimensions of the cells in Meters
        #cell_dim = (0.0254, 0.0508) #h/w in meters. 1"x2"
        self.cell_dim_y, self.cell_dim_x = self.CANVAS_HEIGHT / self.cells_y, self.CANVAS_WIDTH / self.cells_x

        # The brush stroke starts halfway down and 20% over from left edge of cell
        self.down = 0.5 * self.cell_dim_y
        self.over = 0.2 * self.cell_dim_x


        self.MAX_ALPHA = 0#math.pi / 18.

        self.MIN_STROKE_LENGTH = 0.005#0.001
        self.MAX_STROKE_LENGTH = 0.06#0.03#0.04#0.06
        self.MIN_STROKE_Z = 0.1#0.01
        self.MAX_BEND = 0.02 #2cm


        self.MIN_FILL_IN_LENGTH = 0.02
        self.MAX_FILL_IN_LENGTH = 0.06
        self.MIN_FILL_IN_HEIGHT = 0.02
        self.MAX_FILL_IN_HEIGHT = 0.04
        self.num_fill_in_papers = 1

    def initialize(self, parser):
        parser.add_argument('--use_cache', action='store_true')
        parser.add_argument('--dont_plan', action='store_true', help='Use saved plan from last run')
        # parser.add_argument('--discrete', action='store_true')
        parser.add_argument('--diffvg', action='store_true')
        parser.add_argument('--max_height', default=256, type=int, help='How much to downscale canvas for simulated environment')
        parser.add_argument('--num_papers', default=4, type=int, help='How papers of strokes to paint for stroke modelling data.')
        # parser.add_argument('--just_fine', action='store_true', help="Only plan for smallest strokes")
        # parser.add_argument('--use_cached_colors', action='store_true')
        parser.add_argument('--n_colors', default=12, type=int, help='Number of colors of paint to use')
        # parser.add_argument("--file", type=str,
        #     default='/home/peterschaldenbrand/Downloads/david_lynch.csv',
        #     help='Path CSV instructions.')
        # parser.add_argument('--target', type=str, default=None)
        # parser.add_argument('--prompt', type=str, default=None)
        parser.add_argument("--cache_dir", type=str,
            default='/tmp', help='Where to store cached files.')
        parser.add_argument("--tensorboard_dir", type=str,
            default='./painting', help='Where to write tensorboard log to.')
        parser.add_argument("--global_it", type=int,
            default=0, help='Picking up where it left off.')

        parser.add_argument('--brush_length', type=float, default=None)


        parser.add_argument('--num_strokes', type=int, default=400)
        parser.add_argument('--n_stroke_models', type=int, default=5)
        parser.add_argument('--fill_weight', type=float, default=0.5)

        parser.add_argument('--adaptive', action='store_true')
        parser.add_argument('--generate_whole_plan', action='store_true')
        parser.add_argument('--strokes_before_adapting', type=int, default=100)
        parser.add_argument('--remove_prop', type=float, default=0.8, help="Proportion of strokes to remove when adapting")

        parser.add_argument('--adapt_optim_iter', type=int, default=30)

        # parser.add_argument('--type', default='cubic_bezier', type=str, help='Type of instructions: [cubic_bezier | bezier]')
        # parser.add_argument('--continue_ind', default=0, type=int, help='Instruction to start from. Default 0.')
        parser.add_argument('--simulate', action='store_true')



        parser.add_argument('--objective', nargs='*', type=str, help='text|style|clip_conv_loss|l2|clip_fc_loss')
        parser.add_argument('--objective_data', nargs='*', type=str)
        parser.add_argument('--objective_weight', nargs='*', type=float, default=1.0)
        parser.add_argument('--optim_iter', type=int, default=150)
        parser.add_argument('--lr_multiplier', type=float, default=0.2)
        parser.add_argument('--init_lr', type=float, default=3e-2, help="learning rate for initial objective")

        parser.add_argument('--init_objective', nargs='*', type=str, help='text|style|clip_conv_loss|l2|clip_fc_loss')
        parser.add_argument('--init_objective_data', nargs='*', type=str)
        parser.add_argument('--init_objective_weight', nargs='*', type=float, default=1.0)
        parser.add_argument('--init_optim_iter', type=int, default=40)
        parser.add_argument('--n_inits', type=int, default=5, help='Number of times to try different initializations')

        parser.add_argument('--intermediate_optim_iter', type=int, default=40)
        parser.add_argument('--use_colors_from', type=str, default=None, help="Get the colors from this image. \
                None if you want the colors to come from the optimized painting.")

        parser.add_argument('--num_augs', type=int, default=30)
        parser.add_argument('--bin_size', type=int, default=3000)

        parser.add_argument('--plan_gif_dir', type=str, default='/home/frida/Videos/frida/')
        parser.add_argument('--log_frequency', type=int, default=5)

        parser.add_argument("--output_dir", type=str, default="../outputs/", help='Where to write output to.')

        parser.add_argument('--dont_retrain_stroke_model', action='store_true')

        parser.add_argument('--pretrain_stroke_model', type=str, default=None,
                            help="give a path to a folder with pretrained stroke models")



        parser.add_argument('--ink', action='store_true')
        parser.add_argument('--paint_from_image', action='store_true')
        parser.add_argument("--caption", type=str,
            default=None, help='A caption of the image you\'re trying to paint')

        # parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")

        return parser 

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(description="FRIDA Robot Painter")
            parser = self.initialize(parser)

        self.opt = vars(parser.parse_args())

        if not self.simulate and self.brush_length is None:
            print('Must specify --brush_length cmd line param. Measure the brush length.')

        if self.ink:
            self.MAX_STROKE_LENGTH = 0.02
            self.MAX_BEND = 0.01 #1cm


    def __getattr__(self, attr_name):
        return self.opt[attr_name]