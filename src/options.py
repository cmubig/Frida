
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import argparse 
import os
import json

# Based off of Jun-Yan Zhu's Cyclegan implementation's options

class Options(object):
    def __init__(self):
        self.initialized = False
        self.opt = {}

        self.CANVAS_WIDTH_PIX  = None # set these after taking a picture
        self.CANVAS_HEIGHT_PIX = None

        self.HOVER_FACTOR = 0.1


    def initialize(self, parser):
        parser.add_argument("--robot", type=str,
            default='franka', help='Which robot to use "franka" "xarm" or "sawyer"')
        parser.add_argument("--xarm_ip", type=str, default='192.168.1.176', help='IP address of XArm.')

        parser.add_argument('--use_cache', action='store_true')
        parser.add_argument("--materials_json", type=str, 
            default='../materials.json', help='IP address of XArm.')

        parser.add_argument('--dont_plan', action='store_true', help='Use saved plan from last run')
        parser.add_argument('--max_height', default=256, type=int, help='How much to downscale canvas for simulated environment')
        parser.add_argument('--num_papers', default=4, type=int, help='How papers of strokes to paint for stroke modelling data.')
        # parser.add_argument('--use_cached_colors', action='store_true')
        parser.add_argument('--n_colors', default=12, type=int, help='Number of colors of paint to use')

        parser.add_argument("--cache_dir", type=str,
            default='./caches/cache_6_6_cvpr', help='Where to store cached files.')
        parser.add_argument("--tensorboard_dir", type=str,
            default='./painting_log', help='Where to write tensorboard log to.')
        parser.add_argument("--global_it", type=int,
            default=0, help='Picking up where it left off.')

        parser.add_argument('--brush_length', type=float, default=None)


        parser.add_argument('--num_strokes', type=int, default=400)
        parser.add_argument('--num_adaptations', type=int, default=4)
        parser.add_argument('--how_often_to_get_paint', type=int, default=4)
        parser.add_argument('--n_stroke_models', type=int, default=1)
        parser.add_argument('--fill_weight', type=float, default=0.0)

        parser.add_argument('--adaptive', action='store_true')
        parser.add_argument('--generate_whole_plan', action='store_true')
        parser.add_argument('--strokes_before_adapting', type=int, default=100)
        parser.add_argument('--remove_prop', type=float, default=0.8, help="Proportion of strokes to remove when adapting")

        parser.add_argument('--adapt_optim_iter', type=int, default=30)

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


        # CoFRIDA Parameters
        parser.add_argument("--controlnet_dataset", type=str,
            default="laion/laion-art", help='A dataset for training controlnet')
        parser.add_argument("--controlnet_dataset_addition", type=str,
            default="fusing/instructpix2pix-1000-samples", help='A dataset for training controlnet to add material')
        parser.add_argument("--output_parent_dir", type=str,
            help='Where to save the data. Can continue if partially complete.')
        parser.add_argument("--removal_method", type=str,
            default='random',
            help='how to make partial sketchs. [random|salience]')
        parser.add_argument("--max_images", type=int,
            default=20000, help='A dataset for training controlnet')
        parser.add_argument("--max_strokes_added", type=int,
            default=200, help='Final amount of strokes')
        parser.add_argument("--min_strokes_added", type=int,
            default=100, help='Amount of strokes in the partial sketch')
        parser.add_argument("--num_images_to_consider_for_simplicity", type=int,
            default=3, help='Load this many images and take the one with fewest edges for simplicity.')
        parser.add_argument("--n_iters", type=int,
            default=300, help='Number of optimization iterations.')
        parser.add_argument("--colors", type=str,
            default=None, help='Specify a fixed palette of paint colors.')
        parser.add_argument("--turn_takes", type=int,
            default=0, help='How many turns for generating pix2pix training data.')
        parser.add_argument("--codraw_metric_data_dir", type=str,
            default=None, help='Where to store evaluation data.')
        parser.add_argument("--codraw_eval_setting", type=str,
            default=None, help='[same_text_fill_in,same_text_add_detail_different_text,add_background,something_from_nothing]')
        
        parser.add_argument("--cofrida_model", type=str,
            default=None, help='path to pre-trained instruct-pix2pix CoFRIDA model')
        
        ### Argument is not used, but is allowed for flask compatability ###
        parser.add_argument("--app", type=str, nargs='*',
            default="app run", help='Argument is not used, but is allowed for flask compatability')

        return parser 

    def gather_options(self):
        if not self.initialized:
            self.parser = argparse.ArgumentParser(description="FRIDA Robot Painter")
            self.parser = self.initialize(self.parser)

        self.opt = vars(self.parser.parse_args())

        if not os.path.exists(self.materials_json):
            raise Exception('Cannot find materials JSON file: ' + self.materials_json)
        with open(self.materials_json, 'r') as f:
            materials = json.load(f)
        
        self.opt = {**self.opt, **materials}

        thresh = 0.001 # 1cm tolerance on overshooting the canvas
        self.Y_CANVAS_MAX = self.CANVAS_POSITION[1] + self.CANVAS_HEIGHT_M + thresh
        self.Y_CANVAS_MIN = self.CANVAS_POSITION[1] - thresh
        self.X_CANVAS_MAX = self.CANVAS_POSITION[0] + self.CANVAS_WIDTH_M/2 + thresh
        self.X_CANVAS_MIN = self.CANVAS_POSITION[0] - self.CANVAS_WIDTH_M/2 - thresh

    def __getattr__(self, attr_name):
        return self.opt[attr_name]