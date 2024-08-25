# run the signing function from Xarm robot class
# signs an F and illegible scribles afer

from robot import *
from options import *

def main():
    opt = Options()
    opt.gather_options()
    table_z = 0.12700999999999997
    # change this ip address manually, of type string
    robot = XArm('192.168.1.168', debug=True)
    robot.good_morning_robot()
    # robot.go_to_cartesian_pose([[0.002, 0.2,0.2],], [[180, 0, 0, 1],], motion=True)  
    robot.sign_on_canvas(opt.CANVAS_HEIGHT_M, opt.CANVAS_WIDTH_M,
                         opt.X_CANVAS_MAX, opt.Y_CANVAS_MIN,
                         table_z)
    robot.good_night_robot()

if __name__ == '__main__':
    main()