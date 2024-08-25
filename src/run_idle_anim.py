# run the idle_anim function from Xarm robot class
# duration of anim is approx 30 secs 

from robot import *
# from options import *

def main():
    # opt = Options()
    # opt.gather_options()

    # change this ip address manually, of type string
    robot = XArm('192.168.1.168', debug=True)
    # robot.good_morning_robot()
    # robot.go_to_cartesian_pose([[0.002, 0.2,0.2],], [[180, 0, 0, 1],], motion=True)  
    
    robot.idle_anim()
    
     
    robot.good_night_robot()

if __name__ == '__main__':
    
    main()