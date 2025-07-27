"""
键盘控制节点
发布消息到main节点
1. 手动设置运行阶段 
    0 不切换 
    1 过门
    2 巡线
    3 作业
    4 夹球
    5 上浮
2. 手动控制机器人运行 wasdqeik
3. 开启自动运行 g

"""
import rospy
from auv_control.msg import Key
import sys
import termios
import tty
import atexit
"""
Key.msg
bool run
uint8 mode
AUVData force

"""

"""
AUVMotor.msg
int16 TX 
int16 TY
int16 TZ
int16 MX 
int16 MY
int16 MZ
"""

class KeyboardControlNode:
    def __init__(self):
        rospy.init_node('keyboard_control_node')
        self.pub = rospy.Publisher('/auv_keyboard', Key, queue_size=10)
        self.force = rospy.get_param('~force', 50) # 默认力矩大小 50
        self.torque = rospy.get_param('~torque', 50) # 默认转矩大小 50
        self.rate = rospy.Rate(10)
        self.running = False # 默认模式为0（待机）
        self.led_red = False  # 红色LED状态
        self.led_green = False  # 绿色LED状态
        self.servo = False  # 舵机状态
        rospy.loginfo("Control AUV with keyboard:")
        rospy.loginfo("w/s: forward/backward")
        rospy.loginfo("a/d: turn left/right")
        rospy.loginfo("q/e: left/right")
        rospy.loginfo("i/k: up/down")
        rospy.loginfo("0-5: set run stage")
        rospy.loginfo("g: start auto run")
        rospy.loginfo("CTRL+C to quit")


    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def run(self):
        while not rospy.is_shutdown():
            key = self.get_key()
            msg = Key()
            # Reset all values
            msg.run = self.running
            msg.mode = 0
            msg.control.force.TX = 0
            msg.control.force.TY = 0
            msg.control.force.TZ = 0
            msg.control.force.MX = 0
            msg.control.force.MY = 0
            msg.control.force.MZ = 0
            if key == 'g':
                self.running = not self.running
                msg.run = self.running
            if key == 'w':
                msg.control.force.TX = self.force
            elif key == 's':
                msg.control.force.TX = -self.force
            elif key == 'a':
                msg.control.force.MZ = self.torque
            elif key == 'd':
                msg.control.force.MZ = -self.torque
            elif key == 'q':
                msg.control.force.TY = self.force
            elif key == 'e':
                msg.control.force.TY = -self.force
            elif key == 'i':
                msg.control.force.TZ = self.force
            elif key == 'k':
                msg.control.force.TZ = -self.force
            elif key == 'z':
                msg.control.enable = True  # 启用外设控制
                self.led_red = not getattr(self, 'led_red', False)  # 切换红色LED状态
                msg.control.led_red = not self.led_red  # 切换红色LED状态
            elif key == 'x':
                msg.control.enable = True
                self.led_green = not getattr(self, 'led_green', False)  # 切换绿色LED状态
                msg.control.led_green = not self.led_green  # 切换绿色LED状态
            elif key == 'c':
                msg.control.enable = True
                self.servo = not getattr(self, 'servo', False)  # 切换舵机状态
                msg.control.servo = not self.servo  # 切换舵机状态
            elif key in '012345':
                msg.mode = int(key)
            elif key == '\x03':  # CTRL+C
                break
            self.pub.publish(msg)
            self.rate.sleep()

def restore_terminal():
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, termios.tcgetattr(fd))

if __name__ == '__main__':
    atexit.register(restore_terminal)
    try:
        node = KeyboardControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass