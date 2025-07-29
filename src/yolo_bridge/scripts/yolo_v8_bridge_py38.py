#!/home/xhy/xhy_env/bin/python
import rospy
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from stereo_depth.msg import BoundingBox


class YOLOv8Detector:
    def __init__(self, param):
        rospy.init_node("yolov8_detector", anonymous=True)
        
        # self.DetectMode = rospy.get_param('~detect_mode', 2)  
        self.DetectMode = param.detect_mode  # 从命令行参数获取检测模式
        self.top_k = param.top_k  # 从命令行参数获取 top_k
        self.visualization = param.visualization  # 从命令行参数获取是否可视化
        
        
        if self.DetectMode == 1:
            self.model = YOLO("/home/xhy/catkin_ws/models/shapes_model0719.pt")
        elif self.DetectMode == 2:
            self.model = YOLO("/home/xhy/catkin_ws/models/holes_model0719.pt")
        elif self.DetectMode == 3:
            self.model = YOLO("/home/xhy/catkin_ws/models/balls_model0725.pt")
        else:
            rospy.logwarn("DetectMode error: %s, select the shapes model by default", str(self.DetectMode))
            self.model = YOLO("/home/xhy/catkin_ws/models/shapes_model0719.pt")
            
        rospy.loginfo("DetectMode: %d", self.DetectMode)
            
        self.bridge = CvBridge()

        rospy.Subscriber("/left/image_raw", Image, self.image_callback)
        if self.DetectMode in (1, 3):
            self.center_pub = rospy.Publisher("/yolov8/target_center", PointStamped, queue_size=1)
        else:
            self.center_pub = rospy.Publisher("/yolov8/target_bbox", BoundingBox, queue_size=1)
        
        # 控制推断频率
        self.last_infer_time = rospy.Time.now()
        self.infer_interval = rospy.Duration(1)  # 单位秒 

        
        rospy.loginfo("YOLOv8 Detector Node Started")

    def image_callback(self, msg):
        
        now = rospy.Time.now()
        if now - self.last_infer_time < self.infer_interval:
            return  # 距离上次推理太近，跳过此次图像

        self.last_infer_time = now
        
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", str(e))
            return

        results = self.model(img)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names
        
        # print("come in!!")

        for i, box in enumerate(boxes):
            conf = confs[i]
            cls_id = int(classes[i])
            cls_name = class_names[cls_id]

            if conf < 0.2: 
                continue
            
            if self.DetectMode in (1, 3):
                # 提取中心点
                u = int((box[0] + box[2]) / 2)
                v = int((box[1] + box[3]) / 2)

                # 发布图像坐标系下的像素位置（暂时 Z=0）
                pt = PointStamped()
                pt.header = msg.header
                pt.header.frame_id = cls_name
                pt.header.stamp = rospy.Time.now()
                
                pt.point.x = float(u)
                pt.point.y = float(v)
                pt.point.z = float(conf)    # 用 z 存储置信度
                self.center_pub.publish(pt)
                
                rospy.loginfo("object %s, conf: %.2f, u: %.2f, v: %.2f", str(cls_name),float(conf), float(u),float(v) )
                
            else:
                # 发布左上点
                # u = int(box[0])
                # v = int(box[1])
                # 发布图像坐标系下的像素位置（暂时 Z=0）
                bb = BoundingBox()
                bb.header = msg.header
                bb.header.frame_id = cls_name
                bb.header.stamp = rospy.Time.now()
                
                # pt.point.x = float(u)
                # pt.point.y = float(v)
                
                # 修改为发布左上、右下坐标点
                bb.x1 = int(box[0])
                bb.y1 = int(box[1])
                bb.x2 = int(box[2])
                bb.y2 = int(box[3])
                bb.conf = float(conf)    # 用 z 存储置信度
                self.center_pub.publish(bb)
                
                # rospy.loginfo("object %s, conf: %d, u: %.2f, v: %.2f", str(cls_name),float(conf), float(u),float(v) )
                rospy.loginfo("object %s, conf: %.2f, x1: %d, y1: %d, x2: %d, y2: %d", str(cls_name), float(conf),
                              bb.x1, bb.y1, bb.x2, bb.y2)
        # 可视化
        # annotated = results[0].plot()
        # cv2.imshow("YOLOv8 Detection", annotated)
        # cv2.waitKey(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 Detector Node')
    parser.add_argument('--detect_mode', type=int, default=2, help='Detection mode: 1 for shapes, 2 for holes, 3 for balls')
    parser.add_argument('--top_k', type=int, default=5, help='返回前K个检测目标')
    parser.add_argument('--visualization', default=False, help='是否可视化结果')
    args = parser.parse_args()
    
    try:
        YOLOv8Detector(param = args)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass