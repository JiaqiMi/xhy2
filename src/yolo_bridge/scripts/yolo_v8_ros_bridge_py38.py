# #!/home/xhy/xhy_env/bin/python
# import rospy
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from sensor_msgs.msg import Image
# from std_msgs.msg import Header
# from cv_bridge import CvBridge
# from geometry_msgs.msg import PointStamped

# DetectMode = rospy.get_param('~detect_mode', 1)  # 下潜深度，单位米

# class YOLOv8Detector:
#     def __init__(self):
#         rospy.init_node("yolov8_detector", anonymous=True)
#         if DetectMode == 1:
#             self.model = YOLO("/home/xhy/catkin_ws/models/shapes_model0719.pt")
#         elif DetectMode == 2:
#             self.model = YOLO("/home/xhy/catkin_ws/models/holes_model0719.pt")
#         elif DetectMode == 3:
#             self.model = YOLO("/home/xhy/catkin_ws/models/balls_model0719.pt")
#         else:
#             rospy.logwarn("DetectMode error: %s, select the shapes model by default", str(DetectMode))
#             self.model = YOLO("/home/xhy/catkin_ws/models/shapes_model0719.pt")
            
#         self.bridge = CvBridge()

#         rospy.Subscriber("/left/image_raw", Image, self.image_callback)
#         self.center_pub = rospy.Publisher("/yolov8/target_center", PointStamped, queue_size=1)
#         rospy.loginfo("YOLOv8 Detector Node Started")

#     def image_callback(self, msg):
#         try:
#             img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         except Exception as e:
#             rospy.logerr("cv_bridge error: %s", str(e))
#             return

#         results = self.model(img)
#         boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
#         confs = results[0].boxes.conf.cpu().numpy()
#         classes = results[0].boxes.cls.cpu().numpy()
#         class_names = results[0].names

#         for i, box in enumerate(boxes):
#             conf = confs[i]
#             cls_id = int(classes[i])
#             cls_name = class_names[cls_id]

#             if conf < 0.2: 
#                 continue
            
#             if self.model in (1, 3):
#                 # 提取中心点
#                 u = int((box[0] + box[2]) / 2)
#                 v = int((box[1] + box[3]) / 2)

#                 # 发布图像坐标系下的像素位置（暂时 Z=0）
#                 pt = PointStamped()
#                 pt.header = msg.header
#                 pt.header.frame_id = cls_name
#                 pt.header.time = rospy.Time.now()
#                 pt.point.x = float(u)
#                 pt.point.y = float(v)
#                 pt.point.z = float(conf)    # 用 z 存储置信度
#                 self.center_pub.publish(pt)
#             else:
#                 # 发布左上点
#                 u = int(box[0])
#                 v = int(box[1])

#                 # 发布图像坐标系下的像素位置（暂时 Z=0）
#                 pt = PointStamped()
#                 pt.header = msg.header
#                 pt.header.frame_id = cls_name
#                 pt.header.time = rospy.Time.now()
#                 pt.point.x = float(u)
#                 pt.point.y = float(v)
#                 pt.point.z = float(conf)    # 用 z 存储置信度
#                 self.center_pub.publish(pt)
                
#         # 可视化
#         annotated = results[0].plot()
#         cv2.imshow("YOLOv8 Detection", annotated)
#         cv2.waitKey(1)

# if __name__ == "__main__":
#     try:
#         YOLOv8Detector()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass

