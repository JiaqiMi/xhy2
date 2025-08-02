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
from stereo_depth.msg import BoundingBox,LineBox


class YOLOv8Mask:
    def __init__(self, param):
        rospy.init_node("yolov8_mask", anonymous=True)
        
        self.DetectMode = param.detect_mode            # 从命令行参数获取检测模式
        self.top_k = param.top_k                       # 从命令行参数获取 top_k
        self.visualization = int(param.visualization)       # 从命令行参数获取是否可视化
        self.conf_thre = float(param.conf_thre)        # 置信度阈值
        self.detc_type = param.detc_type               # 检测类型（center 或 bbox）
        self.output_type = int(param.output_type)
        
        
        model_list = [
            "/home/xhy/catkin_ws/models/shapes_model0719.pt",
            "/home/xhy/catkin_ws/models/holes_model0719.pt",
            "/home/xhy/catkin_ws/models/balls_model0725.pt" ,
            "/home/xhy/catkin_ws/models/line_mask_0801.pt",
        ]
        if self.DetectMode == 1:
            self.model = YOLO(model_list[0])
        elif self.DetectMode == 2:
            self.model = YOLO(model_list[1])
        elif self.DetectMode == 3:
            self.model = YOLO(model_list[2])
        elif self.DetectMode == 4:
            self.model = YOLO(model_list[3])
        else:
            rospy.logwarn("DetectMode error: %s", str(self.DetectMode))
            return 
        rospy.loginfo(f"Model Path: {model_list[self.DetectMode - 1]}")
    
        self.bridge = CvBridge()
        rospy.Subscriber("/left/image_raw", Image, self.image_callback)
        if self.DetectMode in (1, 3):
            self.center_pub = rospy.Publisher("/yolov8/target_center", PointStamped, queue_size=1)
        elif self.DetectMode == 2:
            self.center_pub = rospy.Publisher("/yolov8/target_bbox", BoundingBox, queue_size=1)
        elif self.DetectMode == 4:
            self.center_pub = rospy.Publisher("/yolov8/line_bbox", LineBox, queue_size=1)
        
        # 控制推断频率
        self.last_infer_time = rospy.Time.now()
        self.infer_interval = rospy.Duration(0.2)  # 单位秒 
        
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
        masks = results[0].masks.data.cpu().numpy() if results[0].masks else None
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names
        
        if masks is not None:
            for i in range(min(len(masks), self.top_k)):
                conf = confs[i]
                if conf < self.conf_thre:
                    continue

                cls_id = int(classes[i])
                cls_name = class_names[cls_id]

                mask = masks[i].astype(np.uint8) * 255
                skeleton = self.get_skeleton(mask)
                
                # 只保留最大的连通区域
                num_labels, labels_im = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
                
                # 统计每个连通区域的像素数量（跳过背景）
                max_label = 1
                max_size = 0
                for label in range(1, num_labels):  # label 0 是背景
                    size = np.sum(labels_im == label)
                    if size > max_size:
                        max_size = size
                        max_label = label

                # 创建一个新图像，只保留最大连通区域
                largest_component = np.uint8(labels_im == max_label)
                
                # 获取骨架上的像素点
                # points = np.column_stack(np.where(skeleton > 0))
                points = np.column_stack(np.where(largest_component > 0))

                # 选取最多10个点发布
                if len(points) >= 3:
                    rospy.loginfo(f"skeleton points num: {len(points)}")
                    # 按照 y 坐标升序排序
                    points_sorted = sorted(points, key=lambda p: p[0])  # y,x 格式

                    if self.detc_type == "quartiles":
                        # 获取下四分位、中位数、上四分位点
                        q1_idx = len(points_sorted) // 4
                        q2_idx = len(points_sorted) // 2
                        q3_idx = 3 * len(points_sorted) // 4
                        rospy.loginfo(f"q1_idx: {q1_idx}, q2_idx: {q2_idx}, q3_idx: {q3_idx}")
                        selected_points = [
                            points_sorted[q1_idx],
                            points_sorted[q2_idx],
                            points_sorted[q3_idx]
                        ]
                    else:
                        rospy.logwarn("Unknown output_mode: %s", self.detc_type)
                        selected_points = []

                    
                    pt = LineBox()
                    pt.header = msg.header
                    pt.header.frame_id = cls_name
                    pt.header.stamp = self.last_infer_time
                    pt.x1 = int(selected_points[0][1])
                    pt.y1 = int(selected_points[0][0])
                    pt.x2 = int(selected_points[1][1])
                    pt.y2 = int(selected_points[1][0])
                    pt.x3 = int(selected_points[2][1])
                    pt.y3 = int(selected_points[2][0])
                    pt.conf = float(conf)
                    self.center_pub.publish(pt)

                    rospy.loginfo("Segmented object: %s, mode: %s, selected points: %d, conf: %.2f",
                                cls_name, self.detc_type, len(selected_points), conf)
                else:
                    rospy.logwarn("No skeleton points found for %s", cls_name)


        else:
            # 如果是检测模型，就走原始的bbox/center逻辑
            for i, box in enumerate(boxes):
                conf = confs[i]
                if conf < self.conf_thre:
                    continue

                cls_id = int(classes[i])
                cls_name = class_names[cls_id]

                if self.detc_type == 'center':
                    u = int((box[0] + box[2]) / 2)
                    v = int((box[1] + box[3]) / 2)
                    pt = PointStamped()
                    pt.header = msg.header
                    pt.header.frame_id = cls_name
                    pt.header.stamp = self.last_infer_time
                    pt.point.x = float(u)
                    pt.point.y = float(v)
                    pt.point.z = float(conf)
                    self.center_pub.publish(pt)
                else:
                    bb = BoundingBox()
                    bb.header = msg.header
                    bb.header.frame_id = cls_name
                    bb.header.stamp = self.last_infer_time
                    bb.x1 = int(box[0])
                    bb.y1 = int(box[1])
                    bb.x2 = int(box[2])
                    bb.y2 = int(box[3])
                    bb.conf = float(conf)
                    self.center_pub.publish(bb)

        # 可视化
        if self.visualization == 1:
            annotated = results[0].plot()
            cv2.imshow("YOLOv8 Detection", annotated)
            cv2.waitKey(1)
            
            
    def get_skeleton(self, binary_img):
        size = np.size(binary_img)
        skel = np.zeros(binary_img.shape, np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(binary_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary_img, temp)
            skel = cv2.bitwise_or(skel, temp)
            binary_img = eroded.copy()

            zeros = size - cv2.countNonZero(binary_img)
            if zeros == size:
                done = True

        return skel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 Detector Node')
    parser.add_argument('--detect_mode', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=5,)
    parser.add_argument('--visualization', default=0)
    parser.add_argument('--conf_thre', default=0.2)
    parser.add_argument('--detc_type', default='center')
    parser.add_argument('--output_type', default='output_type')
    
    args = parser.parse_args()
    try:
        YOLOv8Mask(param = args)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass