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
from stereo_depth.msg import BoundingBox, LineBox

class YOLOv8Node:
    def __init__(self, param):
        # 初始化 ROS 节点
        rospy.init_node("yolov8_node", anonymous=True)

        # 参数设置
        self.detect_mode   = param.detect_mode             # 模型模式
        self.top_k         = param.top_k                   # 最大输出数
        self.visualization = int(param.visualization)      # 是否可视化
        self.conf_thre     = float(param.conf_thre)        # 置信度阈值
        self.detc_type     = param.detc_type               # 输出类型（center, bbox, quartiles）
        self.rate          = rospy.Rate(5.0)               # 推断频率
        self.bridge        = CvBridge()
        self.left_img      = None

        # 模型列表
        model_list = [
            "/home/xhy/catkin_ws/models/shapes_model0719.pt",
            "/home/xhy/catkin_ws/models/holes_model0719.pt",
            "/home/xhy/catkin_ws/models/balls_model0725.pt",
            "/home/xhy/catkin_ws/models/line_mask_0801.pt",
        ]
        if 1 <= self.detect_mode <= len(model_list):
            self.model = YOLO(model_list[self.detect_mode - 1])
            rospy.loginfo(f"Loaded model: {model_list[self.detect_mode - 1]}")
        else:
            rospy.logwarn("Invalid detect_mode: %s", self.detect_mode)
            raise ValueError("detect_mode must be between 1 and {len(model_list)}")

        # 订阅图像话题
        rospy.Subscriber("/left/image_raw", Image, self.image_callback)

        # 根据模式创建发布器
        if self.detect_mode == 4 and self.detc_type == 'quartiles':
            self.pub = rospy.Publisher("/yolov8/line_bbox", LineBox, queue_size=1)
        elif self.detc_type == 'center':
            self.pub = rospy.Publisher("/yolov8/target_center", PointStamped, queue_size=1)
        elif self.detc_type == 'bbox':
            self.pub = rospy.Publisher("/yolov8/target_bbox", BoundingBox, queue_size=1)
        else:
            rospy.logwarn("Invalid detc_type: %s", self.detc_type)
            raise ValueError("detc_type must be 'center', 'bbox', or 'quartiles' ") 

        rospy.loginfo("YOLOv8 Node initialized")

    def image_callback(self, msg):
        """Convert ROS Image message to OpenCV image """
        try:
            self.left_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", str(e))

    def get_skeleton(self, binary_img):
        """提取二值图像的骨架 """
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

    def run(self):
        """主循环，进行推断并发布结果 """
        while not rospy.is_shutdown():
            if self.left_img is None:
                continue

            results = self.model(self.left_img)
            res = results[0]

            # Mask 模式
            if self.detect_mode == 4:
                masks = res.masks.data.cpu().numpy() if res.masks else None
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()
                names = res.names
                if masks is not None:
                    for i in range(min(len(masks), self.top_k)):
                        conf = confs[i]
                        if conf < self.conf_thre:
                            continue
                        cls_id = int(classes[i])
                        cls_name = names[cls_id]
                        mask = (masks[i] * 255).astype(np.uint8)
                        skeleton = self.get_skeleton(mask)
                        # 连通域提取
                        num_labels, labels_im = cv2.connectedComponents(skeleton, connectivity=8)
                        max_label, max_size = 1, 0
                        for l in range(1, num_labels):
                            size = np.sum(labels_im == l)
                            if size > max_size:
                                max_size = size
                                max_label = l
                        largest = (labels_im == max_label).astype(np.uint8)
                        points = np.column_stack(np.where(largest > 0))
                        if len(points) >= 3 and self.detc_type == 'quartiles':
                            # 四分位点
                            pts = sorted(points, key=lambda p: p[0])
                            idxs = [len(pts)//4, len(pts)//2, 3*len(pts)//4]
                            sel = [pts[i] for i in idxs]
                            msg = LineBox()
                            msg.header.stamp = rospy.Time.now()
                            msg.header.frame_id = cls_name
                            for j, (y, x) in enumerate(sel, start=1):
                                setattr(msg, f'x{j}', int(x))
                                setattr(msg, f'y{j}', int(y))
                            msg.conf = float(conf)
                            self.pub.publish(msg)
            # 普通检测模式
            else:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()
                names = res.names
                for i, box in enumerate(boxes):
                    conf = confs[i]
                    if conf < self.conf_thre:
                        continue
                    cls_id = int(classes[i])
                    cls_name = names[cls_id]
                    if self.detc_type == 'center':
                        u = int((box[0] + box[2]) / 2)
                        v = int((box[1] + box[3]) / 2)
                        msg = PointStamped()
                        msg.header.stamp = rospy.Time.now()
                        msg.header.frame_id = cls_name
                        msg.point.x = float(u)
                        msg.point.y = float(v)
                        msg.point.z = float(conf)
                        self.pub.publish(msg)
                    elif self.detc_type == 'bbox':
                        msg = BoundingBox()
                        msg.header.stamp = rospy.Time.now()
                        msg.header.frame_id = cls_name
                        msg.x1, msg.y1, msg.x2, msg.y2 = map(int, box)
                        msg.conf = float(conf)
                        self.pub.publish(msg)

            # 可视化
            if self.visualization == 1:
                annotated = res.plot()
                cv2.imshow("YOLOv8 Detection", annotated)
                cv2.waitKey(1)  
            
            self.rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 Unified Node')
    parser.add_argument('--detect_mode', type=int, default=1, help='1: shapes, 2: holes, 3: balls, 4: line mask')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--visualization', type=int, default=0)
    parser.add_argument('--conf_thre', type=float, default=0.2)
    parser.add_argument('--detc_type', type=str, default='center', help='center | bbox | quartiles')
    args = parser.parse_args()
    try:
        node = YOLOv8Node(param=args)
        node.run()
    except rospy.ROSInterruptException:
        pass
