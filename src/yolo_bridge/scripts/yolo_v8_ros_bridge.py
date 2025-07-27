#!/home/xhy/xhy_env36/bin/python
import cv2
import time
import rospy
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from cv_bridge import CvBridge
# from torchvision.ops import nms
from yolo_bridge.utils import nms
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
cuda.init()                                 # 初始化 CUDA 驱动
device = cuda.Device(0)  # 默认设备
context_cuda = device.make_context()        # 创建 CUDA 上下文（全局上下文）

# 参数定义
DetectMode = rospy.get_param('~detect_mode', 1)
TopK = rospy.get_param('~top_k', 3)

class TRT_YOLOv8:
    def __init__(self, engine_path, input_shape=(1,3,640,640)):
        
        self.context_cuda = context_cuda    # 保存上下文，后面激活要用
        
        # 1) 反序列化 engine
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 2) 分配缓冲区
        self.h_input = np.zeros(input_shape, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        C = 5 + 3                          # 若 80 类，可改
        max_det = 6300                     # 假设最大检测数
        out_shape = (1, max_det, C)
        self.h_output = np.zeros(out_shape, dtype=np.float32)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # 3) 绑定输入输出
        self.bindings = [int(self.d_input), int(self.d_output)]
        

    def preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640,640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None, ...]  # 图像通道变换 NHWC->NCHW
        return img
    

    def infer(self, img_bgr):
        self.context_cuda.push()           # 激活 CUDA 上下文
        
        # 4) 图像预处理
        self.h_input[:] = self.preprocess(img_bgr)

        # 5) DtoH: 将数据从主机内存复制到设备内存
        cuda.memcpy_htod(self.d_input, self.h_input)  

        # 6) 执行推理
        self.context.execute_v2(self.bindings)

        # 7) HtoD
        cuda.memcpy_dtoh(self.h_output, self.d_output)
        self.context_cuda.pop()            # 释放 CUDA 上下文
        return self.h_output[0]            # h_output.shape = (1, max_det, C)

    def postprocess(self, dets, conf_thres=0.25, iou_thres=0.45):
        """Process the detections from the model output.
        
        :param dets: np.array, shape [N, 5+num_classes]
        :param conf_thres: float, confidence threshold
        :param iou_thres: float, IoU threshold for NMS
        
        :return: list of tuples (box, score, class_id)
        """
        
        results = []
        if dets is not None or len(dets) != 0:
            obj = dets[:, 4:5]
            cls = dets[:, 5:]
            scores = (obj * cls.max(1, keepdims=True)[0]).reshape(-1)
            cls_ids = cls.argmax(1)
            mask = scores > conf_thres
            if not mask.any(): return []
            dets, scores, cls_ids = dets[mask], scores[mask], cls_ids[mask]
            x,y,w,h = dets[:,:4].T
            boxes = torch.tensor(np.stack([x-w/2,y-h/2,x+w/2,y+h/2],1))
            
            # 使用 NMS 进行非极大值抑制
            # keep = nms(boxes.cuda(), torch.tensor(scores).cuda(), iou_thres).cpu()
            keep = nms.nms_numpy(np.array(boxes), np.array(scores), iou_thres)
            for i in keep:
                results.append((boxes[i].cpu().numpy().tolist(),
                                float(scores[i]), int(cls_ids[i])))
        return results
    
    def __del__(self):
        self.context.pop()

class YOLONode:
    def __init__(self):
        rospy.init_node("yolov8_tensorrt", anonymous=True)
        engine_path = rospy.get_param('~engine_path',
                                    "/home/xhy/catkin_ws/models/shapes_model0719_fp16.trt")
        self.detector = TRT_YOLOv8(engine_path)
        
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/yolov8/target_center",
                                PointStamped, queue_size=1)
        rospy.Subscriber("/left/image_raw", Image, self.cb)
        rospy.loginfo("YOLOv8 TRT Node Ready")

    def cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        start_time = rospy.Time.now()  # 推理开始时间
        raw = self.detector.infer(img)
        end_time = rospy.Time.now()    # 推理结束时间
        infer_time_ms = (end_time - start_time).to_sec() * 1000  # 单位：ms
        rospy.loginfo("Inference Time: %.2f ms", infer_time_ms)
        
        dets = self.detector.postprocess(raw, conf_thres=0.5, iou_thres=0.45)
        for box, score, cls_id in dets[:TopK]:  # 只取前 TopK 个检测结果
            x1, y1, x2, y2 = map(int, box)
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)
            # u = int((box[0]+box[2])/2)
            # v = int((box[1]+box[3])/2)
            pt = PointStamped()
            pt.header = msg.header
            pt.header.frame_id = str(cls_id)
            pt.header.time = start_time
            pt.point.x = u
            pt.point.y = v 
            pt.point.z = score
            
            rospy.loginfo("Valid target: time=%s, class=%s conf=%.2f -> u=%d v=%d" \
            start_time, cls_id, self.score, u, v)
            self.pub.publish(pt)
            
            # 可视化：绘制边框、类别与置信度
            label = f"Class {int(cls_id)}: {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 显示图像窗口
        cv2.imshow("YOLOv8 Detection", img)
        cv2.waitKey(1)

if __name__=="__main__":
    YOLONode()
    rospy.spin()
