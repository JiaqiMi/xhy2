#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

# 基于中心像素点的目标位姿计算，并同步发布Web端JSON结果
import json

import rospy
import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from auv_control.msg import TargetDetection
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber


def pixel_to_camera_coords(u, v, depth, fx, fy, cx, cy):
    """将像素坐标和深度转换到相机坐标系。"""
    h, w = depth.shape

    u = int(u)
    v = int(v)

    if not (0 <= u < w and 0 <= v < h):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    Z = depth[v, u]

    if not np.isfinite(Z) or Z <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z], dtype=np.float64)


def get_stable_depth(
    u,
    v,
    depth,
    fx,
    fy,
    cx,
    cy,
    window_size=11,
    min_depth=0.5,
    max_depth=2.0
):
    """
    在目标中心附近取窗口，对有效深度求平均，并转换到相机坐标系。

    返回:
        np.array([X, Y, Z])
        若无有效深度，则返回 [nan, nan, nan]
    """
    half_w = window_size // 2
    h, w = depth.shape

    u = int(u)
    v = int(v)

    if not (0 <= u < w and 0 <= v < h):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    umin = max(u - half_w, 0)
    umax = min(u + half_w + 1, w)
    vmin = max(v - half_w, 0)
    vmax = min(v + half_w + 1, h)

    region = depth[vmin:vmax, umin:umax]

    valid = region[
        np.isfinite(region)
        & (region >= min_depth)
        & (region <= max_depth)
    ]

    if valid.size < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    Z = float(np.mean(valid))

    if not np.isfinite(Z) or Z <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z], dtype=np.float64)


class StereoDepthNode:
    def __init__(self):
        rospy.init_node("stereo_depth", anonymous=True)

        self.exp_env = rospy.get_param("~exp_env", "water")
        self.visualization = int(rospy.get_param("~visualization", 0))
        self.conf_thre = float(rospy.get_param("~conf_thre", 0.5))
        self.rate = rospy.Rate(float(rospy.get_param("~rate", 5.0)))

        self.target_uv = None
        self.target_conf = None
        self.target_class = None
        self.target_check_time = None
        self.update = False

        if self.exp_env == "air":
            self.fx = 572.993971
            self.fy = 572.993971
            self.cx = 374.534946
            self.cy = 271.474743
            self.baseline = 34.309807 / 572.993971
        elif self.exp_env == "water":
            self.fx = 798.731044
            self.fy = 798.731044
            self.cx = 348.127430
            self.cy = 269.935493
            self.baseline = 47.694354 / 798.731044
        else:
            rospy.logerr(
                "Invalid exp_env parameter: %s. Use 'air' or 'water'.",
                self.exp_env
            )
            rospy.signal_shutdown("Invalid exp_env parameter")
            return

        self.bridge = CvBridge()

        rospy.Subscriber(
            "/yolo26n/target_center",
            PointStamped,
            self.target_callback,
            queue_size=1
        )

        self.left_sub = Subscriber("/left/image_raw", Image)
        self.right_sub = Subscriber("/right/image_raw", Image)

        self.ts = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=5,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        self.left_img = None
        self.right_img = None
        self.left_header = None
        self.right_header = None

        # 原有自定义消息
        self.target_message = rospy.Publisher(
            "/obj/target_message",
            TargetDetection,
            queue_size=1
        )

        # Web端标准JSON消息
        self.web_pose_pub = rospy.Publisher(
            "/web/pose",
            String,
            queue_size=1
        )

        rospy.loginfo(
            "Stereo Depth Node Initialized: env=%s, conf=%.2f",
            self.exp_env,
            self.conf_thre
        )

    def target_callback(self, msg):
        """保存最新目标的中心像素、置信度、类别和检测时间。"""
        self.target_uv = (
            int(msg.point.x),
            int(msg.point.y)
        )
        self.target_conf = float(msg.point.z)
        self.target_class = str(msg.header.frame_id)
        self.target_check_time = msg.header.stamp
        self.update = True

    def image_callback(self, left_img_msg, right_img_msg):
        try:
            self.left_img = self.bridge.imgmsg_to_cv2(
                left_img_msg,
                desired_encoding="bgr8"
            )
            self.right_img = self.bridge.imgmsg_to_cv2(
                right_img_msg,
                desired_encoding="bgr8"
            )

            self.left_header = left_img_msg.header
            self.right_header = right_img_msg.header

        except Exception as e:
            rospy.logerr("cv_bridge error: %s", str(e))

    def publish_web_pose(
        self,
        valid,
        target_class,
        target_conf,
        u,
        v,
        check_time,
        position=None,
        reason=""
    ):
        """发布Web端使用的JSON位置结果。"""
        if check_time is None or check_time == rospy.Time():
            stamp = rospy.Time.now().to_sec()
        else:
            stamp = check_time.to_sec()

        payload = {
            "stamp": stamp,
            "source": "stereo_depth",
            "frame_id": "camera",
            "class_name": str(target_class),
            "confidence": float(target_conf),
            "valid": bool(valid),
            "pixel_center": {
                "u": int(u),
                "v": int(v)
            }
        }

        if position is not None:
            X, Y, Z = position
            payload["position_m"] = {
                "x": float(X),
                "y": float(Y),
                "z": float(Z)
            }
            payload["orientation_xyzw"] = {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "w": 1.0
            }

        if reason:
            payload["reason"] = reason

        self.web_pose_pub.publish(
            String(
                data=json.dumps(
                    payload,
                    ensure_ascii=False
                )
            )
        )

    def run(self):
        while not rospy.is_shutdown():

            if (
                self.left_img is None
                or self.right_img is None
                or not self.update
                or self.target_uv is None
                or self.target_conf is None
            ):
                self.rate.sleep()
                continue

            # 拷贝当前帧，避免回调线程更新图像
            left_img = self.left_img.copy()
            right_img = self.right_img.copy()

            u, v = self.target_uv
            target_conf = float(self.target_conf)
            target_class = self.target_class
            check_time = self.target_check_time

            grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16 * 6,
                blockSize=7,
                P1=8 * 3 * 7 ** 2,
                P2=32 * 3 * 7 ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )

            disparity = (
                stereo.compute(grayL, grayR)
                .astype(np.float32)
                / 16.0
            )

            # 保留你原来的处理逻辑
            disparity[disparity <= 0.0] = 0.1

            depth = self.fx * self.baseline / disparity

            rospy.loginfo(
                "Find target: time=%s, class=%s, conf=%.2f, u=%d, v=%d",
                str(check_time),
                target_class,
                target_conf,
                u,
                v
            )

            X = np.nan
            Y = np.nan
            Z = np.nan

            if target_conf < self.conf_thre:
                rospy.logwarn(
                    "Confidence too low: class=%s, conf=%.2f",
                    target_class,
                    target_conf
                )

                self.publish_web_pose(
                    valid=False,
                    target_class=target_class,
                    target_conf=target_conf,
                    u=u,
                    v=v,
                    check_time=check_time,
                    reason="confidence_too_low"
                )

                self.update = False
                self.rate.sleep()
                continue

            X, Y, Z = get_stable_depth(
                u,
                v,
                depth,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                window_size=11,
                min_depth=0.5,
                max_depth=2.0
            )

            valid_position = (
                np.isfinite(X)
                and np.isfinite(Y)
                and np.isfinite(Z)
                and (-1.0 < X < 1.0)
                and (-1.0 < Y < 1.0)
                and (0.0 < Z < 3.0)
            )

            if valid_position:
                rospy.loginfo(
                    "Valid target: time=%s, class=%s, conf=%.2f "
                    "-> X=%.3f, Y=%.3f, Z=%.3f",
                    str(check_time),
                    target_class,
                    target_conf,
                    X,
                    Y,
                    Z
                )

                try:
                    msg = TargetDetection()

                    pos_msg = PoseStamped()

                    if check_time is None or check_time == rospy.Time():
                        pos_msg.header.stamp = rospy.Time.now()
                    else:
                        pos_msg.header.stamp = check_time

                    pos_msg.header.frame_id = "camera"

                    pos_msg.pose.position.x = float(X)
                    pos_msg.pose.position.y = float(Y)
                    pos_msg.pose.position.z = float(Z)
                    pos_msg.pose.orientation = Quaternion(
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    )

                    msg.pose = pos_msg
                    msg.type = "center"
                    msg.conf = target_conf
                    msg.class_name = target_class

                    self.target_message.publish(msg)

                    self.publish_web_pose(
                        valid=True,
                        target_class=target_class,
                        target_conf=target_conf,
                        u=u,
                        v=v,
                        check_time=check_time,
                        position=(X, Y, Z)
                    )

                except Exception as e:
                    rospy.logerr(
                        "Error publishing target depth: %s",
                        str(e)
                    )

                    self.publish_web_pose(
                        valid=False,
                        target_class=target_class,
                        target_conf=target_conf,
                        u=u,
                        v=v,
                        check_time=check_time,
                        reason="publish_failed"
                    )

            else:
                rospy.loginfo(
                    "Invalid target: time=%s, class=%s, conf=%.2f "
                    "-> X=%s, Y=%s, Z=%s",
                    str(check_time),
                    target_class,
                    target_conf,
                    str(X),
                    str(Y),
                    str(Z)
                )

                self.publish_web_pose(
                    valid=False,
                    target_class=target_class,
                    target_conf=target_conf,
                    u=u,
                    v=v,
                    check_time=check_time,
                    reason="invalid_depth_or_position"
                )

            if self.visualization:
                depth_vis = cv2.normalize(
                    depth,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                ).astype(np.uint8)

                depth_vis = cv2.applyColorMap(
                    depth_vis,
                    cv2.COLORMAP_JET
                )

                vis_left = left_img.copy()

                if 0 <= u < vis_left.shape[1] and 0 <= v < vis_left.shape[0]:
                    color = (
                        (0, 255, 0)
                        if valid_position
                        else (0, 0, 255)
                    )

                    cv2.circle(
                        vis_left,
                        (u, v),
                        5,
                        color,
                        -1
                    )

                    label = (
                        "{} X:{:.2f} Y:{:.2f} Z:{:.2f}".format(
                            target_class,
                            X,
                            Y,
                            Z
                        )
                        if valid_position
                        else "{} invalid".format(target_class)
                    )

                    cv2.putText(
                        vis_left,
                        label,
                        (max(0, u - 100), max(20, v - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA
                    )

                cv2.imshow("Depth Map", depth_vis)
                cv2.imshow("Target Position", vis_left)
                cv2.waitKey(1)

            self.update = False
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = StereoDepthNode()
        node.run()
    except rospy.ROSInterruptException:
        pass