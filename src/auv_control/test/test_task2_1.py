#! /home/xhy/xhy_env36/bin/python
"""
名称: test_task2_1.py
功能: 接收目标检测结果，转换到map坐标系并存入优先队列
作者: buyegaid
监听：/target_detection (来自视觉节点) 检测目标是圆形的标志物, 移动目标是让T型管插入到圆形中
      /tf (来自tf树)
发布: None

记录：
2025.7.23 1:39
    第一版完成
    TODO 测试
"""
# 用另一个节点去控制机器人定点，然后用这个节点测试目标位置
import rospy
from auv_control.msg import TargetDetection
from geometry_msgs.msg import PoseStamped, PointStamped
import tf
from queue import PriorityQueue
import numpy as np

class Task2Test:
    def __init__(self):
        # 获取目标颜色参数
        self.target_class = rospy.get_param('/task2_target_class', 'green')
        rospy.loginfo(f"test: 目标颜色: {self.target_class}")

        # 初始化tf相关
        self.tf_listener = tf.TransformListener()

        # 初始化优先队列(最多存储10个目标)
        self.target_queue = PriorityQueue(maxsize=10)
        self.target_count = 0  # 记录收到的有效目标数量
        self._counter = 0      # 递增计数器，作为次级排序键
        
        # 添加线程锁
        from threading import Lock
        self._queue_lock = Lock()

        # 订阅目标检测话题
        rospy.Subscriber('/obj/target_message', TargetDetection, self.target_callback)

        # 定时打印队列信息的定时器
        self.print_timer = rospy.Timer(rospy.Duration(5.0), self.print_queue)

    def target_callback(self, msg: TargetDetection):
        """处理目标检测消息"""
        if msg.class_name != self.target_class:
            return

        # 获取队列锁
        with self._queue_lock:
            try:
                # 将目标点从camera坐标系转换到map坐标系
                target_in_map = self.tf_listener.transformPose("map", msg.pose)
            # 如果队列未满，直接添加
                if self.target_queue.qsize()<=10:
                    priority = -msg.conf  # 使用负的置信度作为优先级
                    self._counter += 1
                    # rospy.loginfo(f"input: {(priority, msg)}")
                    self.target_queue.put((priority, self._counter, target_in_map)) # 增加计数器
                    self.target_count += 1
                    rospy.loginfo(f"test: 新目标 #{self.target_count}: 置信度={msg.conf:.5f}, "
                                f"位置=({target_in_map.pose.position.x:.3f}, "
                                f"{target_in_map.pose.position.y:.3f}, "
                                f"{target_in_map.pose.position.z:.3f})")
                else:
                    # 如果队列已满，比较置信度
                    lowest_priority, lowest_counter, _ = self.target_queue.get()
                    if -msg.conf < lowest_priority:  # 新目标置信度更高
                        self._counter += 1
                        self.target_queue.put((-msg.conf, self._counter, target_in_map))
                        rospy.loginfo(f"test: 更新目标: 新置信度={msg.conf:.2f}, "
                                    f"位置=({target_in_map.pose.position.x:.3f}, "
                                    f"{target_in_map.pose.position.y:.3f}, "
                                    f"{target_in_map.pose.position.z:.3f})")
                    else:
                        self.target_queue.put((lowest_priority, lowest_counter, _))  # 放回原来的目标
                        
            except (tf.LookupException, tf.ConnectivityException, 
                    tf.ExtrapolationException) as e:
                rospy.logwarn(f"test: 坐标转换失败: {e}")

    def print_queue(self, event):
        """定期打印队列中的目标信息"""
        # 获取队列锁
        with self._queue_lock:
            if self.target_queue.empty():
                rospy.loginfo("test: 目标队列为空")
                return

            rospy.loginfo("\ntest: 当前目标队列状态:")
            temp_queue = PriorityQueue()
            positions = []
            confidences = []

            # 遍历队列并保存数据
            while not self.target_queue.empty():
                priority, counter, target = self.target_queue.get()
                conf = -priority  # 转回置信度
                pos = target.pose.position
                positions.append([pos.x, pos.y, pos.z])
                confidences.append(conf)
                temp_queue.put((priority, counter, target))
                rospy.loginfo(f"test: - 置信度: {conf:.5f}, 位置: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")

            # 恢复队列
            self.target_queue = temp_queue

            # 计算统计信息
            if positions:
                positions = np.array(positions)
                confidences = np.array(confidences)
                mean_pos = np.mean(positions, axis=0)
                std_pos = np.std(positions, axis=0)
                rospy.loginfo(f"test: 统计信息:")
                rospy.loginfo(f"test: - 平均位置: ({mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f})")
                rospy.loginfo(f"test: - 位置标准差: ({std_pos[0]:.3f}, {std_pos[1]:.3f}, {std_pos[2]:.3f})")
                rospy.loginfo(f"test: - 平均置信度: {np.mean(confidences):.5f}")

if __name__ == '__main__':
    rospy.init_node('test_task2_1')
    try:
        node = Task2Test()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
