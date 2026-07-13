# 上层控制指令：任务节点到 auv_tf_handler。
# target 使用 NED 坐标系的 PoseStamped，auv_tf_handler 转换为 LLA。
uint8 mode                    # 2=定深，3=定深定向，4=动力定位ROV（定点）
geometry_msgs/PoseStamped target
AUVMotor force                # 六自由度力/力矩：TX、TY、TZ、MX、MY、MZ
