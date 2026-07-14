# 下层控制指令：auv_tf_handler 到 debug_driver_v2。
# target 使用 LLA 坐标系的 AUVPose。
uint8 mode                    # 2=定深，3=定深定向，4=动力定位ROV（定点）
AUVPose target
AUVMotor force                # 六自由度力/力矩：TX、TY、TZ、MX、MY、MZ
