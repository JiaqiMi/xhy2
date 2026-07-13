# AUV控制命令消息 — debug_driver_v2 下行控制指令
# 任务节点 → tf_handler (NED→经纬度) → debug_driver_v2 → TCP → AUV
uint8 mode             # 2=定深, 3=定深定向, 4=定点
AUVPose target         # 目标位姿
AUVMotor force         # 6自由度力/力矩 TX,TY,TZ,MX,MY,MZ  范围 0-10000
