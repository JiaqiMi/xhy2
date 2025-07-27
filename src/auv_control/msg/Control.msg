# driver节点订阅，用于控制AUV的运动和状态
# 该文件定义了AUV控制消息的格式
bool enable # 代表该帧是否是外设控制
uint8 led_red # 红色LED开关
uint8 led_green # 绿色LED开关
uint8 servo # 舵机角度
AUVMotor force # AUV的力数据
AUVPose pose # AUV的位姿数据