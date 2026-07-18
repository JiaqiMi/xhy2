# 默认随状态机以 5 Hz 发布；任务节点仅将 HOVER 视为目标到达且定点接管已确认
uint8 IDLE=0
uint8 ALIGN_PATH=1
uint8 ALIGN_PATH_BRAKE=2
uint8 TRANSLATE=3
uint8 TRANSLATE_BRAKE=4
uint8 ALIGN_FINAL=5
uint8 FINAL_BRAKE=6
uint8 CAPTURE=7
uint8 HOVER=8
uint8 SAFE=9

# 状态生成时间和固定坐标系，当前 frame_id 为 map
std_msgs/Header header
# 当前状态编号，取值为上方常量
uint8 state
# 是否持有当前目标或待切换目标；不能用于判断是否到达
bool goal_active
# 当前控制器实际采用的 map 绝对目标位姿
geometry_msgs/PoseStamped goal
# 当前水平位置误差，单位 m，不包含深度误差
float32 position_error
# 当前最终航向误差，单位 rad，范围 [-pi, pi)
float32 yaw_error
# 低通后的水平合速度，单位 m/s
float32 horizontal_speed
# 低通后的航向角速度，单位 rad/s
float32 yaw_rate
# 本周期实际下发的有符号 int16 水平力和偏航力矩
int16 tx
int16 ty
int16 mz
# 当前状态或最近一次状态切换原因
string reason
