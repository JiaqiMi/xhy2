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

# 单轴子状态；X/Y 独立运行，Yaw 在水平停稳后运行
uint8 AXIS_HOLD=0
uint8 AXIS_TRACK=1
uint8 AXIS_BRAKE=2

# 状态生成时间和固定坐标系，当前 frame_id 为 map
std_msgs/Header header
# 当前状态编号，取值为上方常量
uint8 state
# 是否持有当前目标或待切换目标；不能用于判断是否到达
bool goal_active
# 当前控制器实际采用的 map 绝对目标位姿
geometry_msgs/PoseStamped goal
# control_link 到内部旋转中心目标的水平位置误差，单位 m，不包含深度误差
float32 position_error
# 当前 base_link 到任务最终 base_link 目标的水平误差，单位 m
float32 base_position_error
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
# X、Y、Yaw 三个控制轴当前子状态
uint8 x_axis_state
uint8 y_axis_state
uint8 yaw_axis_state
# control_link 当前本体前、右方向误差，单位 m
float32 x_axis_error
float32 y_axis_error
# control_link 当前本体前、右速度，单位 m/s
float32 x_axis_speed
float32 y_axis_speed
# 是否已经完成启动首帧锁定及 mode=4 定点接管
bool startup_complete
# 当前状态或最近一次状态切换原因
string reason
