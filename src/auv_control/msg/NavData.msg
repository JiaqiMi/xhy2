std_msgs/Header header

# -------------------------
# 原始报文信息
# -------------------------
uint16 counter      # 计数器
uint16 checksum     # 校验和
bool checksum_ok    

# -------------------------
# INS
# -------------------------
float64 latitude # 纬度
float64 longitude
float64 altitude
float32 heave
float32 vn
float32 ve
float32 vd
float32 roll
float32 pitch
float32 heading
uint16 ins_status

# -------------------------
# GPS
# -------------------------
float64 gps_latitude
float64 gps_longitude
float64 gps_altitude
float32 gps_vel
float32 gps_heading
uint8 gps_status

# -------------------------
# DVL / OCTANS
# -------------------------
float32 dvl_vx
float32 dvl_vy
float32 dvl_vz
float32 dvl_altitude
uint8 dvl_status

# IMU推导/惯导采样后的速度深度
float32 depth

# -------------------------
# IMU 原始采样
# -------------------------
float32 gyro_x
float32 gyro_y
float32 gyro_z

float32 accel_x
float32 accel_y
float32 accel_z

float32 temperature
uint16 imu_status

# -------------------------
# 时间
# -------------------------
uint32 yymmdd_raw
uint32 hhmmss_raw
uint16 ms

uint8 update_flags
