# 执行器控制消息 - sensor debug协议下行控制
# 对应下位机 CAMERA_LIGHT_SET (cmd=0x10) 和 ACTUATOR_SET (cmd=0x30)

uint8 mode                # 控制模式：0=不响应，1=仅补光灯，2=仅执行器

# --- 补光灯 (CAMERA_LIGHT_SET, cmd=0x10 op=0x00) ---
uint8 light1              # 补光灯1亮度 0-100
uint8 light2              # 补光灯2亮度 0-100

# --- 执行器 (ACTUATOR_SET, cmd=0x30 op=0x00) ---
uint8 heading_servo       # 航向舵机 0-255, 0x00=右, 0x80=中, 0xFF=左
uint8 clamp_servo         # 开合舵机 0-255, 0x00=全开, 0xFF=全闭
uint8 drive_cmd           # 推杆电机动作 0=停止, 1=前进, 2=反转
uint8 drive_speed         # 推杆电机转速 0-255
uint8 red_light           # 红色指示灯 0/1
uint8 yellow_light        # 黄色指示灯 0/1
uint8 green_light         # 绿色指示灯 0/1
