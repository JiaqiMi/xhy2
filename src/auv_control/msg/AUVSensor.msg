float32 temperature       # 舱内温度监测数据（°）
float32 voltage           # 控制电电压（V）
float32 current           # 系统电流（A）
uint8 battery             # 电池电量（%）
bool leak_alarm           # 漏水报警数据，00不漏水，01漏水
uint8 sensor_valid        # 传感器状态（有效性），从低位到高位0代表无效，1代表有效
                            # 0 AHRS
                            # 1 GPS
                            # 2 SBL
                            # 3 VIO
                            # 4 DVL地速
                            # 5 DVL流速
                            # 6 DVL高度
uint8 sensor_updated      # 传感器状态（更新状态），从低位到高位0代表不更新，1代表更新
                            # 0 AHRS
                            # 1 DVL
                            # 2 GPS
                            # 3 SBL
                            # 4 VIO
uint16 fault_status       # 故障状态，从低位到高位0代表无故障，1代表有故障
                            # 0 闭环超深
                            # 1 开环超深
                            # 2 漏水报警
                            # 3 通讯故障
                            # 4 电池电压报警
                            # 5 传感器采集故障
                            # 6 动力系统故障
                            # 7 闭环运行超时
                            # 8 开环运行超时
int16 power_status       # 电源状态，16位对应16个设备的电源，0代表关闭，1代表打开
                            # 从最低位（最右边）开始。分别对应惯导、DVL、USBL、声通信机、高度计、深度计、侧扫声呐、
                            # 前视声呐、单波束、多波束、避碰声呐、频闪灯、ADCP、CTD、浅剖、PC机