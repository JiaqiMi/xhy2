geometry_msgs/PoseStamped pose     # 目标在相机坐标系下的位置
float32 conf                       # 目标识别的置信度
string type                        # 位置信息返回类型，一般为‘center'
string class_name                  # 目标名称

# ======  pose消息内容  ====== #
# 时间戳赋值为图像的真实时间
# frame_id camera
# x y z 米
# 四元数0 0 0 1


# 置信度0-1
# type center/edge
# class_name: rectangle/circle/triangle：形状
#            red/green/black:对应小中大
#            red_ball/blue_ball:两种颜色的球


