#! /home/xhy/xhy_env36/bin/python
"""
名称：test_depth.py
功能：读取debug_data.txt，绘制深度数据对比图
作者：buyegaid
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# 设置中文字体
font = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")

# 读取数据文件
try:
    df = pd.read_csv('/home/hsx/debug_data.csv')
    
    # 创建时间序列（相对于起始时间）
    time_series = df['pc_timestamp'] - df['pc_timestamp'].iloc[0]
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制三条深度曲线
    plt.plot(time_series, df['depth_raw'], 'b-', label='原始深度', alpha=0.5)
    plt.plot(time_series, df['depth_lpf'], 'r-', label='低通滤波', linewidth=2)
    plt.plot(time_series, df['depth_ma'], 'g-', label='移动平均', linewidth=2)
    
    # 设置图形属性
    plt.title('深度数据对比', fontproperties=font)
    plt.xlabel('时间 (秒)', fontproperties=font)
    plt.ylabel('深度 (米)', fontproperties=font)
    plt.grid(True)
    plt.legend(prop=font)
    
    # 计算并显示统计信息
    stats_text = (
        f"原始深度标准差: {df['depth_raw'].std():.3f}m\n"
        f"低通滤波标准差: {df['depth_lpf'].std():.3f}m\n"
        f"移动平均标准差: {df['depth_ma'].std():.3f}m"
    )
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontproperties=font,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 显示图形
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"读取或处理数据失败: {e}")
