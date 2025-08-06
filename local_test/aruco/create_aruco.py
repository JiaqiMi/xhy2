import cv2
import numpy as np
import os


# 生成aruco標記
# 加載預定義的字典
# dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250) # 已棄用
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# 生成標記
markerImage = np.zeros((600, 600), dtype=np.uint8)
# markerImage = cv2.aruco.drawMarker(dictionary, 22, 200, markerImage, 1) # 已棄用
# markerImage = cv2.aruco.generateImageMarker(dictionary, 22, 200)
# cv2.imwrite("marker22.png", markerImage)

# (1-2)確保'label'資料夾存在
if not os.path.exists('label2'):
    os.makedirs('label2')

for i in range(30):
    markerImage = cv2.aruco.generateImageMarker(dictionary, i, 600)
    # 設置路徑
    filename = os.path.join('label2', 'arMark_' + str(i) + '.png')
    cv2.imwrite(filename, markerImage)