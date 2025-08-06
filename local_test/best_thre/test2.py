import cv2
import numpy as np

def nothing(x):
    pass

def create_hsv_trackbars(window_name):
    cv2.createTrackbar('H Min', window_name, 0, 179, nothing)
    cv2.createTrackbar('H Max', window_name, 179, 179, nothing)
    cv2.createTrackbar('S Min', window_name, 0, 255, nothing)
    cv2.createTrackbar('S Max', window_name, 255, 255, nothing)
    cv2.createTrackbar('V Min', window_name, 0, 255, nothing)
    cv2.createTrackbar('V Max', window_name, 255, 255, nothing)

def get_trackbar_values(window_name):
    h_min = cv2.getTrackbarPos('H Min', window_name)
    h_max = cv2.getTrackbarPos('H Max', window_name)
    s_min = cv2.getTrackbarPos('S Min', window_name)
    s_max = cv2.getTrackbarPos('S Max', window_name)
    v_min = cv2.getTrackbarPos('V Min', window_name)
    v_max = cv2.getTrackbarPos('V Max', window_name)
    return h_min, h_max, s_min, s_max, v_min, v_max

def main():
    image_path = './data/my_photo-17.jpg'  # <-- 替换为你的图像路径
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图像，请检查路径")
        return

    window_name = 'HSV Threshold Adjustment'
    cv2.namedWindow(window_name)
    create_hsv_trackbars(window_name)

    while True:
        h_min, h_max, s_min, s_max, v_min, v_max = get_trackbar_values(window_name)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # Combine original and mask side by side
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((image, mask_bgr))

        # Resize combined to fit screen if too large
        max_width = 1200
        scale = max_width / combined.shape[1]
        if scale < 1.0:
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):  # 按 'q' 键退出
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()