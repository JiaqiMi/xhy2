import cv2
import numpy as np
import matplotlib.pyplot as plt

class PipelineSkeletonTester:
    def __init__(self):
        pass

    def get_skeleton(self, mask):
        # Zhang-Suen 细化算法（或 OpenCV 内建的细化算法）
        size = np.size(mask)
        skel = np.zeros(mask.shape, np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(mask, temp)
            skel = cv2.bitwise_or(skel, temp)
            mask = eroded.copy()

            zeros = size - cv2.countNonZero(mask)
            if zeros == size:
                done = True
        return skel

    def process(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 红色分两段，低+高
        # lower_red1 = np.array([0, 70, 50])
        # upper_red1 = np.array([10, 255, 255])
        # lower_red2 = np.array([160, 70, 50])
        # upper_red2 = np.array([180, 255, 255])
        # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        # mask = cv2.bitwise_or(mask1, mask2)
        
        # 蓝色
        lower_blue = np.array([95, 239, 167])
        upper_blue = np.array([109, 255, 190])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # 骨架提取
        skeleton = self.get_skeleton(mask_clean)

        # 提取骨架像素点
        points = np.column_stack(np.where(skeleton > 0))  # [[y,x], ...]

        selected_points = []
        if len(points) >= 1:
            indices = np.linspace(0, len(points) - 1, 10, dtype=int)
            selected_points = points[indices]

        return mask, skeleton, selected_points


def visualize(image, mask, skeleton, selected_points):
    # 转换骨架为可视图像
    skeleton_vis = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    for y, x in selected_points:
        cv2.circle(skeleton_vis, (x, y), 3, (0, 255, 255), -1)

    # 可视化
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Skeleton + Points")
    plt.imshow(cv2.cvtColor(skeleton_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 加载图像
    image_path = "./data/my_photo-14.jpg"       # <-- 替换为你的图像路径
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图像，请检查路径")
        exit(1)

    tester = PipelineSkeletonTester()
    mask, skeleton, selected_points = tester.process(image)

    visualize(image, mask, skeleton, selected_points)
