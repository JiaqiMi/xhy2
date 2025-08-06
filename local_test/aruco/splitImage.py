
import os
import cv2

# BEGIN: 实现分割双目图像的功能
def split_stereo_image(image_path, output_left_path, output_right_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 获取图像的宽度
    height, width, _ = image.shape

    # 分割图像
    left_image = image[:, :width // 2]
    right_image = image[:, width // 2:]

    # 保存分割后的图像
    cv2.imwrite(output_left_path, left_image)
    cv2.imwrite(output_right_path, right_image)

def main():
    data_folder = './local_test/aruco/data'
    for filename in os.listdir(data_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(data_folder, filename)
            output_left_path = os.path.join(data_folder, f'left_{filename}')
            output_right_path = os.path.join(data_folder, f'right_{filename}')
            split_stereo_image(image_path, output_left_path, output_right_path)

if __name__ == "__main__":
    main()
# END:
