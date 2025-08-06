#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import argparse


def compute_depth_map(left_img, right_img, fx, baseline):
    """Compute the disparity and depth map from stereo image pairs."""
    grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 8,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=1,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan

    depth_map = fx * baseline / disparity
    return disparity, depth_map


def normalize_depth_for_display(depth_map, max_depth=5.0):
    """Normalize depth map to 0-255 for colormap mapping."""
    disp = np.copy(depth_map)
    disp[np.isnan(disp)] = 0
    disp = np.clip(disp, 0, max_depth)
    return (disp / max_depth * 255).astype(np.uint8), max_depth


def draw_colorbar(img, max_depth, bar_h_ratio=0.3, bar_w=20, margin=10):
    """
    在 img 右下角绘制竖直 colorbar，并标注 0 和 max_depth。
    bar_h_ratio: colorbar 高度占图像高度比例
    bar_w: colorbar 宽度（px）
    margin: 与图像边缘的间距
    """
    h, w = img.shape[:2]
    bar_h = int(h * bar_h_ratio)
    # bar_x0 = w - bar_w - margin
    bar_x0 = bar_w + 5 * margin
    bar_y0 = h - bar_h - margin

    # 绘制每一行色条
    for i in range(bar_h):
        # 由下到上映射深度值
        val = (bar_h - i - 1) / (bar_h - 1)  # 0~1
        color_idx = int(val * 255)
        color = cv2.applyColorMap(
            np.array([[color_idx]], dtype=np.uint8),
            cv2.COLORMAP_JET
        )[0][0].tolist()
        cv2.line(img, (bar_x0, bar_y0 + i), (bar_x0 + bar_w, bar_y0 + i), color, 1)

    # 文本标注
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.5
    text_thick = 1
    # 最大深度
    txt_max = f"{max_depth:.2f}m"
    txt_size, _ = cv2.getTextSize(txt_max, font, text_scale, text_thick)
    cv2.putText(img, txt_max,
                (bar_x0 - txt_size[0] - 5, bar_y0 + txt_size[1]),
                font, text_scale, (255, 255, 255), text_thick)
    # 最小深度 0
    txt_min = "0.00m"
    txt_size, _ = cv2.getTextSize(txt_min, font, text_scale, text_thick)
    cv2.putText(img, txt_min,
                (bar_x0 - txt_size[0] - 5, bar_y0 + bar_h),
                font, text_scale, (255, 255, 255), text_thick)

    return img


def main():
    parser = argparse.ArgumentParser(description="Stereo Depth Map Calculation (Local Version)")
    parser.add_argument("--left", type=str, default='./local_test/calculate_depth/images/left_000308.jpg')
    parser.add_argument("--right", type=str, default='./local_test/calculate_depth/images/right_000308.jpg')
    parser.add_argument("--output", type=str, default="./local_test/calculate_depth/results/depth_output.png")
    args = parser.parse_args()

    left_img = cv2.imread(args.left)
    right_img = cv2.imread(args.right)
    if left_img is None or right_img is None:
        raise FileNotFoundError("Failed to read input images.")

    fx = 1080.689861
    baseline = 81.420154 / fx

    disparity, depth_map = compute_depth_map(left_img, right_img, fx, baseline)

    depth_disp, max_depth = normalize_depth_for_display(depth_map)
    colormap = cv2.applyColorMap(depth_disp, cv2.COLORMAP_JET)

    # 在深度图上叠加 colorbar
    colormap_with_bar = draw_colorbar(colormap.copy(), max_depth)

    # 将左右图像和深度图拼接在一起
    combined_img = np.hstack((right_img, colormap_with_bar))

    cv2.imshow("Combined View", combined_img)
    cv2.imwrite(args.output, colormap_with_bar)
    print(f"Depth map with legend saved to: {args.output}")

    # 可选：保存原始深度矩阵
    # np.save(args.output.replace(".png", ".npy"), depth_map)
    # print(f"Depth matrix (float) saved to: {args.output.replace('.png', '.npy')}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
