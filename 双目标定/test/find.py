import cv2
import numpy as np

# 读取图像
image = cv2.imread('chessboard.png')
# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', gray_image)
cv2.waitKey(0)
# 设置棋盘格大小 (9x6)
pattern_size = (8, 5)

# 查找棋盘格角点
ret, corners = cv2.findChessboardCorners(gray_image, pattern_size)

# 检查是否成功找到角点
if ret:
    print("成功找到棋盘格角点！")
    # 在图像中绘制角点
    cv2.drawChessboardCorners(image, pattern_size, corners, ret)
    # 显示结果
    cv2.imshow('Chessboard', image)
    cv2.waitKey(0)
else:
    print("未找到棋盘格角点")

# 释放窗口
cv2.destroyAllWindows()
