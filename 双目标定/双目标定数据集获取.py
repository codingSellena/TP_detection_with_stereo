import cv2
import os
import numpy as np

# 选择双目摄像头编号（可能需要更改为 0、1、2 试试）
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 电脑设置0，双目设置1
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# 确保摄像头成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    exit()

# 设定保存目录
left_dir = "left0411"
right_dir = "right0411"
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

# 读取摄像头分辨率（用于切割）
ret, frame = cap.read()
if not ret:
    print("错误：无法读取摄像头画面！")
    cap.release()
    exit()

# 假设双目摄像头的图像是水平拼接的
frame_height, frame_width, _ = frame.shape
half_width = frame_width // 2  # 假设是左右拼接

print(f"检测到分辨率：{frame_width}x{frame_height}，左右图像宽度：{half_width}")
image_count = 0  # 计数器

# 设定棋盘格参数（用于相机标定）
chessboard_size = (8,5)  # 9x6 的棋盘格,有8*5的内点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 角点检测优化参数

while True:
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        print("警告：无法读取帧！跳过当前帧，继续尝试...")
        continue  # 如果无法读取帧，跳过并继续读取下一帧

    # 切割为左、右画面
    left_image = frame[:, :half_width]  # 左半部分
    right_image = frame[:, half_width:]  # 右半部分

    # 复制图像用于检测
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # 查找左、右摄像头中的棋盘格角点
    left_found, left_corners = cv2.findChessboardCorners(left_gray, chessboard_size, None)
    right_found, right_corners = cv2.findChessboardCorners(right_gray, chessboard_size, None)

    # 在界面上标注检测到的棋盘格
    display_left = left_image.copy()
    display_right = right_image.copy()

    if left_found:
        cv2.drawChessboardCorners(display_left, chessboard_size, left_corners, left_found)

    if right_found:
        cv2.drawChessboardCorners(display_right, chessboard_size, right_corners, right_found)
        # cv2.putText(display_right, "Chessboard Detected!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # print("右边找到角点")

    # 显示左右画面
    cv2.imshow("Left Camera", display_left)
    cv2.imshow("Right Camera", display_right)

    # 同时显示灰度图像，用于调试
    # cv2.imshow("Left Gray", left_gray)
    # cv2.imshow("Right Gray", right_gray)

    # 监听按键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and (left_found or right_found):  # 仅在检测到棋盘格时拍摄
        left_path = os.path.join(left_dir, f"left_{image_count}.jpg")
        right_path = os.path.join(right_dir, f"right_{image_count}.jpg")

        # 仅保存原始图像，不包含绘制内容
        cv2.imwrite(left_path, left_image)
        cv2.imwrite(right_path, right_image)

        print(f"✅ 已保存：{left_path} 和 {right_path}")
        image_count += 1  # 计数器 +1

    elif key == ord('q'):  # 按 'q' 退出
        print("退出程序")
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
