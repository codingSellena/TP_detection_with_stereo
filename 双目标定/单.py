import cv2
import numpy as np
import glob

# 设置棋盘格的规格（根据你使用的棋盘格尺寸来设置）
chessboard_size = (8,5)  # 9列，6行
square_size = 0.21  # 每个方格的实际尺寸，单位可以是毫米或者厘米

# 棋盘格的3D坐标，假设棋盘格的Z坐标为零
obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
obj_points *= square_size

# 用于保存所有图像的物体点和图像点
object_points = []
image_points = []

# 获取所有棋盘格图像
images = glob.glob('left0411/*.jpg')  # 调整图像路径

print(images)  # 打印检查是否读取到了图像文件

for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        object_points.append(obj_points)
        image_points.append(corners)

        # 在棋盘格图像上绘制角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
    else:
        print(f"未检测到棋盘格角点：{img_path}")

cv2.destroyAllWindows()

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# 输出相机内参和畸变系数
print("相机内参矩阵：")
print(mtx)
print("畸变系数：")
print(dist)

# 计算重投影误差
total_error = 0
for i in range(len(object_points)):
    imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(object_points)
print(f"\n⚠️ 平均重投影误差 (RMS error): {mean_error:.4f}")

