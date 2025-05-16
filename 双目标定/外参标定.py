import cv2
import numpy as np
import glob

# 设置棋盘格的尺寸
chessboard_size = (8, 5)  # 假设棋盘格是 9x6 的内角点

# 设置标定板的世界坐标（单位：米）
square_size = 0.021  # 每个方格的尺寸，单位为米
obj_points = []  # 3D点
img_points_left = []  # 左相机的2D点
img_points_right = []  # 右相机的2D点

# 生成棋盘格的世界坐标点
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

# 读取图像
images_left = glob.glob('left0411/*.jpg')
images_right = glob.glob('right0411/*.jpg')

# 左右相机的内参矩阵和畸变系数
K1 = np.array([[528.38109373, 0., 236.83141157],
               [0., 532.10325101, 227.29957483],
               [0., 0., 1.]])
dist1 = np.array([[-0.08867956, 0.21417696, 0.0121195, -0.00883573, -0.1450075]])

K2 =np.array([[522.31245578, 0., 352.6552961],
                           [0., 526.72619472, 261.63689191],
                           [0., 0., 1.]])
dist2 = np.array([[-1.15535817e-01, 4.01528692e-01, 3.64205417e-04, 2.83462588e-03, -4.85837394e-01]])

for left_img_path, right_img_path in zip(images_left, images_right):
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        img_points_left.append(corners_left)
        img_points_right.append(corners_right)
        obj_points.append(objp)

        # 亚像素精度优化角点
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1),
                                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        # 可视化检测到的角点
        cv2.drawChessboardCorners(left_img, chessboard_size, corners_left, ret_left)
        cv2.drawChessboardCorners(right_img, chessboard_size, corners_right, ret_right)

        cv2.imshow("Left", left_img)
        cv2.imshow("Right", right_img)

cv2.destroyAllWindows()

# 获取外参矩阵：旋转矩阵和位移向量
ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left, img_points_right,
    K1, dist1, K2, dist2,
    gray_left.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)

# 输出外参矩阵
print("Rotation Matrix (R):")
print(R)
print("Translation Vector (T):")
print(T)
