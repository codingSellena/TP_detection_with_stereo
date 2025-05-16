import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# 相机内参矩阵与畸变系数
mat_inter_left = np.array([[528.38109373, 0., 236.83141157],
                           [0., 528.18779028, 245.05737806],
                           [0., 0., 1.]])
coff_dis_left = np.array([[-5.71546664e-02, 2.37964897e-01, 6.36537316e-04, 1.55096858e-04, -2.83687647e-01]])

mat_inter_right = np.array([[522.31245578, 0., 352.6552961],
                            [0., 522.66849156, 264.62299847],
                            [0., 0., 1.]])
coff_dis_right = np.array([[-8.52047297e-02, 4.04945322e-01, 1.87443906e-03, 1.09121555e-06, -5.75595393e-01]])

# 相机的基线（单位为米）
baseline = 0.1  # 例如 10cm，设置为实际的基线距离

# 焦距
f_x = mat_inter_left[0, 0]  # 取左相机内参矩阵中的焦距
f_y = mat_inter_left[1, 1]  # 取左相机内参矩阵中的焦距

# 初始化立体匹配
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # 必须是16的倍数
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("警告：无法读取帧！")
        continue

    # 分割左右视图
    height, width = frame.shape[:2]
    half_width = width // 2
    left_img = frame[:, :half_width]
    right_img = frame[:, half_width:]

    # 去畸变
    left_undistorted = cv2.undistort(left_img, mat_inter_left, coff_dis_left)
    right_undistorted = cv2.undistort(right_img, mat_inter_right, coff_dis_right)

    # 转为灰度图
    gray_left = cv2.cvtColor(left_undistorted, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_undistorted, cv2.COLOR_BGR2GRAY)

    # 计算视差图
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32)

    # 视差图的空洞填充（中值滤波 + 双边滤波）
    disparity = cv2.medianBlur(disparity, 5)  # 中值滤波
    disparity = cv2.bilateralFilter(disparity, 9, 75, 75)  # 双边滤波

    # 处理视差图中的零值，避免除以零
    disparity[disparity == 0] = 0.1  # 可以选择一个小的非零值来避免除以零

    # 转为深度图
    # 深度计算公式：depth = (f * B) / disparity
    # 深度值计算
    depth = (f_x * baseline) / disparity

    # 归一化深度图用于显示
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 显示结果
    cv2.imshow("Left Undistorted", left_undistorted)
    cv2.imshow("Right Undistorted", right_undistorted)
    cv2.imshow("Depth", depth_vis)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
