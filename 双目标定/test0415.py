import cv2
import numpy as np

# 打开左眼和右眼摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 电脑设置0，双目设置1
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# 读取摄像头分辨率（用于切割）
ret, frame = cap.read()
if not ret:
    print("错误：无法读取摄像头画面！")
    cap.release()
    exit()

# 获取图像尺寸
frame_height, frame_width, _ = frame.shape
half_width = frame_width // 2  # 假设是左右拼接

# 相机内参和畸变系数
mat_inter_left = np.array([[485.31810141, 0., 299.78121813],
                           [0., 486.58790624, 278.01553071],
                           [0., 0., 1.]])
coff_dis_left = np.array([[-0.08967496,  0.20176819,  0.00913968, -0.00456322, -0.12016892]])

mat_inter_right = np.array([[494.0977399,   0., 358.30701335],
                            [0., 494.26305562, 267.99135094],
                            [0., 0., 1.]])
coff_dis_right = np.array([[-0.11759724, 0.41773862, 0.00034209, 0.00292024, -0.4879663]])

# 立体匹配器和滤波器
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5,
    blockSize=11,
    P1=8 * 3 * 11 ** 2,
    P2=32 * 3 * 11 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(800)
wls_filter.setSigmaColor(1.5)

# 假设基线和焦距
baseline = 0.06  # 单位：米
focal_length = mat_inter_left[0, 0]

while True:
    ret, frame = cap.read()
    if not ret:
        print("警告：无法读取帧！跳过当前帧...")
        continue

    # 切割为左右画面
    left_frame = frame[:, :half_width]
    right_frame = frame[:, half_width:]

    # 去畸变
    left_rectified = cv2.undistort(left_frame, mat_inter_left, coff_dis_left)
    right_rectified = cv2.undistort(right_frame, mat_inter_right, coff_dis_right)

    # 灰度图
    gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    # 计算左右视差图（并归一化）
    disp_left = left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    disp_right = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0

    # WLS 滤波
    filtered_disparity = wls_filter.filter(disp_left, gray_left, None, disp_right)

    # 归一化视差图用于显示
    disparity_display = cv2.normalize(filtered_disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_display = np.uint8(disparity_display)

    # 深度计算（避免除零）
    disp_nonzero = np.where(filtered_disparity <= 0, 1e-5, filtered_disparity)
    depth_map = (baseline * focal_length) / disp_nonzero
    depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
    depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 显示中心点测距
    h, w = filtered_disparity.shape
    cx, cy = w // 2, h // 2
    cv2.circle(left_rectified, (cx, cy), 5, (0, 0, 255), 2)
    center_disparity = filtered_disparity[cy, cx]
    if center_disparity > 0:
        center_depth = (baseline * focal_length) / center_disparity
    else:
        center_depth = 0
    cv2.putText(left_rectified, f"Depth: {center_depth:.2f}m", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 显示图像
    cv2.imshow("Left Camera", left_rectified)
    cv2.imshow("Disparity", disparity_display)
    cv2.imshow("Depth Map", depth_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
