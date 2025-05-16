import cv2
import numpy as np

# 鼠标点击获取深度
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_map = param
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            depth_val = depth_map[y, x]
            print(f"点击坐标 ({x}, {y}) 的深度为：{depth_val:.2f} 米")


# 初始化摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 双目摄像头需确认设备索引（可能是1）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# 检查摄像头是否成功打开
ret, frame = cap.read()
if not ret:
    print("错误：无法读取摄像头画面！")
    cap.release()
    exit()

# 切割左右画面（假设水平拼接）
frame_height, frame_width, _ = frame.shape
half_width = frame_width // 2

# 相机标定参数（替换为你的实际标定结果）
mat_inter_left = np.array([[485.31810141, 0., 299.78121813],
                          [0., 486.58790624, 278.01553071],
                          [0., 0., 1.]])
coff_dis_left = np.array([[-0.08967496, 0.20176819, 0.00913968, -0.00456322, -0.12016892]])

mat_inter_right = np.array([[494.0977399, 0., 358.30701335],
                           [0., 494.26305562, 267.99135094],
                           [0., 0., 1.]])
coff_dis_right = np.array([[-0.11759724, 0.41773862, 0.00034209, 0.00292024, -0.48796630]])

# 立体匹配参数
baseline = 0.06  # 单位：米（根据实际测量调整）
focal_length = (mat_inter_left[0, 0] + mat_inter_right[0, 0]) / 2
print(f"焦距: {focal_length:.2f} pixels")

# 初始化StereoSGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*6,  # 初始值（通过滑条调整）
    blockSize=5,
    P1=8*3*5**2,
    P2=32*3*5**2,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=50,
    speckleRange=1,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# 初始化WLS滤波器
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
wls_filter.setLambda(1200)  # 初始值（通过滑条调整）
right_matcher = cv2.ximgproc.createRightMatcher(stereo)  # 右视差计算器

# 创建显示窗口和滑条
cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
cv2.createTrackbar('numDisparities', 'Disparity', 6, 20, lambda x: None)  # 范围6-20（对应96-320）
cv2.createTrackbar('lambda', 'Disparity', 12, 50, lambda x: None)        # 范围12-50（对应1200-5000）
cv2.createTrackbar('minDepth', 'Depth Map', 10, 100, lambda x: None)     # 最小深度（0.1m~10m）

try:
    while True:
        # 获取滑条参数
        num_disp = max(16, cv2.getTrackbarPos('numDisparities', 'Disparity') * 16)  # 确保≥16
        lambda_val = max(100, cv2.getTrackbarPos('lambda', 'Disparity') * 100)     # 确保≥100
        min_depth = max(0.1, cv2.getTrackbarPos('minDepth', 'Depth Map') / 10)     # 转换为米

        # 更新算法参数
        stereo.setNumDisparities(num_disp)
        wls_filter.setLambda(lambda_val)

        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("警告：帧读取失败，重试...")
            continue

        # 切割左右图像 + 去畸变
        left_frame = frame[:, :half_width]
        right_frame = frame[:, half_width:]
        left_undist = cv2.undistort(left_frame, mat_inter_left, coff_dis_left)
        right_undist = cv2.undistort(right_frame, mat_inter_right, coff_dis_right)

        # 转换为灰度图
        gray_left = cv2.cvtColor(left_undist, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_undist, cv2.COLOR_BGR2GRAY)

        # 计算视差图
        left_disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0  # 转换为真实视差
        right_disp = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0

        # WLS滤波
        filtered_disp = wls_filter.filter(left_disp, gray_left, None, right_disp)
        filtered_disp = cv2.medianBlur(filtered_disp, 5)  # 中值滤波去噪

        # 计算深度图
        disp_no_zero = np.where(filtered_disp == 0, 1e-5, filtered_disp)
        depth_map = (baseline * focal_length) / disp_no_zero
        depth_map = np.clip(depth_map, min_depth, 10)  # 限制深度范围

        # 归一化显示
        disp_norm = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

        # 在左图画中心点深度
        h, w = filtered_disp.shape
        center_x, center_y = w//2, h//2
        center_depth = depth_map[center_y, center_x]
        cv2.circle(left_undist, (center_x, center_y), 5, (0, 0, 255), 2)
        cv2.putText(left_undist, f"Depth: {center_depth:.2f}m",
                   (center_x+10, center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


        # 显示结果
        cv2.imshow("Left Camera", left_undist)
        cv2.imshow("Disparity", disp_norm.astype(np.uint8))
        cv2.imshow("Depth Map", depth_norm.astype(np.uint8))
        cv2.setMouseCallback("Left Camera", on_mouse_click, depth_map)

        # 按Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"程序异常: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()