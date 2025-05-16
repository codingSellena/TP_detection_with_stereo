import os
import numpy as np
import cv2
import glob


# 相机参数标定得到相机的内参矩阵，畸变矩阵
def calib(inter_corner_shape, size_per_grid, img_dir_left, img_dir_right, img_type):
    w, h = inter_corner_shape
    # cpp_int：int形式的角点，以“ int”形式保存世界空间中角点的坐标
    cp_int = np.zeros((w * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    # cp_world：世界空间中的角点，保存世界空间中角点的坐标。
    cp_world = cp_int * size_per_grid

    obj_points = []  # 世界空间中的点
    img_points_left = []  # 左图像空间中的点
    img_points_right = []  # 右图像空间中的点

    # 获取左右图像的路径
    images_left = glob.glob(img_dir_left + os.sep + '**.' + img_type)
    images_right = glob.glob(img_dir_right + os.sep + '**.' + img_type)

    for fname_left, fname_right in zip(images_left, images_right):
        img_name_left = fname_left.split(os.sep)[-1]
        img_name_right = fname_right.split(os.sep)[-1]

        # 读取左右图像
        img_left = cv2.imread(fname_left)
        img_right = cv2.imread(fname_right)

        gray_img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 查找左右图像中的棋盘格角点
        ret_left, cp_img_left = cv2.findChessboardCorners(gray_img_left, (w, h), None)
        ret_right, cp_img_right = cv2.findChessboardCorners(gray_img_right, (w, h), None)

        # 如果左右都找到角点，保存角点
        if ret_left and ret_right:
            obj_points.append(cp_world)
            img_points_left.append(cp_img_left)
            img_points_right.append(cp_img_right)

            # 显示角点
            cv2.drawChessboardCorners(img_left, (w, h), cp_img_left, ret_left)
            cv2.drawChessboardCorners(img_right, (w, h), cp_img_right, ret_right)

            # 可选：保存带角点标注的图像
            cv2.imwrite(save_dir1 + os.sep + "left_" + img_name_left, img_left)
            cv2.imwrite(save_dir1 + os.sep + "right_" + img_name_right, img_right)
            cv2.imshow('FoundCorners', np.hstack([img_left, img_right]))
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # 校准相机
    ret, mat_inter_left, coff_dis_left, v_rot_left, v_trans_left = cv2.calibrateCamera(obj_points, img_points_left,
                                                                                       gray_img_left.shape[::-1], None,
                                                                                       None)
    ret, mat_inter_right, coff_dis_right, v_rot_right, v_trans_right = cv2.calibrateCamera(obj_points, img_points_right,
                                                                                           gray_img_right.shape[::-1],
                                                                                           None, None)

    print("Left Camera:")
    print("ret:", ret)
    print("internal matrix:\n", mat_inter_left)
    print("distortion coefficients:\n", coff_dis_left)
    print("rotation vectors:\n", v_rot_left)
    print("translation vectors:\n", v_trans_left)

    print("Right Camera:")
    print("ret:", ret)
    print("internal matrix:\n", mat_inter_right)
    print("distortion coefficients:\n", coff_dis_right)
    print("rotation vectors:\n", v_rot_right)
    print("translation vectors:\n", v_trans_right)

    # 计算重新投影的误差
    total_error_left = 0
    total_error_right = 0
    for i in range(len(obj_points)):
        img_points_repro_left, _ = cv2.projectPoints(obj_points[i], v_rot_left[i], v_trans_left[i], mat_inter_left,
                                                     coff_dis_left)
        img_points_repro_right, _ = cv2.projectPoints(obj_points[i], v_rot_right[i], v_trans_right[i], mat_inter_right,
                                                      coff_dis_right)

        error_left = cv2.norm(img_points_left[i], img_points_repro_left, cv2.NORM_L2) / len(img_points_repro_left)
        error_right = cv2.norm(img_points_right[i], img_points_repro_right, cv2.NORM_L2) / len(img_points_repro_right)

        total_error_left += error_left
        total_error_right += error_right

    print("Average Error of Reprojection (Left Camera): ", total_error_left / len(obj_points))
    print("Average Error of Reprojection (Right Camera): ", total_error_right / len(obj_points))

    return mat_inter_left, coff_dis_left, mat_inter_right, coff_dis_right


# 根据得到的内参矩阵，畸变矩阵来进行图像纠偏
def dedistortion(inter_corner_shape, img_dir_left, img_dir_right, img_type, save_dir, mat_inter_left, coff_dis_left,
                 mat_inter_right, coff_dis_right):
    (w, h) = (640, 480)

    # 获取左右图像的路径
    images_left = glob.glob(img_dir_left + os.sep + '**.' + img_type)
    images_right = glob.glob(img_dir_right + os.sep + '**.' + img_type)

    for fname_left, fname_right in zip(images_left, images_right):
        img_name_left = fname_left.split(os.sep)[-1]
        img_name_right = fname_right.split(os.sep)[-1]

        # 读取左右图像
        img_left = cv2.imread(fname_left)
        img_right = cv2.imread(fname_right)

        # 去畸变
        dst_left = cv2.undistort(img_left, mat_inter_left, coff_dis_left)
        dst_right = cv2.undistort(img_right, mat_inter_right, coff_dis_right)

        # 可选：裁剪图像
        newcameramtx_left, roi_left = cv2.getOptimalNewCameraMatrix(mat_inter_left, coff_dis_left, (w, h), 0, (w, h))
        newcameramtx_right, roi_right = cv2.getOptimalNewCameraMatrix(mat_inter_right, coff_dis_right, (w, h), 0,
                                                                      (w, h))

        # 保存去畸变后的图像
        cv2.imwrite(save_dir + os.sep + "left_" + img_name_left, dst_left)
        cv2.imwrite(save_dir + os.sep + "right_" + img_name_right, dst_right)

        # 显示去畸变图像
        cv2.imshow('Left Image', dst_left)
        cv2.imshow('Right Image', dst_right)
        cv2.waitKey(1)

    print('Dedistorted images have been saved to: %s successfully.' % save_dir)


if __name__ == '__main__':
    # 棋盘格格点
    inter_corner_shape = (8, 5)
    # 格点尺寸
    size_per_grid = 0.021
    # 左右图像文件夹
    img_dir_left = ".\\left0411"
    img_dir_right = ".\\right0411"
    img_type = "jpg"
    # 保存识别角点文件夹
    save_dir1 = ".\\save_coner0411"
    # 相机标定
    mat_inter_left, coff_dis_left, mat_inter_right, coff_dis_right = calib(inter_corner_shape, size_per_grid,
                                                                           img_dir_left, img_dir_right, img_type)
    # 保存矫正后图像文件夹
    save_dir = ".\\save_dedistortion0411"
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 进行去畸变处理
    dedistortion(inter_corner_shape, img_dir_left, img_dir_right, img_type, save_dir, mat_inter_left, coff_dis_left,
                 mat_inter_right, coff_dis_right)
