import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import cv2


# 分别找左右两幅图的角点
def stereo_calib_find_corners(imgL, imgR, rlt_dir_L, rlt_dir_R, img_idx, col, row):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    retL, cornersL = cv2.findChessboardCorners(grayL, (col, row), None)
    print("retL = ", retL)
    # 为了得到稍微精确一点的角点坐标，进一步对角点进行亚像素寻找
    corners2L = cv2.cornerSubPix(grayL, cornersL, (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.001))

    if retL == True:
        # 保存角点图像
        sav_pathL = rlt_dir_L + "\\" + str(img_idx) + "_corner.jpg"
        cv2.drawChessboardCorners(imgL, (col, row), corners2L, retL)
        cv2.imwrite(sav_pathL, imgL)

    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    retR, cornersR = cv2.findChessboardCorners(grayR, (col, row), None)
    print("retR = ", retR)
    # 为了得到稍微精确一点的角点坐标，进一步对角点进行亚像素寻找
    corners2R = cv2.cornerSubPix(grayR, cornersR, (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.001))

    if retR == True:
        # 保存角点图像
        sav_pathR = rlt_dir_R + "\\" + str(img_idx) + "_corner.jpg"
        cv2.drawChessboardCorners(imgR, (col, row), corners2R, retR)
        cv2.imwrite(sav_pathR, imgR)

    return (retL, corners2L, retR, corners2R)


def stereo_calib_calibrate(img_dir_L, img_dir_R, row_num, col_num, calib_img_num, square_sz):
    w = 1440
    h = 1080
    all_cornersL = []
    all_cornersR = []
    patterns = []
    # 标定相机,先搞这么一个假想的板子，标定就是把图像中的板子往假想的板子上靠，靠的过程就是计算参数的过程
    # 比如说(0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    pattern_points = np.zeros((7 * 6, 3), np.float32)
    pattern_points[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    pattern_points *= square_sz

    for i in range(0, calib_img_num):
        img_pathL = img_dir_L + "\\left_" + str(i) + ".jpg"
        img_pathR = img_dir_R + "\\right_" + str(i) + ".jpg"
        print(img_pathL)
        print(img_pathR)
        # 读取图像
        imgL = cv2.imread(img_pathL)
        imgR = cv2.imread(img_pathR)
        # 把图像搞成统一大小1440*1080
        imgLL = cv2.resize(imgL, (w, h))
        imgRR = cv2.resize(imgR, (w, h))
        # 提取角点
        rlt_dir_L = img_dir_L + "\\corners"
        rlt_dir_R = img_dir_R + "\\corners"
        retl, cornersl, retr, cornersr = stereo_calib_find_corners(imgLL, imgRR, rlt_dir_L, rlt_dir_R, i, col_num,
                                                                   row_num)

        # 合并所有角点
        all_cornersL.append(cornersl)
        all_cornersR.append(cornersr)
        patterns.append(pattern_points)

    rmsL, cameraMatrixL, distCoeffsL, rvecsL, tvecsL = cv2.calibrateCamera(patterns, all_cornersL, (w, h), None, None)
    rmsR, cameraMatrixR, distCoeffsR, rvecsR, tvecsR = cv2.calibrateCamera(patterns, all_cornersR, (w, h), None, None)

    # 计算两个相机之间的关系矩阵模型
    flags = 0
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1, 1e-5)
    ret, Ml, dl, Mr, dr, R, T, E, F = cv2.stereoCalibrate(patterns,
                                                          all_cornersL,
                                                          all_cornersR,
                                                          cameraMatrixL,
                                                          distCoeffsL,
                                                          cameraMatrixR,
                                                          distCoeffsR,
                                                          (w, h),
                                                          criteria=stereocalib_criteria,
                                                          flags=flags)

    print('Intrinsic_mtx_l', Ml)
    print('dist_l', dl)
    print('Intrinsic_mtx_r', Mr)
    print('dist_r', dr)
    print('R', R)
    print('T', T)
    print('E', E)
    print('F', F)

    camera_model = dict([('Ml', Ml), ('Mr', Mr), ('dl', dl),
                         ('dr', dr), ('rl', rvecsL),
                         ('rr', rvecsR), ('R', R), ('T', T),
                         ('E', E), ('F', F)])
    return camera_model


def stereo_calib_correct(crct_img_L_dir, crct_img_R_dir, corct_img_num, cameraModel, rlt_dir):
    w = 1440
    h = 1080
    for i in range(1, corct_img_num):
        img_pathL = crct_img_L_dir + "\\" + str(i) + ".jpg"
        img_pathR = crct_img_R_dir + "\\" + str(i) + ".jpg"
        print(img_pathL)
        print(img_pathR)
        # 读取图像
        imgL = cv2.imread(img_pathL)
        imgR = cv2.imread(img_pathR)
        # 把图像搞成统一大小1440*1080
        imgLL = cv2.resize(imgL, (w, h))
        imgRR = cv2.resize(imgR, (w, h))
        grayL = cv2.cvtColor(imgLL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgRR, cv2.COLOR_BGR2GRAY)

        # 极线矫正，也就是把两幅图的极线搞成水平，就像在博客中描述的那样，把任意位置的像平面，搞成两个平行的像平面
        Rl, Rr, Pl, Pr, Q, validPixROIl, validPixROIr = cv2.stereoRectify(cameraModel['Ml'], cameraModel['dl'],
                                                                          cameraModel['Mr'], cameraModel['dr'], (w, h),
                                                                          cameraModel['R'], cameraModel['T'])
        '''
        mapl_1, mapl_2 = cv2.initUndistortRectifyMap(cameraModel['Ml'], cameraModel['dl'], Rl, Pl, (w,h), cv2.CV_32FC1)
        mapr_1, mapr_2 = cv2.initUndistortRectifyMap(cameraModel['Mr'], cameraModel['dr'], Rr, Pr, (w,h), cv2.CV_32FC1)
        rltl = cv2.remap(grayL, mapl_1, mapl_2, cv2.INTER_LINEAR)
        rltr = cv2.remap(grayR, mapr_1, mapr_2, cv2.INTER_LINEAR)
        '''
        newcameramtxL, roiL = cv2.getOptimalNewCameraMatrix(cameraModel['Ml'], cameraModel['dl'], (w, h), 1, (w, h))
        newcameramtxR, roiR = cv2.getOptimalNewCameraMatrix(cameraModel['Mr'], cameraModel['dr'], (w, h), 1, (w, h))
        dstL = cv2.undistort(grayL, cameraModel['Ml'], cameraModel['dl'], None, newcameramtxL)
        dstR = cv2.undistort(grayR, cameraModel['Mr'], cameraModel['dr'], None, newcameramtxR)

        # 保存极限图
        recPath = rlt_dir + "\\e_line.jpg"
        rlt = np.concatenate((dstL, dstR), axis=1)
        rlt[::40, :] = 0
        cv2.imwrite(recPath, rlt)

        # 生成深度图
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(grayL, grayR)  # 用原始图像

        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)
        cv2.imwrite(rlt_dir + "\\depth.jpg", disp)

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取标定的图片并保存结果", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--img_dir_L", help="左边相机标定图片路径", type=str, metavar='',
                        default="D:\\a大创\\双目标定\\left0411")
    parser.add_argument("--img_dir_R", help="右边相机标定图片路径", type=str, metavar='',
                        default="D:\\a大创\\双目标定\\right0411")
    parser.add_argument("--crct_img_L_dir", help="测试图像路径", type=str, metavar='', default="D:\\newdata\\corct_L")
    parser.add_argument("--crct_img_R_dir", help="测试图像路径", type=str, metavar='', default="D:\\newdata\\corct_R")
    parser.add_argument("--rlt_dir", help="测试图像路径", type=str, metavar='', default="D:\\newdata\\rlt")
    parser.add_argument("--row_num", help="每一行有多少个角点，边缘处的不算", type=int, metavar='', default="8")
    parser.add_argument("--col_num", help="每一列有多少个角点，边缘处的不算", type=int, metavar='', default="5")
    parser.add_argument("--calib_img_num", help="多少幅图像", type=int, metavar='', default="19")
    parser.add_argument("--corct_img_num", help="多少幅图像", type=int, metavar='', default="19")
    parser.add_argument("--square_sz", help="标定板放个尺寸，单位毫米", type=int, metavar='', default="21")

    args = parser.parse_args()

    # 标定相机
    cameraModel = stereo_calib_calibrate(args.img_dir_L, args.img_dir_R, args.row_num, args.col_num, args.calib_img_num,
                                         args.square_sz)

    # 矫正图片
    # ret = stereo_calib_correct(args.crct_img_L_dir, args.crct_img_R_dir, args.corct_img_num, cameraModel, args.rlt_dir)
