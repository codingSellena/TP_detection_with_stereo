
#include <iostream>
#include <string>
#include <ctime>
#include <stdio.h>
#include <omp.h>

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Module.hpp>

#include <opencv2/ximgproc/disparity_filter.hpp>
#include <httplib.h> // ( cpp-httplib
#include <json.hpp> // JSON https://github.com/nlohmann/json	
#include <mutex>
#include <fstream>

#include "utils.h"
#define use_camera 1
#define mnnd 1


// left matrix
cv::Mat mat_inter_left = (cv::Mat_<double>(3, 3) << 
    485.31810141, 0.0, 299.78121813,
    0.0, 486.58790624, 278.01553071,
    0.0, 0.0, 1.0);

//  left_dis matrix
cv::Mat coff_dis_left = (cv::Mat_<double>(1, 5) << 
    -0.08967496, 0.20176819, 0.00913968, -0.00456322, -0.12016892);

// right
cv::Mat mat_inter_right = (cv::Mat_<double>(3, 3) << 
    494.0977399, 0.0, 358.30701335,
    0.0, 494.26305562, 267.99135094,
    0.0, 0.0, 1.0);

// 
cv::Mat coff_dis_right = (cv::Mat_<double>(1, 5) << 
    -0.11759724, 0.41773862, 0.00034209, 0.00292024, -0.48796630);


//  Rotation matrix
cv::Mat R = (cv::Mat_<double>(3, 3) <<
0.99994184, -0.00311024, -0.01032727,
0.00280429,  0.9995606,  -0.02950826,
0.01041451,  0.02947758,  0.99951119);

//  T matrix
cv::Mat T = (cv::Mat_<double>(3, 1) <<
-0.05726944,
-0.00165544,
-0.00096162);

	
float baseline = 0.06f; // s
float focal_length = (mat_inter_left.at<double>(0,0) + mat_inter_right.at<double>(0,0)) / 2.0;

using json = nlohmann::json;
std::mutex mtx;
std::vector<json> latest_result;
nlohmann::json result_json = latest_result;

std::string get_current_time() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::tm tm_buf;
    #ifdef _WIN32
    localtime_s(&tm_buf, &in_time_t);  // Windows
    #else
    localtime_r(&in_time_t, &tm_buf);  // Linux/macOS
    #endif
    
    std::ostringstream ss;
    ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// 
void update_inference_result(const std::vector<json>& results) {
    std::lock_guard<std::mutex> lock(mtx);
    latest_result = results;
    result_json = { 
        {"status", "ok"},
        {"timestamp", get_current_time()},  
        {"objects", latest_result}     
    };
}


void draw_box(cv::Mat &cv_mat, std::vector<BoxInfo> &boxes, MatInfo &mmat_objection,
              cv::VideoWriter *vw,cv::VideoWriter *vw_dis, const cv::Mat& disparity_map)
{
		
	// get current time
	std::time_t now = std::time(nullptr);
	std::tm* now_tm = std::localtime(&now);
	char time_str[64];
	std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", now_tm);
	std::string timestamp(time_str); // convert std::string


    static const char* class_names[] = {
        "TP", "person", "tree", "car", "bike", "motorcycle",
        "obstacle", "obstacle_stone", "tfcone"
    };

    int x1, y1, objw, objh;
    char text[256];
    std::vector<json> frame_results;

    for (const auto& box : boxes)
    {
        x1 = int(box.x1 / mmat_objection.ratio) - int(mmat_objection.Padw / mmat_objection.ratio);
        y1 = int(box.y1 / mmat_objection.ratio) - int(mmat_objection.Padh / mmat_objection.ratio);
        objw = int((box.x2 - box.x1) / mmat_objection.ratio);
        objh = int((box.y2 - box.y1) / mmat_objection.ratio);
        cv::Point pos = cv::Point(x1, y1 - 5);

        cv::Rect rect(x1, y1, objw, objh);
        cv::rectangle(cv_mat, rect, cv::Scalar(0, 255, 0));

        // ---------- depth get ----------
        float depth = -1.0f;

        if (!disparity_map.empty() && disparity_map.type() == CV_32F) {
            // clip to box
            cv::Rect roi = rect & cv::Rect(0, 0, disparity_map.cols, disparity_map.rows);

            if (roi.area() > 0) {
                cv::Mat roi_disp = disparity_map(roi);
                std::vector<float> valid_disp;

                for (int y = 0; y < roi_disp.rows; ++y) {
                    for (int x = 0; x < roi_disp.cols; ++x) {
                        float d = roi_disp.at<float>(y, x);
                        if (d > 1e-5 && d < 1e4) valid_disp.push_back(d);
                    }
                }

                if (!valid_disp.empty()) {
                    std::nth_element(valid_disp.begin(),
                                     valid_disp.begin() + valid_disp.size() / 2,
                                     valid_disp.end());
                    float median_disp = valid_disp[valid_disp.size() / 2];
                    if (median_disp > 1e-5 && focal_length > 1.0 && baseline > 1e-3)
                        depth = (focal_length * baseline) / median_disp;
                }
            }
        }

        // ---------- text ----------
        if (depth > 0)
            sprintf(text, "%s %.1f%% %.2f m", class_names[box.label], box.score * 100, depth);
        else
            sprintf(text, "%s %.1f%%", class_names[box.label], box.score * 100);

        cv::putText(cv_mat, text, pos,
                    cv::FONT_HERSHEY_SIMPLEX,
                    ((float)objh / (float)mmat_objection.maxSide) * 5,
                    cv::Scalar(0, 0, 255), 1);

        frame_results.push_back({
            {"object", class_names[box.label]},
            {"score", box.score},
            {"depth", depth},
            {"timestamp", timestamp}
        });

    }
    update_inference_result(frame_results);



#if use_camera
    //cv::Mat saved_frame = cv_mat(cv::Rect(0, 0, cv_mat.cols / 2, cv_mat.rows));
    cv::imshow("Fourcc", cv_mat); 
    //std::cout << "cv_mat Size: " << cv_mat.size() << std::endl;
    //if (saved_frame.size() != cv::Size(320,480)) {
      //  std::cerr << "Frame size mismatch!" << std::endl;
    //}
    //vw->write(saved_frame);
    vw->write(cv_mat);
    cv::waitKey(1);
    // dis save
    if (!disparity_map.empty()) {
        //std::cout << "Disparity Map Size: " << disparity_map.size() << std::endl;
        cv::imshow("Disparity Map", disparity_map);
        if (disparity_map.size() != cv::Size(320,480)) {
            std::cerr << "Frame size mismatch!" << std::endl;
            std::cerr << "size = "<<disparity_map.size()<<std::endl;
        }
        cv::Mat color_disp;
		cv::applyColorMap(disparity_map, color_disp, cv::COLORMAP_JET); // l*ir
		vw_dis->write(color_disp);  
        cv::waitKey(1);  
    }
    
#else
    cv::imwrite("result.jpg", cv_mat);
#endif
}

void getDisparityMap(const cv::Mat& left_gray, const cv::Mat& right_gray, cv::Mat& disparity_out) {
    int blockSize = 5;
	int img_channels = 1; // 

	int minDisparity = 1;
	int numDisparities = 4*16; // 

	int P1 = 8 * img_channels * blockSize * blockSize;
	int P2 = 32 * img_channels * blockSize * blockSize;
	int disp12MaxDiff = -1;
	int preFilterCap = 140;
	int uniquenessRatio = 1;
	int speckleWindowSize = 100;
	int speckleRange = 100;

	auto left_matcher = cv::StereoSGBM::create(
		minDisparity,
		numDisparities,
		blockSize,
		P1,
		P2,
		disp12MaxDiff,
		preFilterCap,
		uniquenessRatio,
		speckleWindowSize,
		speckleRange,
		cv::StereoSGBM::MODE_HH // Jh@9M (High Quality)
	);

    auto right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

    cv::Mat disp_left, disp_right;
    left_matcher->compute(left_gray, right_gray, disp_left);
    right_matcher->compute(right_gray, left_gray, disp_right);

    auto wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
    wls_filter->setLambda(1000.0);
    wls_filter->setSigmaColor(1.5);

    cv::Mat filtered_disp;
    wls_filter->filter(disp_left, left_gray, filtered_disp, disp_right);

    // 
    cv::ximgproc::getDisparityVis(filtered_disp, disparity_out, 1.0);
}


// liti jiaozheng
cv::Mat mapLx, mapLy, mapRx, mapRy;

void initStereoRectify(int image_width, int image_height)
{
    cv::Mat R1, R2, P1, P2, Q;

    cv::stereoRectify(mat_inter_left, coff_dis_left,
                      mat_inter_right, coff_dis_right,
                      cv::Size(image_width, image_height),
                      R, T, R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0, cv::Size(), 0, 0);

    
    cv::initUndistortRectifyMap(mat_inter_left, coff_dis_left, R1, P1, 
                                cv::Size(image_width, image_height), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(mat_inter_right, coff_dis_right, R2, P2, 
                                cv::Size(image_width, image_height), CV_32FC1, mapRx, mapRy);
}



std::vector<BoxInfo> decode(cv::Mat &cv_mat, std::shared_ptr<MNN::Interpreter> &net, MNN::Session *session, int INPUT_SIZE)
{
    std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data = nhwc_Tensor->host<float>();
    auto nhwc_size = nhwc_Tensor->size();
    std::memcpy(nhwc_data, cv_mat.data, nhwc_size);

    auto inputTensor = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    net->runSession(session);
    MNN::Tensor *tensor_scores = net->getSessionOutput(session, "outputs");
    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    auto pred_dims = tensor_scores_host.shape();

#if mnnd
    const unsigned int num_proposals = pred_dims.at(1);
    const unsigned int num_classes = pred_dims.at(2) - 5;
    std::vector<BoxInfo> bbox_collection;

    for (unsigned int i = 0; i < num_proposals; ++i)
    {
        const float *offset_obj_cls_ptr = tensor_scores_host.host<float>() + (i * (num_classes + 5)); // row ptr
        float obj_conf = offset_obj_cls_ptr[4];
        if (obj_conf < 0.5)
            continue;

        float cls_conf = offset_obj_cls_ptr[5];
        unsigned int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float tmp_conf = offset_obj_cls_ptr[j + 5];
            if (tmp_conf > cls_conf)
            {
                cls_conf = tmp_conf;
                label = j;
            }
        }

        float conf = obj_conf * cls_conf; 
        if (conf < 0.50)
            continue;

        float cx = offset_obj_cls_ptr[0];
        float cy = offset_obj_cls_ptr[1];
        float w = offset_obj_cls_ptr[2];
        float h = offset_obj_cls_ptr[3];

        float x1 = (cx - w / 2.f);
        float y1 = (cy - h / 2.f);
        float x2 = (cx + w / 2.f);
        float y2 = (cy + h / 2.f);

        BoxInfo box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float)INPUT_SIZE - 1.f);
        box.y2 = std::min(y2, (float)INPUT_SIZE - 1.f);
        box.score = conf;
        box.label = label;
        bbox_collection.push_back(box);
    }
#else
    const unsigned int num_proposals = pred_dims.at(0);
    const unsigned int num_datainfo = pred_dims.at(1);
    std::vector<BoxInfo> bbox_collection;
    for (unsigned int i = 0; i < num_proposals; ++i)
    {
        const float *offset_obj_cls_ptr = tensor_scores_host.host<float>() + (i * num_datainfo); // row ptr
        float obj_conf = offset_obj_cls_ptr[4];
        if (obj_conf < 0.5)
            continue;

        float x1 = offset_obj_cls_ptr[0];
        float y1 = offset_obj_cls_ptr[1];
        float x2 = offset_obj_cls_ptr[2];
        float y2 = offset_obj_cls_ptr[3];

        BoxInfo box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float)INPUT_SIZE - 1.f);
        box.y2 = std::min(y2, (float)INPUT_SIZE - 1.f);
        box.score = offset_obj_cls_ptr[4];
        box.label = offset_obj_cls_ptr[5];
        bbox_collection.push_back(box);
    }
#endif
    delete nhwc_Tensor;
    return bbox_collection;
}

std::string getCurrentDateTimeStr() {
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(now, "%Y%m%d_%H%M%S");  
    return oss.str();
}



int main(int argc, char const *argv[])
{
    httplib::Server svr;
    
    svr.Get("/", [](const httplib::Request& req, httplib::Response& res) {
        try {
            std::ifstream file("/home/pi/mnn/src/web/index.html");
            if (!file.is_open()) {
                res.status = 404;
                res.set_content("404 Not Found", "text/plain");
                return;
            }

            std::string html((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
            res.set_content(html, "text/html");
        } catch(...) {
            res.status = 500;
            res.set_content("Internal Server Error", "text/plain");
        }
    });
    // 
    svr.Get("/infer", [](const httplib::Request& req, httplib::Response& res) {
		try {
			res.set_header("Access-Control-Allow-Origin", "*");
			res.set_header("Access-Control-Allow-Methods", "GET");
			res.set_header("Access-Control-Allow-Headers", "Content-Type");

			std::lock_guard<std::mutex> lock(mtx);
			if(latest_result.empty()) {
				 res.status = 503; 
				 res.set_content("{\"message\": \"No data available\"}", "application/json");
			} else {
				res.set_content(result_json.dump(4), "application/json");
			}
		} catch(...) {
			res.status = 500;
			res.set_content("Internal Server Error", "text/plain");
		}
	});
	
	svr.Options("/infer", [](const httplib::Request& req, httplib::Response& res) {
		res.set_header("Access-Control-Allow-Origin", "*");
		res.set_header("Access-Control-Allow-Methods", "GET, OPTIONS");
		res.set_header("Access-Control-Allow-Headers", "Content-Type");
		res.status = 200;
	});



    // 
    std::thread server_thread([&svr]() {
        std::cout << "[INFO] start application..." << std::endl;
        if (!svr.listen("0.0.0.0", 8000)) {
            std::cerr << "[ERROR] fail to start application, please consider whether the port has been taken up" << std::endl;
        }
    });

    // 
    std::cout << "[INFO] inital MNN model and camera..." << std::endl;


	
    //cv::VideoWriter outputVideo;
    //outputVideo.open("/home/pi/video.avi", -1, 15, cv::Size(1920, 1080), true);
    if(argc < 1)
    {
        printf("Usage:\n\t%s mnn_model_path\n", argv[0]);
        return -1;
    }
	
    std::string model_name = argv[1];

    //std::string model_name = "/home/pi/best_mnnd.mnn";
    auto modelName= argv[1];
  
    std::shared_ptr<MNN::Interpreter> net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_name.c_str()));
    if (nullptr == net)
    {
		printf("nullptr==net");
        return 0;
    }
    MNN::Interpreter::SessionMode mode = MNN::Interpreter::Session_Debug;
    net->setSessionMode(mode);
    //printf(net->getModelVersion());
    printf("Get Model Version:");
    printf(net->getModelVersion());
    MNN::ScheduleConfig config;
    config.numThread = 4;
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    MNN::BackendConfig backendConfig;
    //backendConfig.precision = (MNN::BackendConfig::PrecisionMode)1;
    backendConfig.precision = MNN::BackendConfig::Precision_Low_BF16;
    config.backendConfig = &backendConfig;
    
    printf("Module Create Start\n");
//    Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
    std::shared_ptr<MNN::Express::Module> model;
    MNN::Express::Module::Config mdconfig;
    mdconfig.rearrange = true;
    MNN::Express::Module *model_module=MNN::Express::Module::load(std::vector<std::string>{},std::vector<std::string>{}, modelName, &mdconfig);
    if (nullptr == &model_module)
    {
        printf("nullptr==model_module");
        return 0;
    }
    printf("Module Create Success\n");
    model.reset(model_module);
    
    printf("create Session Start\n");
    MNN::Session *session = net->createSession(config);
    if (nullptr == session) {  
		printf("create Session Fail");
    
		return -1; 
    }  
    printf("create Session Finish\n");
    std::vector<BoxInfo> bbox_collection;
    cv::Mat image;
    MatInfo mmat_objection;
    mmat_objection.inpSize = 320;

#if use_camera
    cv::VideoCapture capture;
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (fps <= 0 || fps > 60) fps = 25.0;
    capture.open(0,cv::CAP_V4L2);
    int frame_height=capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_width=capture.get(cv::CAP_PROP_FRAME_WIDTH); 
	cv::Size video_size(frame_width / 2, frame_height);
	
	std::string datetime_str = getCurrentDateTimeStr();

    std::string filename_rgb = "/home/pi/VideoTest/test_" + datetime_str + ".avi";
    std::string filename_dis = "/home/pi/VideoTest/test_dis_" + datetime_str + ".avi";
    
    cv::VideoWriter* vw = new cv::VideoWriter(
        filename_rgb,
        cv::VideoWriter::fourcc('I', '4', '2', '0'),
        fps,
        cv::Size(frame_width / 2, frame_height)
    );

    cv::VideoWriter* vw_dis = new cv::VideoWriter(
        filename_dis,
        cv::VideoWriter::fourcc('I', '4', '2', '0'),
        fps,
        cv::Size(frame_width / 2, frame_height)
    );
    
    if(vw->isOpened()==false||vw_dis->isOpened()==false)
    {
		printf("fail to open vw\n");
		return -1;
	}
	else
	{
		printf("success open vw");
	}
    cv::Mat frame;
    initStereoRectify(frame_width / 2, frame_height); 

    while (true)
    {
        bbox_collection.clear();
        
        struct timespec begin, end;
        long time;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        
        capture >> frame;
		if (frame.empty()) continue;
        //std::cout << "frame Size: " << frame.size() << std::endl;

		cv::Mat left_raw = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
		cv::Mat right_raw = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
		
        cv::Mat raw_image = frame;

        cv::Mat pimg = preprocess(raw_image, mmat_objection);
        
        bbox_collection = decode(pimg, net, session, mmat_objection.inpSize);
        nms(bbox_collection, 0.50);
        

        cv::Mat left_rectified, right_rectified;
        cv::remap(left_raw, left_rectified, mapLx, mapLy, cv::INTER_LINEAR);
        cv::remap(right_raw, right_rectified, mapRx, mapRy, cv::INTER_LINEAR);


        cv::Mat left_gray, right_gray;
        cv::cvtColor(left_rectified, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_rectified, right_gray, cv::COLOR_BGR2GRAY);

        //std::cout << "left_gray Size: " << left_gray.size() << std::endl;
        //std::cout << "right_gray Size: " << right_gray.size() << std::endl;

		cv::Mat disparity;
		getDisparityMap(left_gray, right_gray, disparity);
		
        draw_box(raw_image, bbox_collection, mmat_objection,vw,vw_dis, disparity);

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);
        if(time > 0) printf(">> Time : %lf ms\n", (double)time / 1000000);
        int key = cv::waitKey(1);
		if (key == 27 || cv::getWindowProperty("Fourcc", cv::WND_PROP_VISIBLE) < 1) break;

    }
    vw->release();
    vw_dis->release();
#else
    for (size_t i = 0; i < 100; i++)
    {
        bbox_collection.clear();

        struct timespec begin, end;
        long time;
        clock_gettime(CLOCK_MONOTONIC, &begin);

        std::string image_name = "../images/0513.jpg";
        cv::Mat raw_image = cv::imread(image_name.c_str());

        cv::Mat pimg = preprocess(raw_image, mmat_objection);

        bbox_collection = decode(pimg, net, session, mmat_objection.inpSize);

		
        nms(bbox_collection, 0.50);

        draw_box(raw_image, bbox_collection, mmat_objection);
        //draw_box(raw_image, bbox_collection, mmat_objection, vw, disparity);


        clock_gettime(CLOCK_MONOTONIC, &end);
        time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);
        if(time > 0) printf(">> Time : %lf ms\n", (double)time / 1000000);
    }
#endif
    return 0;
}
