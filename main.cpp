#define _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING

//Tensor
//#include <torch/extension.h>
#include <torch/torch.h>
//Image
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

//I/O
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>
//Time
#include <chrono>
#include <time.h>

#include "Darknet.h" //YOLO
#include "Sppe.h" //SPPE
#include "Hungarian.h"
#include "KalmanTracker.h"
#include "utils.h"
#include "TrackingBox.h"
#include "Img_tns.h"


Utils utils_main = Utils::Utils();

using namespace std::chrono;
#ifndef DEBUG
#define RELEASE
#endif

//MACROS
#define SCREEN_W 960
#define SCREEN_H 540
#define YOLO_TENSOR_W 416
#define YOLO_TENSOR_H 416
#define YOLO_TENSOR_C 3
#define YOLO_TENSOR_N 1

#define B_BOX_ENLARGE_SCALE 0.2f

#define SPPE_TENSOR_W 256
#define SPPE_TENSOR_H 320
#define SPPE_TENSOR_C 3

//typedef struct TrackingBox {
//	int frame;
//	int id;
//	cv::Rect_<float> box;
//}TrackingBox;    

//new Sort
int max_age = 90;//max time object disappear
int min_hits = 3; //min time target appear
double iouThreshold = 0.3;//matching IOU
double resize_ratio;

struct MatchItems {
	std::set<int> unmatchedDet;
	std::set<int> unmatchedTracker;
	std::vector<cv::Point> matchedPairs;
};



double GetIOU(cv::Rect_<float> bb_dr, cv::Rect_<float> bb_gt) {
	float in = (bb_dr & bb_gt).area();
	float un = bb_dr.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	double iou = in / un;

	return iou;
}

int boundary(int n, int lower, int upper)
{
	return (n > upper ? upper : (n < lower ? lower : n));
}


#define CNUM 20

std::vector<KalmanTracker> trackers;
std::vector<std::vector<TrackingBox>> detFrameData;
std::vector<TrackingBox> SORT(std::vector<vector<float>> bbox, int fi);





std::vector<cv::Rect_<float>> get_predictions() {
	std::vector<cv::Rect_<float>> predBoxes;
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		cv::Rect_<float> pBox = (*it).predict();
		//std::cout << pBox.x << " " << pBox.y << std::endl;
		if (pBox.x >= 0 && pBox.y >= 0)
		{
			predBoxes.push_back(pBox);
			it++;
		}
		else
		{
			it = trackers.erase(it);
			//cerr << "Box invalid at frame: " << frame_count << endl;
		}
	}
	return predBoxes;
}

MatchItems Sort_match(std::vector<std::vector<TrackingBox>> detFrameData, int __, std::vector<cv::Rect_<float>> predictedBoxes) {
	int f_num = detFrameData.size() - 1;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;
	trkNum = predictedBoxes.size();
	detNum = detFrameData[f_num].size();

	std::vector<std::vector<double>> iouMatrix;
	std::vector<int> assignment;

	std::set<int> unmatchedDetections;
	std::set<int> unmatchedTrajectories;
	std::set<int> allItems;
	std::set<int> matchedItems;
	// result
	std::vector<cv::Point> matchedPairs;
	MatchItems matched_result;

	iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));


	for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
	{
		for (unsigned int j = 0; j < detNum; j++)
		{
			// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
			iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[f_num][j].box);
		}
	}

	// solve the assignment problem using hungarian algorithm.
	// the resulting assignment is [track(prediction) : detection], with len=preNum
	HungarianAlgorithm HungAlgo;
	HungAlgo.Solve(iouMatrix, assignment);

	// find matches, unmatched_detections and unmatched_predictions
	if (detNum > trkNum) //	there are unmatched detections
	{
		for (unsigned int n = 0; n < detNum; n++)
			allItems.insert(n);

		for (unsigned int i = 0; i < trkNum; ++i)
			matchedItems.insert(assignment[i]);

		// calculate the difference between allItems and matchedItems, return to unmatchedDetections
		std::set_difference(allItems.begin(), allItems.end(),
			matchedItems.begin(), matchedItems.end(),
			insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
	}
	else if (detNum < trkNum) // there are unmatched trajectory/predictions
	{
		for (unsigned int i = 0; i < trkNum; ++i)
			if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
				unmatchedTrajectories.insert(i);
	}
	else
		;

	// filter out matched with low IOU
	// output matchedPairs
	for (unsigned int i = 0; i < trkNum; ++i)
	{
		if (assignment[i] == -1) // pass over invalid values
			continue;
		if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
		{
			unmatchedTrajectories.insert(i);
			unmatchedDetections.insert(assignment[i]);
		}
		else
			matchedPairs.push_back(cv::Point(i, assignment[i]));
	}
	matched_result.matchedPairs = matchedPairs;
	matched_result.unmatchedDet = unmatchedDetections;
	matched_result.unmatchedTracker = unmatchedTrajectories;
	return matched_result;
};


std::vector<TrackingBox> update_trackers(int _, MatchItems M_items) {
	int f_num = detFrameData.size() - 1;
	std::vector<TrackingBox> Sort_result;
	std::vector<cv::Point> matchedPairs = M_items.matchedPairs;
	std::set<int> unmatchedDetections = M_items.unmatchedDet;

	int detIdx, trkIdx;
	for (unsigned int i = 0; i < matchedPairs.size(); i++)
	{
		trkIdx = matchedPairs[i].x;
		detIdx = matchedPairs[i].y;
		trackers[trkIdx].update(detFrameData[f_num][detIdx].box);
	}

	// create and initialize new trackers for unmatched detections
	for (auto umd : unmatchedDetections)
	{
		KalmanTracker tracker = KalmanTracker(detFrameData[f_num][umd].box);
		trackers.push_back(tracker);
	}

	// get trackers' output
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		if (((*it).m_time_since_update < 1) &&
			((*it).m_hit_streak >= min_hits || f_num <= min_hits))
		{
			TrackingBox res;
			res.box = (*it).get_state();
			res.id = (*it).m_id + 1;
			res.frame = f_num;
			Sort_result.push_back(res);
			it++;
		}
		else
			it++;

		// remove dead tracklet
		if (it != trackers.end() && (*it).m_time_since_update > max_age)
			it = trackers.erase(it);
	}

	//std::cout << "SORT time : " << duration.count() << " ms" << std::endl;

	return Sort_result;
};


void update_dataFrame(int f_num, std::vector<vector<float>> bbox) {
	std::vector<TrackingBox> detData;
	for (int i = 0; i < bbox.size(); i++) {
		TrackingBox tb;
		tb.frame = f_num + 1;
		tb.box = Rect_<float>(cv::Point_<float>(bbox[i][0], bbox[i][1]), cv::Point_<float>(bbox[i][2], bbox[i][3]));
		detData.push_back(tb);
	}
	detFrameData.push_back(detData);
}

std::vector<TrackingBox> get_first_frame_result(int __) {
	int f_num = detFrameData.size() - 1;
	std::vector<TrackingBox> first_frame;
	for (unsigned int i = 0; i < detFrameData[f_num].size(); i++) {
		KalmanTracker trk = KalmanTracker(detFrameData[f_num][i].box);
		trackers.push_back(trk);
	}
	// output the first frame detections
	for (unsigned int id = 0; id < detFrameData[f_num].size(); id++) {
		TrackingBox tb = detFrameData[f_num][id];
		tb.id = id;
		first_frame.push_back(tb);
		//std::cout << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height  << std::endl;
	}
	return first_frame;
};





int main() {

	torch::DeviceType device_type;

	if (torch::cuda::is_available()) {
		device_type = torch::kCUDA;
		std::cout << "GPU--version" << endl;
	}
	else {
		device_type = torch::kCPU;
		std::cout << "CPU--version" << endl;
	}
	torch::Device device(device_type);

	Darknet yolo("../models/yolo/1008/yolov3-original-1cls-leaky.cfg", &device);
	std::map<std::string, std::string> *info = yolo.get_net_info();
	(*info)["height"] = "416";
	yolo.load_weights("../models/yolo/1008/best.weights");
	yolo.to(device);

	//torch::jit::script::Module sppeModule = torch::jit::load("../models/sppe/posemodel.pt");
	//sppeModule.to(at::kCUDA);

	//torch::jit::script::Module tcn_module = torch::jit::load("../models/TCN/tcn_model.pt",device);
	//torch::jit::script::Module tcn_module = torch::jit::load("../models/TCN/tcn_model.pt");
	//tcn_module.to(at::kCUDA);


	torch::NoGradGuard no_grad;//NO feedback


	//cv::VideoCapture vc(0);
	cv::VideoCapture vc1("../videos/ceiling_long_video.mp4");
	cv::VideoCapture vc2("../videos/ceiling_long_video.mp4");
	cv::VideoCapture vc3("../videos/ceiling_long_video.mp4");
	cv::VideoCapture vc4("../videos/ceiling_long_video.mp4");


	//initialize data containers
	cv::Mat frame1, frame2, frame3, frame4; 

	cv::Size input_img_sz(YOLO_TENSOR_W, YOLO_TENSOR_H);
	cv::Size cropped_img_sz(SPPE_TENSOR_W, SPPE_TENSOR_H);
	cv::Size full_screen(SCREEN_W, SCREEN_H);
	int b_box0[30][34];
	int cnt = 0;
	KalmanTracker::kf_count = 0;

	double image_height_pixel = SCREEN_H;
	double image_width_pixel = SCREEN_W;

	double box_num = 10; // 10 x 10
	double w_num = 10;
	double h_num = 10;
	double width = image_width_pixel / box_num;
	double height = image_height_pixel / box_num;
	bool write = false;
	cv::Mat im_raw(image_height_pixel, image_width_pixel, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Scalar grey_value(128, 128, 128);
	cv::Mat padded_img(416, 416, CV_8UC3, grey_value);


	int fi = 0;
	//loop
	while (vc1.isOpened()/*Check if any image is needed to process*/)
	{

		vc1 >> frame1;
		vc2 >> frame2;
		vc3 >> frame3;
		vc4 >> frame4;
		if (frame1.empty())
		{
			break;
		}

		cv::resize(frame1, frame1, full_screen);
		cv::resize(frame2, frame2, full_screen);
		cv::resize(frame3, frame3, full_screen);
		cv::resize(frame4, frame4, full_screen);

		cv::Mat im_cnt = im_raw.clone();
		cv::Mat frame = frame1;
		cv::Mat s_frame = frame.clone();

		double orig_w = frame.cols;
		double orig_h = frame.rows;

		if (frame.cols > frame.rows)
		{
			resize_ratio = (double)416 / (double)frame.cols;
		}
		else
		{
			resize_ratio = (double)416 / (double)frame.rows;
		}

		torch::Tensor input_tensor_yolo = yolo_tns(yolo_img(frame.clone(), padded_img, resize_ratio)).to(device);

		auto yolo_start = std::chrono::high_resolution_clock::now();
		torch::Tensor output_tensor_yolo = yolo.write_results(yolo.forward(input_tensor_yolo), 1, 0.6, 0.4).to(torch::kCPU); //Process input tensor
		auto yolo_end = std::chrono::high_resolution_clock::now();
#ifdef DEBUG
		std::cout << "yolo GPU process..." << std::endl;
#endif
		auto yolo_duration = std::chrono::duration_cast<std::chrono::milliseconds>(yolo_end - yolo_start);
		std::cout << "Time taken for yolo: " << yolo_duration.count() << " ms" << endl;
		std::cout << "yolo output tensor: " << output_tensor_yolo << std::endl;

#ifdef DEBUG
		std::cout << "yolo CPU post-process..." << std::endl;
#endif

		auto yolo_cpu_operation_start = std::chrono::high_resolution_clock::now();
		//torch::Tensor->vector<cv::Rect>
		std::vector<cv::Rect> b_boxes;
		std::vector<cv::Rect> b_boxes_modified;
		bool regconized_b_box = (output_tensor_yolo.dim() > 1 ? true : false);
		if (regconized_b_box)
		{
#ifdef DEBUG
			std::cout << "Converting bounding boxes..." << std::endl;
#endif
			double resized_w = resize_ratio * orig_w;
			double resized_h = resize_ratio * orig_h;
			output_tensor_yolo.select(1, 1).add_(-(double)0.5*((double)YOLO_TENSOR_W - (resize_ratio * orig_w))).mul_((double)frame.cols / (double)resized_w);
			output_tensor_yolo.select(1, 2).add_(-(double)0.5*((double)YOLO_TENSOR_H - (resize_ratio * orig_h))).mul_((double)frame.rows / (double)resized_h);
			output_tensor_yolo.select(1, 3).add_(-(double)0.5*((double)YOLO_TENSOR_W - (resize_ratio * orig_w))).mul_((double)frame.cols / (double)resized_w);
			output_tensor_yolo.select(1, 4).add_(-(double)0.5*((double)YOLO_TENSOR_H - (resize_ratio * orig_h))).mul_((double)frame.rows / (double)resized_h);


			auto data_yolo = output_tensor_yolo.accessor<float, 2>();

			for (int i = 0; i < output_tensor_yolo.size(0); i++)
			{
				int temp[4], temp2[4];
				int init_w = data_yolo[0][3] - data_yolo[0][1];
				int init_h = data_yolo[0][4] - data_yolo[0][2];
				for (int j = 1; j < 5; j++)
				{
					temp[j - 1] = boundary(data_yolo[i][j], 0, (j % 2 == 1 ? frame.cols - 1 : frame.rows - 1));
					if (j % 2 == 1)
					{
						if (0.5*j < 1)
							temp2[j - 1] = boundary(data_yolo[i][j] - (int)(B_BOX_ENLARGE_SCALE*0.5*init_w), 0, frame.cols - 1);
						else
							temp2[j - 1] = boundary(data_yolo[i][j] + (int)(B_BOX_ENLARGE_SCALE*0.5*init_w), 0, frame.cols - 1);
					}
					else
					{
						if (0.4*j < 1)
							temp2[j - 1] = boundary(data_yolo[i][j] - (int)(B_BOX_ENLARGE_SCALE*0.5*init_h), 0, frame.rows - 1);
						else
							temp2[j - 1] = boundary(data_yolo[i][j] + (int)(B_BOX_ENLARGE_SCALE*0.5*init_h), 0, frame.rows - 1);
					}
				}

				cv::Rect b_box(temp[0], temp[1], temp[2] - temp[0], temp[3] - temp[1]);
				b_boxes.push_back(b_box);
				cv::Rect b_box_modified(temp2[0], temp2[1], temp2[2] - temp2[0], temp2[3] - temp2[1]);
				b_boxes_modified.push_back(b_box_modified);
				cv::rectangle(s_frame, b_box_modified, cv::Scalar(255, 0, 255), 4, 4);

				}

			std::vector<std::vector<float>> untracked_boxes;
			for (auto &box : b_boxes_modified) {
				float x = box.x;
				float y = box.y;
				std::vector<float> box_vec = { x, y, x + box.width, y + box.height };
				untracked_boxes.push_back(box_vec);
			}

			auto SORT_start = std::chrono::high_resolution_clock::now();

			std::vector<TrackingBox> frameTrackingResult;
			update_dataFrame(fi, untracked_boxes);

			if (trackers.size() == 0) {
				frameTrackingResult = get_first_frame_result(fi);
			}
			else {
				std::vector<cv::Rect_<float>> predictedBoxes = get_predictions();
				MatchItems matched_items = Sort_match(detFrameData, fi, predictedBoxes);
				frameTrackingResult = update_trackers(fi, matched_items);
			}

			auto SORT_end = std::chrono::high_resolution_clock::now();
			auto SORT_duration = duration_cast<milliseconds>(SORT_end - SORT_start);
			std::cout << "Time taken for SORT " << SORT_duration.count() << " ms" << endl;

			for (auto tb : frameTrackingResult) {
				string num = std::to_string(tb.id);
				cv::Point pt = cv::Point(tb.box.x, tb.box.y);
				cv::putText(s_frame, num, pt, cv::FONT_HERSHEY_DUPLEX, 4.0, cv::Scalar(0, 255, 255), 2);
			}


			}




		cv::imshow("Result", s_frame);
		cv::waitKey(1);
		fi++;



		/*
		torch::Tensor input_tensor_sppe = sppe_tns(c_frames).to(device);

#ifdef DEBUG
			std::cout << "SPPE CPU process done\nSPPE GPU process..." << std::endl;
#endif

			auto sppe_start = std::chrono::high_resolution_clock::now();
			torch::Tensor output_tensor_sppe = sppeModule.forward({ input_tensor_sppe }).toTensor().to(torch::kCPU);
			auto sppe_end = std::chrono::high_resolution_clock::now();
#ifdef DEBUG
			std::cout << "SPPE GPU process done" << std::endl;
#endif
			auto sppe_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sppe_end - sppe_start);
			std::cout << "Time taken for sppe: " << sppe_duration.count() << " ms" << endl;

			auto data_sppe = std::get<1>(torch::max(output_tensor_sppe.flatten(2, 3), 2));

			int i = 0;
			for (auto itr = list.itrWarningListBegin(); itr != list.itrWarningListEnd(); itr++)
			{
				float temp[34];
				for (int j = 0; j < 17; j++)
				{
					int gg = (int)(data_sppe[i][j].item().toFloat());
					int x = (gg % 64); //0 <= x <= 79
					int y = (gg / 64); //0 <= y <= 63

					temp[2 * j] = bbb[i].x + (int)(bbb[i].width*x / 64);
					temp[2 * j + 1] = bbb[i].y + (int)(bbb[i].height*y / 80);

					cv::Point p(temp[2 * j], temp[2 * j + 1]);
					cv::circle(frame, p, b_boxes[i].width / 60, cv::Scalar(0, 0, 255, 1), 1);
				}
				list.updateSkeletons(itr->second, skeleton(temp));
				i++;
			}

			auto sppe_cpu_operation_end = std::chrono::high_resolution_clock::now();
			auto sppe_cpu_operation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sppe_cpu_operation_end - sppe_cpu_operation_start);
			std::cout << "Time taken for sppe_cpu_operation: " << sppe_cpu_operation_duration.count() << " ms" << endl;
			*/
			//}
	}
	//uninitialization	
	return 0;
}

