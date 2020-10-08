#include "utils.h"
//Image
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

Utils::Utils(){

}
bool Utils::is_element_in_vector(std::vector<std::pair<double, double>> v, std::pair<double, double> element){
	std::vector< std::pair<double, double>>::iterator it;
	it = find(v.begin(), v.end(), element);
	if (it != v.end()) {
		return true;
	}
	else {
		return false;
	}
};

std::pair<double, double> Utils::cal_center_point(std::vector<double> box) {
	return { ((box[2] - box[0]) / 2) + box[0] , ((box[3] - box[1]) / 2) + box[1] };
}

void drawboundary(cv::Mat image, double image_width_pixel, double image_height_pixel, double box_num) {

	double x_unit = image_width_pixel / box_num;
	double y_unit = image_height_pixel / box_num;
	cv::Scalar colorLine(0, 255, 0); // Green
	int lineThickness = 2;

	for (int i = 0; i < box_num + 1; i++) {
		cv::Point pt1, pt2, pt3, pt4;

		pt1.x = i * x_unit; pt1.y = 0;
		pt2.x = i * x_unit; pt2.y = box_num * y_unit;
		pt3.x = 0; pt3.y = i * y_unit;
		pt4.x = box_num * x_unit; pt4.y = i * y_unit;
		cv::line(image, pt1, pt2, colorLine, lineThickness);
		cv::line(image, pt3, pt4, colorLine, lineThickness);
	}
	return;
}


bool Utils::is_int_element_in_vector(std::vector<int> v, int element) {
	std::vector<int>::iterator it;
	it = find(v.begin(), v.end(), element);
	if (it != v.end()) {
		return true;
	}
	else {
		return false;
	}
};


/*
bool is_element_in_vector(std::vector< std::pair<double, double>> v, std::pair<double, double> element) {
	std::vector< std::pair<double, double>>::iterator it;
	it = find(v.begin(), v.end(), element);
	if (it != v.end()) {
		return true;
	}
	else {
		return false;
	}
}
*/

