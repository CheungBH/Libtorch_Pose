#include <torch/script.h>


// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/*class SPPE{

public:
    cv::Mat CroptheImage(cv::Mat &image , int &xmin , int &ymin , int &xmax , int &ymax);
    torch::Tensor ImagetoTensor(cv::Mat &image);
    vector<float> OutputtoOrigImg(int personid , cv::Mat &image , torch::Tensor maxid ,int &croppedimage_w ,int &croppedimage_h , int &xmin , int &ymin );
protected:
private:

};*/

namespace sppe {
	cv::Mat cropFrame(const cv::Mat &frame, const cv::Rect &b_box, const cv::Size &cv);
	torch::Tensor mat2tns(const cv::Mat &mat);
}
