#define IMG_TNS_H
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <vector>
cv::Mat& yolo_img(cv::Mat& img, cv::Mat& padded_img, double& resize_ratio, bool greyScale = false)
{
	cv::Size sz(416, 416);

	double new_w = (double)img.cols * resize_ratio;
	double new_h = (double)img.rows * resize_ratio;
	double padded_x;
	double padded_y;
	if (img.cols > img.rows)
	{
		padded_x = 0;
		padded_y = (0.5)*((double)416 - new_h);
		//std::cout << "w: " << img.cols << " h: " << img.rows << " resize_ratio: " << resize_ratio <<std::endl;
	}
	else
	{
		padded_x = (0.5)*((double)416 - new_w);
		padded_y = 0;
		//std::cout << "w: " << img.cols << " h: " << img.rows << " resize_ratio: " << resize_ratio <<std::endl;
	}

	cv::Size new_sz(new_w, new_h);
	cv::Scalar grey_value(128, 128, 128);
	//cv::Mat padded_img(416, 416, CV_8UC3, grey_value); 	//Create an image with grey background
	std::cout << "new_w: " << new_w << " new_h: " << new_h << std::endl;

	try
	{
		//Letter box
		cv::resize(img, img, new_sz);
		//Put the resized image inside the grey image

		//cv::Mat p_i = padded_img(cv::Rect(padded_x, padded_y, resized_img.cols, resized_img.rows)).clone();
		//std::cout << "2.5" << std::endl;
		img.copyTo(padded_img(cv::Rect(padded_x, padded_y, img.cols, img.rows)));

		if (greyScale) {
			cv::cvtColor(padded_img, padded_img, cv::COLOR_BGR2GRAY);
			cv::cvtColor(padded_img, padded_img, cv::COLOR_GRAY2BGR);
		}
		else {
			cv::cvtColor(padded_img, padded_img, cv::COLOR_BGR2RGB);
		}

		padded_img.convertTo(padded_img, CV_32FC3, 1.0f / 255.0f);
		cv::resize(img, img, cv::Size(416, 416));
		img = padded_img.clone();


		return img;
	}
	catch (cv::Exception e)
	{
		std::cerr << e.msg << std::endl;

	}

	std::cout << "Finished" << std::endl;

	return img;
}


torch::Tensor yolo_tns(cv::Mat& img)
{
	return torch::from_blob(img.data, { 1, 416, 416, 3 }).permute({ 0, 3, 1, 2 });
}

cv::Mat& sppe_img(cv::Mat& img, const cv::Rect& rect)
{
	img = img(rect);
	if (rect.width * rect.height > 256 * 320)
	{
		cv::resize(img, img, cv::Size(256, 320));
		img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
	}
	else {
		img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
		cv::resize(img, img, cv::Size(256, 320));
	}
	return img;
}
torch::Tensor sppe_tns(std::vector<cv::Mat>& imgs)
{
	std::vector<torch::Tensor> tns;
	for (auto itr = imgs.begin(); itr != imgs.end(); itr++)
	{
		tns.push_back(torch::from_blob(itr->data, { 1, 320, 256, 3 }));
	}
	return torch::cat(tns).permute({ 0, 3, 1, 2 });
}
