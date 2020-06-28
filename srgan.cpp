
/*
#include "net.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include<vector>
using namespace std;
void convert(ncnn::Mat& img, float a = 1., float b = 0) {
	for (int c = 0; c < 3; ++c) {
		for (int r = 0; r < img.h; ++r) {
			for (int l = 0; l < img.w; ++l) {
				img.channel(c).row<float>(r)[l] = img.channel(c).row<float>(r)[l] * a + b;
			}
		}
	}
}
void visualize(cv::Mat img_input, ncnn::Mat& m)
{
	cv::Mat channels[3];
	std::vector<cv::Mat> normed_feats(m.c);
	cv::Mat img_output(m.h, m.w, CV_8UC3);
	convert(m, 255. / 2, 255. / 2);
	m.to_pixels(img_output.data, ncnn::Mat::PIXEL_RGB2BGR);
	cv::imshow("input", img_input);
	cv::imshow("output", img_output);
	cv::waitKey(0);
}
int main(int argc, char* argv[]) {
	cv::Mat img_src = cv::imread("C:/Users/daddy/Desktop/input.jpg");//CV 读入的图片维度HWC BGR 需要先转成RGB
	if (img_src.empty()) {
		std::cout << "input image is empty." << std::endl;
		return -1;
	}
	ncnn::Net net;
	if (net.load_param("C:/Users/许加源/Desktop/srgan.param") == -1 ||
		net.load_model("C:/Users/许加源/Desktop/srgan.bin") == -1) {
		std::cout << "load ga model failed." << std::endl;
		return -1;
	}
	ncnn::Extractor ex = net.create_extractor();
	ncnn::Mat img_ncnn = ncnn::Mat::from_pixels(img_src.data, ncnn::Mat::PIXEL_BGR2RGB, img_src.cols, img_src.rows);//ncnn转的时候 转成RGB和CHW
	convert(img_ncnn, 2. / 255, -1);//,-1);

	ex.input("input.1", img_ncnn);
	ncnn::Mat img_out;
	ex.extract("113", img_out);
	visualize(img_src, img_out);
	return 0;
}

 */



#include "net.h"

#include "opencv2/core.hpp"

#include "opencv2/imgproc.hpp"

#include "opencv2/highgui.hpp"

#include <iostream>
#include<vector>
using namespace std;
void convert(ncnn::Mat& img, float a = 1., float b = 0) {
	for (int c = 0; c < 3; ++c) {
		for (int r = 0; r < img.h; ++r) {
			for (int l = 0; l < img.w; ++l) {
				img.channel(c).row<float>(r)[l] = img.channel(c).row<float>(r)[l] * a + b;
			}
		}
	}
}
void ncnn_debug(ncnn::Mat& ncnn_img, string img_name)
{
	convert(ncnn_img, 255./2, 255./2);
	cv::Mat imageDate(ncnn_img.h, ncnn_img.w, CV_8UC3);
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < ncnn_img.h; i++)
		{
			for (int j = 0; j < ncnn_img.w; j++)
			{
				float t = ((float*)ncnn_img.data)[j + i * ncnn_img.w + c * ncnn_img.h * ncnn_img.w];
				imageDate.data[(2 - c) + j * 3 + i * ncnn_img.w * 3] = t;
			}
		}
	}
	//cv::normalize(imageDate, imageDate, 0, 255, cv::NORM_MINMAX, CV_8UC3);

	cv::imwrite("C:/Users/daddy/Desktop/output.jpg", imageDate);
}

void visualize(const char* title, const ncnn::Mat& m)

{
	cv::Mat channels[3];
	std::vector<cv::Mat> normed_feats(m.c);
	for (int i = 0; i < m.c; i++)
	{
		cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));

		cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8UC3);
		//  cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);
		// check NaN
		for (int y = 0; y < m.h; y++)
		{
			const float* tp = tmp.ptr<float>(y);
			uchar* sp = normed_feats[i].ptr<uchar>(y);
			for (int x = 0; x < m.w; x++)
			{
				float v = tp[x];
				if (v != v)
				{
					sp[0] = 0;
					sp[1] = 0;
					sp[2] = 255;
				}
				sp += 3;
			}
		}
	}

	channels[0] = normed_feats[2];//三个通道的图 R G B
	channels[1] = normed_feats[1];
	channels[2] = normed_feats[0];
	cv::Mat c;

	//cv::imwrite("C:/users/asus/desktop/tesstR.jpg", R);
	//cv::imwrite("C:/users/asus/desktop/tesstG.jpg", G);
	//cv::imwrite("C:/users/asus/desktop/tesstB.jpg", B);
	cv::merge(channels, 3, c);
	//cv::normalize(c, c, 0, 255, cv::NORM_MINMAX, CV_8UC3);
	/*for (int i = 0; i < channelss; ++i)

	{
		for (int j = 0; j < height; ++j)
		{
			for (int k = 0; k < width; ++k)
			{
				float z = c.data[i * height * width + j * width + k];
				z *= 255.0;
				c.data[i * height * width + j * width + k] = z;
			}
		}
	}*/

	cv::imwrite("C:/Users/daddy/Desktop/output.jpg", c);

}




int main(int argc, char* argv[]) {

	cv::Mat img_src = cv::imread("C:/Users/daddy/Desktop/input.jpg");//CV 读入的图片维度HWC BGR 需要先转成RGB 
	int annels = img_src.channels();
	int dim = img_src.dims;


	if (img_src.empty()) {

		std::cout << "input image is empty." << std::endl;

		return -1;

	}

	ncnn::Net net;

	if (net.load_param("C:/Users/daddy/Desktop/srgan.param") == -1 ||

		net.load_model("C:/Users/daddy/Desktop/srgan.bin") == -1) {

		std::cout << "load ga model failed." << std::endl;

		return -1;

	}

	int height = img_src.rows;
	int width = img_src.cols;
	int channels = img_src.channels();
	//float z = img_src.data[1 * width * channels + 1 * channels + 1];//访问像素值
	/*for (int i = 0; i < channels; ++i)

	{
		for (int j = 0; j < height; ++j)
		{
			for (int k = 0; k < width; ++k)
			{
				double z = img_src.data[i * height * width + j * width + k];
				z /= 255.0;
				img_src.data[i * height * width + j * width + k] = z;
			}
		}
	}*/





	ncnn::Extractor ex = net.create_extractor();
	ncnn::Mat img_ncnn = ncnn::Mat::from_pixels_resize(img_src.data,

		ncnn::Mat::PIXEL_BGR2RGB, img_src.cols, img_src.rows, img_src.cols, img_src.rows);//ncnn转的时候 转成RGB和CHW 


	ex.input("input.1", img_ncnn);
	convert(img_ncnn, 2. / 255, -1);
	ncnn::Mat img_out;
	ex.extract("113", img_out);
	ncnn_debug(img_out,"test.jpg");
	//visualize("1", img_out);
	system("pause");
	return 0;








}


