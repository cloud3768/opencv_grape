#pragma once

#include <opencv2/opencv.hpp>

// H分量以及根据H分量的评级
struct HueGrade
{
	float hue{};
	std::string grade;
};

// 字体风格
struct FontStyle
{
	std::string text;
	cv::Point point;
	cv::Scalar color;
	float scale{};
	int thickness{};
	int face{};
};

/**
* RGB转HSI
**/
void calculate_hue(const cv::Mat& hsv, cv::Point center, int radius, HueGrade& hue_grade);


/**
* 重构绘制文字函数
**/
void self_putText(cv::Mat img, FontStyle& font_style);
