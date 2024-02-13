#pragma once

#include <opencv2/opencv.hpp>

// H�����Լ�����H����������
struct HueGrade
{
	float hue{};
	std::string grade;
};

// ������
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
* RGBתHSI
**/
void calculate_hue(const cv::Mat& hsv, cv::Point center, int radius, HueGrade& hue_grade);


/**
* �ع��������ֺ���
**/
void self_putText(cv::Mat img, FontStyle& font_style);
