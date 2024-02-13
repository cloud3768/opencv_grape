#include "infer.h"

/**
* RGBתHSI
**/
void calculate_hue(const cv::Mat& hsv, cv::Point center, int radius, HueGrade& hue_grade) {
	
	// ����Բ����Ĥͼ��
	cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8UC1);
	
	// ��ģͼ�ϻ���Բ��
	cv::circle(mask, center, radius, cv::Scalar(255), -1);

	// ����ɫ����ƽ��ֵ
	cv::Scalar mean_hue = cv::mean(hsv, mask);
	float average_hue = mean_hue[0] / 180;

	// ����ƽ��ɫ����������
	if (average_hue < 0.167 && average_hue > 0)	
	{
		hue_grade.grade = "G2";
	}
	else if (std::to_string(average_hue).substr(0, 5) == "0.167")
	{
		hue_grade.grade = "G2";
	}
	else if (average_hue < 0.333 && average_hue > 0.167)
	{
		hue_grade.grade = "G1";
	}
	else if (std::to_string(average_hue).substr(0, 5) == "0.333")
	{
		hue_grade.grade = "G1";
	}
	else if (average_hue < 0.833 && average_hue > 0.333)
	{
		hue_grade.grade = "G3";
	}
	else if (std::to_string(average_hue).substr(0, 5) == "0.833")
	{
		hue_grade.grade = "G3";
	}
	else if (average_hue < 1 && average_hue > 0.833 && average_hue == 0)
	{
		hue_grade.grade = "G4";
	}

	hue_grade.hue = average_hue;
}

/**
* �ع��������ֺ���
**/
void self_putText(cv::Mat img, FontStyle& font_style)
{
	cv::putText(img, font_style.text, font_style.point, font_style.face, font_style.scale, font_style.color, font_style.thickness);
}