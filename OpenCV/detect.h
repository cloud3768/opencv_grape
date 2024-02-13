#pragma once

#include <opencv2/opencv.hpp>

const std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255) };

// ����ͼƬ��С�� Ĭ��640*640
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;

// ���÷�
const float SCORE_THRESHOLD = 0.5;

// NMS���
const float NMS_THRESHOLD = 0.5;

// ���Ŷ�
const float CONFIDENCE_THRESHOLD = 0.5;

// Ԥ����
struct Detection
{
	int class_id;
	float confidence;
	cv::Rect box;
};

/**
 * ����ͼ���С��ʹ����Ӧģ������ߴ�
 **/
cv::Mat format_yolov5(const cv::Mat& source);

/**
 * Ŀ���⺯��
 **/
void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className);