#pragma once

#include <opencv2/opencv.hpp>

const std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255) };

// 输入图片大小， 默认640*640
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;

// 类别得分
const float SCORE_THRESHOLD = 0.5;

// NMS检测
const float NMS_THRESHOLD = 0.5;

// 置信度
const float CONFIDENCE_THRESHOLD = 0.5;

// 预测结果
struct Detection
{
	int class_id;
	float confidence;
	cv::Rect box;
};

/**
 * 调整图像大小，使其适应模型输入尺寸
 **/
cv::Mat format_yolov5(const cv::Mat& source);

/**
 * 目标检测函数
 **/
void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className);