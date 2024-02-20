#include <fstream>
#include <algorithm>
#include "detect.h"
#include "infer.h"

int main()
{
	struct HueGrade HGrade;
	struct FontStyle NumGrain, GrainGrade, GrapeGrade, Hue, GrapeConfidence;
	 
	// 模型预测结果
	std::vector<Detection> output;

	// 等级分布为0：2；1：3和4
	std::vector<int> num_grade(2);

	// 果穗置信度
	std::string grape_confidednce;

	// 总体类别
	std::vector<std::string> class_list = { "grape","grain" };
	// 图片路径
	cv::Mat img = cv::imread("F:/C++/vs/OpenCV/config_files/image/TEST_2.JPG");

	// 模型路径
	cv::dnn::Net net = cv::dnn::readNet("F:/C++/vs/OpenCV/config_files/weights/best.onnx");

	if (img.empty())
	{
		std::cout << "End of stream\n";
	}
	
	// 转HSV图
	cv::Mat hsv;
	cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

	// 模型预测
	detect(img, net, output, class_list);

	int num_detect = output.size();

	for (int i = 0; i < num_detect; ++i)
	{
		auto& detection = output[i];
		auto box = detection.box;
		auto classID = detection.class_id;
		const auto& color = colors[classID % colors.size()];

		// 类别检测 0：葡萄串；1：果粒
		if (classID == 0)
		{
			// 绘制Grape及其置信度
			GrapeConfidence.text = "Grape: " + std::to_string(output[i].confidence).substr(0, 4);
			GrapeConfidence.point = cv::Point(box.x, box.y - 40);
			GrapeConfidence.face = cv::FONT_HERSHEY_SIMPLEX;
			GrapeConfidence.scale = 1.0;
			GrapeConfidence.color = color;
			GrapeConfidence.thickness = 3;
			self_putText(img, GrapeConfidence);

			GrapeGrade.point = cv::Point(box.x, box.y - 10);
			GrapeGrade.color = color;
			cv::rectangle(img, box, color, 3);
		}
		else
		{
			calculate_hue(hsv, cv::Point(box.x + box.width * 0.5, box.y + box.height * 0.5), 0.25 * (box.height + box.width), HGrade);

			if (HGrade.grade == "G2")
				num_grade[0]++;
			else if (HGrade.grade == "G3" || HGrade.grade == "G4")
				num_grade[1]++;

			//  绘制H分量值
			Hue.text = std::to_string(HGrade.hue).substr(0, 5);
			Hue.point = cv::Point(box.x + box.width * 0.5 - 47, box.y + box.height * 0.5 + 11);
			Hue.face = cv::FONT_HERSHEY_SIMPLEX;
			Hue.scale = 1.0;
			Hue.color = cv::Scalar(0, 0, 0);
			Hue.thickness = 3;
			self_putText(img, Hue);

			// 绘制根据H分量值判断的果粒的等级
			GrainGrade.text = HGrade.grade;
			GrainGrade.point = cv::Point(box.x + box.width * 0.5 - 22, box.y + box.height * 0.5 + 40);
			GrainGrade.face = cv::FONT_HERSHEY_SIMPLEX;
			GrainGrade.scale = 1.0;
			GrainGrade.color = cv::Scalar(0,0,0);
			GrainGrade.thickness = 3;
			self_putText(img, GrainGrade);

			cv::circle(img, cv::Point(box.x + box.width * 0.5, box.y + box.height * 0.5), 0.25 * (box.height + box.width), color, 3);
		}
	}

	// 判断整体果穗等级并绘制
	// GrapeGrade.point = cv::Point(cv::Point(50, 350));
	GrapeGrade.face = cv::FONT_HERSHEY_SIMPLEX;
	GrapeGrade.scale = 1;
	// GrapeGrade.color = cv::Scalar(0, 0, 0);
	GrapeGrade.thickness = 3;

	float pre_grade = (num_grade[0] + num_grade[1]) / (num_detect - static_cast<float>(1));

	if (pre_grade < 0.2)
		GrapeGrade.text = "Maturity: D";
	else if (pre_grade >= 0.2 && pre_grade < 0.6)
		GrapeGrade.text = "Maturity: C";
	else if (pre_grade >= 0.6 && pre_grade < 0.8)
		GrapeGrade.text = "Maturity: B";
	else if ((num_grade[1] / (num_detect - static_cast<float>(1))) >= 0.8)
		GrapeGrade.text = "Maturity: A";
	self_putText(img, GrapeGrade);

	// 绘制左上角的果粒数量统计
	NumGrain.text = std::to_string(num_detect - 1);
	NumGrain.point = cv::Point(50, 150);
	NumGrain.face = cv::FONT_HERSHEY_SIMPLEX;
	NumGrain.scale = 5.0;
	NumGrain.color = cv::Scalar(0, 0, 0);
	NumGrain.thickness = 5;
	self_putText(img, NumGrain);

	// 图片输出
	cv::imwrite("output.jpg", img);

	return 0;
}