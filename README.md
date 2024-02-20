# opencv_grape
yolov5s的C++推理部署程序

## 工作状态

### 已实现

计数

圆形Hough变换

熟度检测且评级

### 预测目标

优化程序

数学建模实现最终数量预测

## 项目结构

│  OpenCV.sln	  ## VS项目

│  README.md	## 文档

├─config_files	## 配置文件夹

│  │  weights	 ## 权重文件夹

│  │  classes.txt	## 分类信息

│  └─image	## 图片

├─OpenCV	## 项目文件夹

│  │  detect.cpp	## OpenCV实现预测

│  │  detect.h	    ## OpenCV实现预测头文件

│  │  infer.cpp	  ## 后续推理程序

│  │  infer.h	      ## 后续推理程序头文件

│  │  main.cpp	 ## 主程序

│  │  OpenCV.vcxproj

│  │  OpenCV.vcxproj.filters

│  │  OpenCV.vcxproj.user

