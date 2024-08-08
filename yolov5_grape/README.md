# yolov5_grape
大赛的Yolov5s模型仓库

## 注意

yolov5文件中不上传run文件内容（即训练结果保存在本地）

对应网络结构存储在record指定文件夹内

**在workers 12 下，GhostNet达到了 1.0ms部署，42.3ms推理，4.3ms NMS处理！望周知**

## autodl子账户

[https://www.autodl.com/subAccountLogin]

账户:grape_guest@323262f14928
密码:323262f14928

## 网络结构

### 初始网络结构图

![YOLOv5s网络结构](./yolov5s网络结构.png)

修改C3模块为C2f

使用GIOU作为损失函数

#### Backbone

9层添加CA注意力机制（去除）

RepVGG替换C2f模块（很差）

使用GhostNet作为主体网络（采用）

#### head

16层、20层、24层添加SimAM注意力机制（去除）

使用GhostNet作为Conv、C3模块（采用）

### 预修改方向

修改Conv为更轻量级的卷积层（取消）

RepVGG推理代码（取消）

调整RepVGG替换位置（取消）

#### Backbone

使用MobileNetv3作为主体网络

使用ShuttleNetv2作为主体网络

使用GhostNet作为主体网络（采用）

上述主体网络基础上添加注意力机制模组等

#### head

使用BoderDet作为预测头

## 项目结构

│  .gitignore	## git提交忽略的项目

│  berry_number_prediction.ipynb	## Jupyter Notebook yolov5模型具体操作

│  contour_detection.py	## 边缘检测（已抛弃）	

│  dataset.py	## 数据集分割

│  data_enhance.py	## 数据增强脚本

│  delete_1_row.py	## 删除以1开头的标注坐标txt

│  json2yolo.py	## json转yolo的txt

│  process_negative.py	## 处理图片标注中部分错误标注（负数、重复）

│  README.md	## 项目文档

│  repair_jpg.py	## 修复图片损坏警告

│  segment_grape_grains.py	## 后续推理代码（已抛弃）

│ label_plots.py	## label分析图像绘制

│ detect_grape.py 	## yolov5自带detect推理

├─datasets	## 数据集

├─yolov5	## yolov5模型

├─detect_grape	## 嚛lov自带detect推理结果保存

└─yolov5prune	## yolov5剪枝模型

