import os
import json
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

#只适用于对原始数据的增强，无法对增强后数据进行增强。
# 定义增强器
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # 随机水平翻转，概率为0.5
    iaa.Flipud(0.5), # 随机垂直翻转，概率为0.5
    iaa.Rot90((0, 3)), # 随机旋转90度的倍数，概率为1，即每张图片都会进行旋转操作。参数 (0, 3) 表示旋转次数的范围为[0, 3]，即旋转角度的范围为[0, 270]度。
    iaa.Multiply((0.8, 1.2)), # 随机乘以一个系数，系数范围为[0.8, 1.2]，即每张图片的亮度会随机增加或减少 20%。
    iaa.GaussianBlur(sigma=(0, 1.0)), # 随机高斯模糊，标准差范围为[0, 1.0]
    iaa.AdditiveGaussianNoise(scale=(0, 0.3*255)) # 随机添加高斯噪声，噪声强度范围为[0, 0.3*255]
])

# 定义输入和输出路径
input_dir = "datasets/image"
output_dir = "datasets/image_enhance"

# 循环处理每个图像
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        # 读取图像和标注信息
        img_path = os.path.join(input_dir, filename)
        json_path = os.path.join(input_dir, filename[:-4] + ".json")
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        img = cv2.imread(img_path)

        # 将标注信息转换为imgaug格式
        bbs = []
        for shape in json_data["shapes"]:
            label = shape["label"]
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[2]
            bbs.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label))
        bbs = ia.BoundingBoxesOnImage(bbs, shape=img.shape)

        # 进行数据增强
        rotate_degree = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315]) # 随机选择旋转角度
        seq[2] = iaa.Rot90(rotate_degree // 90) # 更新旋转操作
        images_aug, bbs_aug = seq(images=[img], bounding_boxes=[bbs])

        # 保存增强后的图像和标注信息
        for i, image_aug in enumerate(images_aug):
            new_filename = filename[:-4] + "_aug" + str(i) + ".jpg"
            new_img_path = os.path.join(output_dir, new_filename)
            new_json_path = os.path.join(output_dir, new_filename[:-4] + ".json")
            cv2.imwrite(new_img_path, image_aug)
            if "imagePath" in json_data:
                json_data["imagePath"] = new_filename
            new_json_data = {
                "version": "2.3.0",
                "flags": {},
                "shapes": [],
                "imagePath": new_filename,
                "imageData": None,
                "imageHeight": image_aug.shape[0],
                "imageWidth": image_aug.shape[1],
                "text": ""
            }
            for bb in bbs_aug[i].bounding_boxes:
                new_json_data["shapes"].append({
                    "label": bb.label,
                    "points": [[float(bb.x1), float(bb.y1)], [float(bb.x2), float(bb.y2)]],
                    "group_id": None,
                    "description": "",
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {}
                })
            with open(new_json_path, 'w') as f:
                json.dump(new_json_data, f, indent=4)
