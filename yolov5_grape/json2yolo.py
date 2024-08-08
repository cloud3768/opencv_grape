import os
import json

# 定义类别名称及其对应的编号
class_names = {"grain": 1, "grape": 0}

# 定义函数，将矩形框的坐标转换为YOLOv5训练所使用的格式
def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    if len(box) == 2: # 适用于第一个json格式
        x = (box[0][0] + box[1][0]) / 2.0
        y = (box[0][1] + box[1][1]) / 2.0
        w = box[1][0] - box[0][0]
        h = box[1][1] - box[0][1]
    else: # 适用于第二个json格式
        x = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4.0
        y = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4.0
        w = max(box[0][0], box[1][0], box[2][0], box[3][0]) - min(box[0][0], box[1][0], box[2][0], box[3][0])
        h = max(box[0][1], box[1][1], box[2][1], box[3][1]) - min(box[0][1], box[1][1], box[2][1], box[3][1])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# 定义函数，将json文件中的矩形框转换为YOLOv5训练所使用的格式
def convert_annotation(image_id, json_dir, label_dir):
    # 读取json文件
    with open(os.path.join(json_dir, image_id + '.json'), 'r') as f:
        data = json.load(f)
    # 获取图片尺寸
    width = data['imageWidth']
    height = data['imageHeight']
    # 遍历所有矩形框
    boxes = []
    for shape in data['shapes']:
        # 获取矩形框的类别名称、坐标
        class_name = shape['label']
        box = shape['points']
        # 将矩形框的坐标转换为YOLOv5训练所使用的格式
        x, y, w, h = convert_coordinates((width, height), box)
        # 将矩形框的类别名称和转换后的坐标保存到boxes列表中
        boxes.append((class_names[class_name], x, y, w, h))
    # 将所有矩形框的类别名称和转换后的坐标保存到txt文件中
    with open(os.path.join(label_dir, image_id + '.txt'), 'w') as f:
        for box in boxes:
            f.write(str(box[0]) + " " + " ".join([str(a) for a in box[1:]]) + '\n')

# 输入文件夹和输出文件夹的路径填充位置
json_dir = 'datasets/label_grape_json'
label_dir = 'datasets/label_grape'

# 遍历所有json文件所在的文件夹
for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        # 获取json文件的名称和路径
        image_id = os.path.splitext(filename)[0]
        # 将json文件中的矩形框转换为YOLOv5训练所使用的格式
        convert_annotation(image_id, json_dir, label_dir)
