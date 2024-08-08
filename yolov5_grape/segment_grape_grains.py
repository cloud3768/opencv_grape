import os
from PIL import Image, ImageDraw, ImageFont, ImageOps

image_path = 'datasets/grape/1970_01_01_18_41_IMG_0689.JPG'
label_path = 'yolov5/runs/detect/exp9/labels/1970_01_01_18_41_IMG_0689.txt'
save_path = 'yolov5/runs/detect/exp9/OTSU_img/1970_01_01_18_41_IMG_0689.jpg'

def text_lines_with_1(label_path):
    try:
        grain_coordinates = []
        grape_coordinates = []
        grain_coordinates_list = []
        grape_coordinates_list = []
        with open(label_path, 'r') as label:
            for line in label:
                if line.startswith('1'):
                    grain_coordinates.append(line.lstrip("['1").strip("']"))
                elif line.startswith('0'):
                    grape_coordinates.append(line.lstrip("['0").strip("']"))
            for line in grain_coordinates:
                grain_coordinates_list.append(line.split())
            for line in grape_coordinates:
                grape_coordinates_list.append(line.split())
        return grain_coordinates_list,grape_coordinates_list
    except IOError:  # 处理文件读取错误
        print(f"无法读取文件：{label_path}")


def yolo_to_actual_coordinates(img):
    width, height = img.size

    # 获取标签中以'1'开头的行对应的坐标信息
    grain_coordinates_list, grape_coordinates_list  = text_lines_with_1(label_path)
    for i in range(len(grain_coordinates_list)):
        for j in range(len(grain_coordinates_list[i])):
            grain_coordinates_list[i][j] = float(grain_coordinates_list[i][j])
    
    for i in range(len(grape_coordinates_list)):
        for j in range(len(grape_coordinates_list[i])):
            grape_coordinates_list[i][j] = float(grape_coordinates_list[i][j])

    yolo_grain_data = []
    yolo_grade_data = []
    
    for line in grape_coordinates_list:
        [x, y, w, h] = line
        center_x = x * width
        center_y = y * height
        box_width = w * width
        box_height = h * height
        yolo_grade_data.append([center_x, center_y, box_width, box_height])
    
    # 将YOLO格式的坐标转换为实际坐标
    for line in grain_coordinates_list:
        [x, y, w, h] = line
        center_x = x * width
        center_y = y * height
        box_width = w * width
        box_height = h * height
        yolo_grain_data.append([center_x, center_y, box_width, box_height])
    return yolo_grain_data, yolo_grade_data


def segment_grain_circle(img, yolo_grain_data):
    draw = ImageDraw.Draw(img)
    center = (yolo_grain_data[0], yolo_grain_data[1])
    radius = (yolo_grain_data[2] + yolo_grain_data[3]) / 4
    draw.ellipse((center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius), outline='yellow', width=10)

  
def segment_grape_rectangle(img, yolo_grape_data):
    draw = ImageDraw.Draw(img)
    [left, top, right, bottom] = [yolo_grape_data[0] - yolo_grape_data[2] / 2,
                                  yolo_grape_data[1] - yolo_grape_data[3] / 2,
                                  yolo_grape_data[0] + yolo_grape_data[2] / 2,
                                  yolo_grape_data[1] + yolo_grape_data[3] / 2]
    draw.rectangle([left, top, right, bottom], outline='red', width=10)
    
    
try:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
except OSError:  # 处理文件不存在错误
    print(f"文件 {image_path} 不存在！")


yolo_grain_data, yolo_grape_data = yolo_to_actual_coordinates(img)
for j in yolo_grain_data:
    segment_grain_circle(img, j)
    
for i in yolo_grape_data:
    segment_grape_rectangle(img, i)

draw = ImageDraw.Draw(img)
text = str(len(yolo_grain_data))
ttf = ImageFont.truetype('arial.ttf', 150)
text_color = (255, 255, 255)
text_pos = (125, 125)
draw.text(text_pos, text, font=ttf, fill=text_color)
directory = os.path.dirname(save_path)
if not os.path.exists(directory):
    os.makedirs(directory)
else:
    img.save(save_path)
