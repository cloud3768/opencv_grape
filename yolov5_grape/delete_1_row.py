import os

input_folder_path = "datasets/label - 副本"  # 输入文件夹路径
output_folder_path = "datasets/label"  # 输出文件夹路径

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for file_name in os.listdir(input_folder_path):
    if file_name.endswith(".txt"):  # 只处理txt文件
        input_file_path = os.path.join(input_folder_path, file_name)
        output_file_path = os.path.join(output_folder_path, file_name)
        with open(input_file_path, "r") as f:
            lines = f.readlines()
        with open(output_file_path, "w") as f:
            for line in lines:
                if not line.startswith("1"):  # 不以1开头的行保留
                    f.write(line)
