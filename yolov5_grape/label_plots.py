import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib.colors import LinearSegmentedColormap


# 绘制柱状图
def plot_class(data_path, save_dir='', color=[]):
    # 计算数量
    data_class = data_path.iloc[:, 0]
    grape, berry = 0, 0

    for i in range(len(data_class)):
        if data_class[i] == 0:
            grape += 1
        elif data_class[i] == 1:
            berry += 1

    # 设置图片大小
    fig, ax = plt.subplots(figsize=(8, 8))

    # 隐藏右侧与上侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    class_x = np.array(['grape', 'berry'])
    class_y = np.array([grape, berry])

    # 绘制柱状图
    plt.bar(class_x[0], class_y[0], color=color[0], label='grape')
    plt.bar(class_x[1], class_y[1], color=color[1], label='berry')

    # 设置刻度线
    plt.tick_params(which='both', direction='in',
                    labelleft=True, labelbottom=True)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1000))

    # 绘制图例
    plt.legend(loc='upper left')

    # 优化布局且保存
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'classes.jpg'), dpi=300)


# 绘制方框图
def plot_rec(data_path, save_dir='', color=[]):
    # 获取类别cls与相对坐标值box
    cls = data_path.iloc[:, 0]
    box = data_path.iloc[:, 5:9]

    # 设置图片大小
    fix, ax = plt.subplots(figsize=(8, 8))

    # 类别标签
    label = {
        0: color[0],  # grape
        1: color[1]  # berry
    }

    # 隐藏右侧与上侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置刻度线
    plt.tick_params(which='both', direction='in',
                    labelleft=True, labelbottom=True)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(50))

    # 设置坐标轴范围
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)

    # 创建图例
    rec_patches = []

    for i in range(1000):
        # 确定方框颜色
        color = label[cls[i]]
        # 矩形框大小
        width = box.iloc[i, 2] - box.iloc[i, 0]
        height = box.iloc[i, 3] - box.iloc[i, 1]
        # 绘制方框
        rect = plt.Rectangle((1000 - width / 2, 1000 - height / 2), width, height,
                             edgecolor=color, facecolor='none', linewidth=0.5)
        ax.add_patch(rect)
        # 确定图例
        rec_patches.append(rect)

    # 添加图例
    legend_labels = ['grape', 'berry']
    ax.legend(rec_patches, legend_labels, loc='upper left')

    # 优化布局且保存
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'boxes.jpg'), dpi=300)


# 绘制密度图
def plot_histpost(data_boxes, save_dir='', row=[], save_name='', color=[]):
    # 获取box数据:x, y, width, height
    data_boxes = pd.DataFrame(data_boxes.iloc[:, 1:5], columns=row)
    # 设置图片大小
    fig, ax = plt.subplots(figsize=(8, 8))

    # 对坐标轴进行设置
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    # 隐藏右侧与上侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 刻度线向内
    plt.tick_params(which='both', direction='in')

    # 选择自动缩放坐标轴
    plt.autoscale(False)

    # 自定义cmap，[]内为自定义左右两端颜色
    colormap = LinearSegmentedColormap.from_list('Chinese_Color',
                                                 color, 256)

    # 绘制密度图
    sn.histplot(data_boxes, x=row[0], y=row[1], ax=ax, bins=50, pmax=0.9,
                cbar=True, hue_norm=True, cmap=colormap)

    # 优化布局且保存
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)


def main():
    plot_rec(data, 'yolov5/record/labels', color=['#E1BEE7', '#6adeb0'])
    plot_class(data, 'yolov5/record/labels', color=['#ff9558', '#c1c1c1'])
    plot_histpost(data, 'yolov5/record/labels', ['x', 'y'],
                  'boxes_xy.jpg', color=['#7F7FD5', '#91EAE4'])
    plot_histpost(data, 'yolov5/record/labels', ['width', 'height'],
                  'boxes_wh.jpg', color=['#46BF6E', '#46307D'])


if __name__ == "__main__":
    data = pd.read_csv('yolov5/record/data.csv')
    main()
