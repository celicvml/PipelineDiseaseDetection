import os
import random
import xml.etree.ElementTree as ET
import numpy as np

from utils.utils import get_classes

# 类别文件
classes_path = 'model_data/voc_classes.txt'

# 比例参数
# 训练集 + 验证集 : 测试集 = 9 : 1
trainval_percent = 0.9
# 训练集 : 验证集 = 9 : 1
train_percent = 0.9

# 数据集路径
# 数据集所在文件夹
VOCdevkit_path = r'D:\DataSet\VOCdevkit'
# 表示数据集的划分方式 一个是训练集 一个是测试集
VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]

# 获取所有类别列表
classes, _ = get_classes(classes_path)

# 创建空的数组 存放每个病害目标的数量
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
nums = np.zeros(len(classes))


# 解析 VOC数据集 中的标注文件
# year 年份
# image_id 图像的名称(去掉后缀)
# list_files 已经打开的 txt 文件
def convert_annotation(year_annontation, image_name, list_files):
    # 获取标签文件的路径
    in_file = os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year_annontation, image_name))
    # 解析标签文件
    tree=ET.parse(in_file)
    # 获取标签文件的根节点
    root = tree.getroot()
    # 遍历标注框
    for obj in root.iter('object'):
        # 获取类别
        cls = obj.find('name').text
        # 如果类别不存在 跳过该框
        if cls not in classes:
            continue
        # 获取类别对应的 id
        cls_id = classes.index(cls)
        # 获取位置信息
        xmlbox = obj.find('bndbox')
        # 获取 xmin ymin xmax ymax
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        # 在列表各个元素之间添加 ,  最后添加上类别
        # example: 410,245,512,302,4 18,419,144,571,7(以空格开头)
        list_files.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        # 然后统计每一种病害的数量
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


if __name__ == "__main__":

    # 生成随机数种子
    # 将种子设置为 0 可以使得每次程序运行时生成的随机数序列相同
    random.seed(0)

    # os.path.abspath() 获取数据集存放路径的绝对路径
    # 判断数据集存放的路径是否有空格 如果存在空格 则抛出一个异常
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集绝对路径存在空格,请保证路径没有空格")

    # 生成 ImageSets/Main 中的 txt文件
    print("Generate txt in ImageSets.")
    # 获取 标签文件 存放的 文件夹路径
    xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
    # 获取生成的 txt文件 存放的 文件夹路径
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
    # 获取标签文件夹下面的 所有文件(可能包括除了xml以外的文件)
    temp_xml = os.listdir(xmlfilepath)
    # 标签文件夹所有的 xml文件
    total_xml = []
    # 获取所有的 xml文件
    for xml in temp_xml:
        # 判断文件 是否是以 .xml字符 结尾
        if xml.endswith(".xml"):
            # 添加到 total_xml 列表中
            total_xml.append(xml)
    # 标签文件的总数量
    num = len(total_xml)
    # 一个可迭代对象 数量为 标签文件的数量
    list = range(num)
    # 训练集 测试集 按照占比 计算的数量
    tv = int(num * trainval_percent)
    # 训练集 按照占比 计算的数量
    tr = int(tv * train_percent)
    # random.sample(x, y) 从 x 序列中 随机选取 y 个 元素 返回一个列表
    # 随机选出 训练集 和测试集
    trainval = random.sample(list, tv)
    # 随机选出 训练集
    train = random.sample(trainval, tr)
    # 打印 训练集 + 测试集 训练集 数量
    print("train and val size", tv)
    print("train size", tr)

    # 训练集 + 验证集 文件
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    # 测试集 文件
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    # 训练集 文件
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    # 验证集文件
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    # 遍历所有标签文件
    for i in list:
        # 获取标签文件的名称(去掉 .xml) 然后加上换行符
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            # 训练集 + 验证集
            ftrainval.write(name)
            if i in train:
                # 训练集
                ftrain.write(name)
            else:
                # 验证集
                fval.write(name)
        else:
            # 测试集
            ftest.write(name)
    # 关闭文件
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Generate 2007_train.txt and 2007_val.txt for train.")
    # VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
    # year, image_set = '2007', 'train'
    # year, image_set = '2007', 'val'
    for year, image_set in VOCdevkit_sets:
        # 读取 Main 里面生成的 已经分类好的 训练集 验证集
        # read() 读取文件里面的所有的内容
        # strip() 删除字符串前后的 空格 和 换行符
        # split() 将字符串按照换行符进行分隔 生成一个列表
        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)), encoding='utf-8').read().strip().split()
        # 新建生成的 txt文件
        list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
        # 依次遍历所有的图像 id
        for image_id in image_ids:
            # 将图像绝对路径名写入 txt文件中
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))
            # 将标注信息也写入 txt文件中
            convert_annotation(year, image_id, list_file)
            # 写完一张图像的内容后 后面加上 换行
            list_file.write('\n')
        # 关闭文件
        list_file.close()
    print("Generate 2007_train.txt and 2007_val.txt for train done.")

    # 获取每一种病害的数量 转换成字符串
    str_nums = [str(int(x)) for x in nums]
    # 打印每种病害的数量
    for i in range(len(classes)):
        print('| ' + classes[i] + ' | ' + str_nums[i] + ' |')
