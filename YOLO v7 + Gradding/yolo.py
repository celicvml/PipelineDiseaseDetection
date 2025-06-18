import colorsys
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox


class YOLO(object):
    _defaults = {

        "model_path": 'logs/last_epoch_weights.pth',
        "classes_path": 'model_data/voc_classes.txt',

        "anchors_path": 'model_data/yolo_anchors.txt',
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],

        "input_shape": [640, 640],
        "phi": 'l',

        # 置信度
        "confidence": 0.5,
        # 非极大值抑制参数
        "nms_iou": 0.3,

        # 是否对图像进行不失真的resize
        "letterbox_image": False,
        # 是否使用CUDA
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # 初始化YOLO
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        # 获取种类 数量
        self.class_names, self.num_classes = get_classes(self.classes_path)
        # 获取先验框 数量
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        # 初始化解码
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        # 画框设置
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        # 显示参数
        show_config(**self._defaults)

    # 生成模型
    def generate(self, onnx=False):
        # 初始化YOLO
        self.net = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        # 设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载模型
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # 检测图像
    def detect_image(self, image, crop = False, count = False):
        # 获取图像的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        # 将图像转换为RGB
        image = cvtColor(image)
        # 对图像进行resize
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 调整图像的维度 h, w, 3 -> 3, h, w -> 1, 3, h, w
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # 当前计算不需要反向传播
        with torch.no_grad():
            # 将图像转换为torch格式
            images = torch.from_numpy(image_data)
            # 将图像转移到CUDA
            if self.cuda:
                images = images.cuda()
            # 将图像输入到网络中
            outputs = self.net(images)
            # 对网络的输出进行解码
            outputs = self.bbox_util.decode_box(outputs)
            # 对解码的结果进行非极大值抑制
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)

            # 没有预测出任何结果
            if results[0] is None:
                return image
            # 获取标签
            top_label = np.array(results[0][:, 6], dtype='int32')
            # 获取置信度 是否包含物体 种类置信度
            top_conf = results[0][:, 4] * results[0][:, 5]
            # 预测框
            top_boxes = results[0][:, :4]

        # 设置字体和边框厚度
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # 计数
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        # 是否对目标进行裁剪
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # 图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        # 打开txt文件
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8')
        # 获取高宽
        image_shape = np.array(np.shape(image)[0:2])
        # 将图像转换为RGB图像
        image = cvtColor(image)
        # 对图像进行resize
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 添加维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            # 转换为torch格式
            images = torch.from_numpy(image_data)
            # 转移到cuda
            if self.cuda:
                images = images.cuda()
            # 将图像输入到网络中
            outputs = self.net(images)
            # 对网络输出结果进行解码
            outputs = self.bbox_util.decode_box(outputs)
            # 进行非极大值抑制
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
            # 没有预测结果
            if results[0] is None: 
                return 

            # 获取标签
            top_label = np.array(results[0][:, 6], dtype='int32')
            # 获取置信度 两部分组成
            top_conf = results[0][:, 4] * results[0][:, 5]
            # 获取预测框
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
