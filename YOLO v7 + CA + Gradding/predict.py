from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    # 初始化模型
    yolo = YOLO()
    # 循环读取图片进行预测
    while True:
        # 输入图像路径
        img = input('Input image filename:')
        # 打开图像
        image = Image.open(img)
        # 对图像进行预测
        r_image = yolo.detect_image(image, crop=False, count=False)
        # 展示图像
        r_image.show()
