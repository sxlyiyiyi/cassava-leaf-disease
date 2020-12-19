from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def show_image(img, lab, class_names):
    img = (img.numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # n = tf.where(tf.equal(lab, 1))
    n = lab
    img = cv2ImgAddText(img, class_names[int(n)], 20, 20,  (0, 0, 255))
    # img = cv2.putText(img, class_names[int(n)], (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


