import yaml
from tool.predictor import Predictor
import numpy as np
import time
from pathlib import Path
from PIL import Image
import cv2
import os


class TextRecognition(object):
    def __init__(self):
        self.config = self.read_from_config('./capcha_base.yml')
        self.config['device'] = 'cpu'
        self.predictor = Predictor(self.config)

    @staticmethod
    def read_from_config(file_yml):
        with open(file_yml, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def remove_noise(self, image, pass_factor):
        for column in range(image.size[0]):
            for line in range(image.size[1]):
                value = self.remove_noise_by_pixel(image, column, line, pass_factor)
                image.putpixel((column, line), value)
        return image

    @staticmethod
    def remove_noise_by_pixel(image, column, line, pass_factor):
        if image.getpixel((column, line)) < pass_factor:
            return 0
        return image.getpixel((column, line))

    def preprocess(self, image):
        if 'L' != image.mode:
            img_bg = image.convert('L')
        else:
            img_bg = image

        # get background img
        img_bg = self.remove_noise(img_bg, pass_factor=170)
        img_bg = np.asarray(img_bg)

        img_final = np.asarray(image.convert('L')) - np.asarray(img_bg)
        img_final = cv2.cvtColor(img_final, cv2.COLOR_GRAY2BGR)

        return img_final

    def predict(self, image):
        # remove noise from image
        image = self.preprocess(image)

        r = self.predictor.predict(image)
        return r

    def predict_on_batch(self, images):
        batch_images = [self.preprocess(img) for img in images]
        return self.predictor.batch_predict(batch_images)


model = TextRecognition()
start = time.time()
img = Image.open('./test.png')
print(model.predict(img))
end = time.time()
print('Estimated time: {}s'.format(end - start))
