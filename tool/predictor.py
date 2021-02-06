from tool.translate import build_model, translate
import numpy as np
import torch


class Predictor(object):
    def __init__(self, config):
        device = 'cpu'

        model, vocab = build_model(config)
        weights = './transformerocr1.pth'

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)), strict=False)

        config['device'] = device
        self.config = config
        self.model = model
        self.vocab = vocab

    @staticmethod
    def process_image(image):
        # convert to numpy array
        img = np.asarray(image)
        img = img.transpose(2, 0, 1)
        img = img / 255.0

        return img

    def predict(self, img):
        img = self.process_image(img)
        img = np.expand_dims(img, axis=0)
        img = torch.FloatTensor(img)
        img = img.to(self.config['device'])

        s = translate(img, self.model)[0].tolist()

        s = self.vocab.decode(s)

        return s

    def batch_predict(self, images):
        """
        param: images : list of ndarray
        """
        batch_images = [self.process_image(np.asarray(img)) for img in images]
        batch_images = torch.FloatTensor(batch_images)
        batch_images = batch_images.to(self.config['device'])

        sent = translate(batch_images, self.model).tolist()
        result = self.vocab.batch_decode(sent)

        return result

