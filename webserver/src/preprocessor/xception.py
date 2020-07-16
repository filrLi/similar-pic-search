import numpy as np
from numpy import linalg as LA

from common.const import input_shape
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing import image


class XceptNet:
    def __init__(self):
        self.input_shape = (299, 299, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = Xception(weights=self.weight,
                              input_shape=(
                                  self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                              pooling=self.pooling,
                              include_top=False)
        self.model.predict(np.zeros((1, 299, 299, 3)))

    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(
            self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        norm_feat = [i.item() for i in norm_feat]
        return norm_feat


def extract_feat(img_path, model, graph, sess):
    with sess.as_default():
        with graph.as_default():
            img = image.load_img(img_path, target_size=(
                input_shape[0], input_shape[1]))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feat = model.predict(img)
            norm_feat = feat[0] / LA.norm(feat[0])
            norm_feat = [i.item() for i in norm_feat]
            return norm_feat
