import tensorflow as tf
from tensorflow import keras
from keras import layers

class TextModel(keras.Model):
    def __init__(self):
        super().__init__(self)
        self.final_layer = layers.Dense()

    def call(self, inputs):
        final_outputs = self.final_layer(inputs)
        return final_outputs
