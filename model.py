import tensorflow as tf
from tensorflow import keras
from keras import layers
from query import QueryLayer

class TextModel(keras.Model):
    def __init__(self, embedding_dimension, layer_count, vocabulary_size, dropout_rate):
        super().__init__()
        self.embedding = layers.Embedding(vocabulary_size, embedding_dimension)
        self.query_layers = [QueryLayer(embedding_dimension, dropout_rate) for _ in range(layer_count)]
        self.final_layer = layers.Dense(vocabulary_size, activation='softmax')

    def call(self, inputs):
        embeddings = self.embedding(inputs)
        for query_layer in self.query_layers:
            embeddings = query_layer(embeddings)
        final_outputs = self.final_layer(embeddings)
        return final_outputs
