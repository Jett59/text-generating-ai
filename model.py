import tensorflow as tf
from tensorflow import keras
from keras import layers
from concept import ConceptLayer

class TextModel(keras.Model):
    def __init__(self, embedding_dimension, layer_count, head_count, vocabulary_size, dropout_rate):
        super().__init__()
        self.embedding = layers.Embedding(vocabulary_size, embedding_dimension)
        self.concept_layers = [ConceptLayer(embedding_dimension, head_count, dropout_rate) for _ in range(layer_count)]
        self.final_layer = layers.Dense(vocabulary_size, activation='softmax')

    def call(self, inputs):
        embeddings = self.embedding(inputs)
        for concept_layer in self.concept_layers:
            embeddings = concept_layer(embeddings)
        final_outputs = self.final_layer(embeddings)
        return final_outputs
