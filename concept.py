import tensorflow as tf
from tensorflow import keras
from keras import layers

class ConceptLayer(layers.Layer):
    def __init__(self, embedding_dimension, dropout_rate):
        super().__init__()
        self.dropout = layers.Dropout(dropout_rate)
        self.normalize = layers.LayerNormalization()
        self.concept_map = self.add_weight(name='concept_map', shape=(embedding_dimension, embedding_dimension), trainable=True)


    def call(self, input):
        positional_masks = []
        for i in range(input.shape[1]):
            positional_factors = []
            # Each token is multiplied by 1/the distance to the current token.
            for j in range(i):
                positional_factors.append(1 / (i - j))
            for j in range(i, input.shape[1]):
                positional_factors.append(0)
            positional_factors = tf.convert_to_tensor(positional_factors, dtype=input.dtype)
            positional_masks.append(positional_factors)
        positional_masks = tf.stack(positional_masks, axis=0)
        result = tf.einsum('bki,blj,ij,kl->bki', input, input, self.concept_map, positional_masks)
        # Add and normalize, then we're done.
        result += input
        result = self.normalize(result)
        return self.dropout(result)
