import tensorflow as tf
from tensorflow import keras
from keras import layers

class ConceptLayer(layers.Layer):
    def __init__(self, embedding_dimension, dropout_rate):
        super().__init__()
        self.dropout = layers.Dropout(dropout_rate)
        self.normalize = layers.LayerNormalization()
        self.concept_map = self.add_weight('concept_map', (embedding_dimension, embedding_dimension, embedding_dimension), trainable=True)

    def get_relative_meaning(self, primary_token, secondary_token, distance):
        # We need the primary and secondary tokens to have three dimensions so that we can use matmul properly.
        primary_token = tf.expand_dims(primary_token, axis=-2)
        secondary_token = tf.expand_dims(secondary_token, axis=-2)
        secondary_token /= tf.cast(distance ** 2, secondary_token.dtype)
        # The idea is to map each combination of input concepts to a corresponding set of output concepts.
        # We do this by getting the product of every pair of input concepts, and then (element wise) multiplying it by the concept map and summing each matrix.
        conceptual_matrix = tf.matmul(primary_token, secondary_token, transpose_a=True)
        # The concept map is in the shape (batch_size, embedding_dimension, embedding_dimension).
        # We want it to be in the shape (batch_size, 1, embedding_dimension, embedding_dimension) so we can multiply it with the concept map, which has shape (embedding_dimension, embedding_dimension, embedding_dimension).
        conceptual_matrix = tf.expand_dims(conceptual_matrix, axis=1)
        # We then multiply the conceptual matrix by the concept map and sum along the last two axes to get the relative meaning.
        result = tf.reduce_sum(conceptual_matrix * self.concept_map, axis=(2, 3))
        return result

    def transform_meaning(self, current_token, preceding_tokens):
        preceding_token_count = preceding_tokens.shape[1]
        if preceding_token_count > 0:
            relative_meanings = tf.vectorized_map(lambda i: self.get_relative_meaning(current_token, preceding_tokens[:, i], preceding_token_count - i), tf.range(preceding_token_count))
            # relative_meanings has the shape (preceding_token_count, batch_size, embedding_dimension).
            # This is ok, since we are going to sum anyway so we can just sum along the first axis instead of the second one.
        else:
            relative_meanings = tf.zeros((0, current_token.shape[0], current_token.shape[-1]), dtype=current_token.dtype)
        return tf.reduce_sum(relative_meanings, axis=0)

    def call(self, input):
        result = []
        for i in range(input.shape[-2]):
            result.append(self.transform_meaning(input[:, i], input[:, :i]))
        result = tf.stack(result, axis=-2)
        result += input
        result = self.normalize(result)
        return self.dropout(result)
