import tensorflow as tf
from tensorflow import keras
from keras import layers

class Head(layers.Layer):
    def __init__(self, head_embedding_dimension, dropout_rate):
        super().__init__()
        self.head_embedding_dimension = head_embedding_dimension
        self.dropout_rate = dropout_rate
        self.concept_map = self.add_weight(name='concept_map', shape=(head_embedding_dimension, head_embedding_dimension, head_embedding_dimension), trainable=True)
        self.dropout = layers.Dropout(dropout_rate)

    def calculate_summed_conceptual_matrix(self, current_token, summed_positional_preceding_tokens):
        # We need to add axes between batch and embedding_dimension, otherwise the matrix multiplication will go across the batch.
        current_token = tf.expand_dims(current_token, axis=1)
        summed_positional_preceding_tokens = tf.expand_dims(summed_positional_preceding_tokens, axis=1)
        summed_conceptual_matrix = tf.matmul(current_token, summed_positional_preceding_tokens, transpose_a=True)
        return summed_conceptual_matrix

    def call(self, input):
        conceptual_matrices = []
        summed_positional_preceding_tokens = tf.zeros((input.shape[0], input.shape[-1]), dtype=input.dtype)
        # The first matrix is unique in that there are no preceding tokens, which means that it is always equal to 0.
        conceptual_matrices.append(tf.zeros((input.shape[0], input.shape[-1], input.shape[-1]), dtype=input.dtype))
        for i in range(1, input.shape[-2]):
            summed_positional_preceding_tokens += input[:, i-1]
            summed_positional_preceding_tokens /= 1.2 # This makes the positional factor equal to 1/1.2^d, where d is the distance.
            # This is because the token placed in the list first will be divided over and over again in this loop, giving the geometric pattern of 1/1.2^d.
            conceptual_matrices.append(self.calculate_summed_conceptual_matrix(input[:, i], summed_positional_preceding_tokens))
        conceptual_matrices = tf.stack(conceptual_matrices, axis=1)
        # concept_matrices is in the shape (batch_size, sequence_length, embedding_dimension, embedding_dimension).
        # We want it to be in the shape (batch_size, sequence_length, 1, embedding_dimension, embedding_dimension) so we can multiply it with the concept map, which has shape (embedding_dimension, embedding_dimension, embedding_dimension).
        conceptual_matrices = tf.expand_dims(conceptual_matrices, axis=2)
        result = conceptual_matrices * self.concept_map
        # Now we have to sum along the last two axes to get it back into the shape (batch_size, sequence_length, embedding_dimension).
        result = tf.reduce_sum(result, axis=[-2, -1])
        return self.dropout(result)

class ConceptLayer(layers.Layer):
    def __init__(self, embedding_dimension, head_count, dropout_rate):
        assert embedding_dimension % head_count == 0
        super().__init__()
        self.dense = layers.Dense(embedding_dimension)
        self.normalize = layers.LayerNormalization()
        self.heads = [Head(embedding_dimension // head_count, dropout_rate) for _ in range(head_count)]

    def call(self, input):
        input = self.dense(input)
        # We need to give each head the right shape.
        input = tf.reshape(input, (input.shape[0], input.shape[1], len(self.heads), -1))
        head_outputs = [head(input[:, :, i]) for i, head in enumerate(self.heads)]
        result = tf.concat(head_outputs, axis=-1)
        return self.normalize(result)
