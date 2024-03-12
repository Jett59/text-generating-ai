import tensorflow as tf
from tensorflow import keras
from keras import layers

class ConceptLayer(layers.Layer):
    def __init__(self, embedding_dimension, head_count, dropout_rate):
        assert embedding_dimension % head_count == 0
        embeddings_per_head = embedding_dimension // head_count
        super().__init__()
        self.head_count = head_count
        self.dense = layers.Dense(embedding_dimension)
        self.normalize = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.concept_map = self.add_weight(shape=(head_count, embeddings_per_head, embeddings_per_head, embeddings_per_head), trainable=True)

    def calculate_summed_conceptual_matrix(self, current_token, summed_positional_preceding_tokens):
        # We need to add axes between head and embedding_dimension, otherwise the matrix multiplication will go across the heads.
        current_token = tf.expand_dims(current_token, axis=-1)
        summed_positional_preceding_tokens = tf.expand_dims(summed_positional_preceding_tokens, axis=-2)
        summed_conceptual_matrix = tf.matmul(current_token, summed_positional_preceding_tokens)
        return summed_conceptual_matrix

    def call(self, input):
        input = self.dense(input)
        # We need to give each head the right shape.
        input = tf.reshape(input, (input.shape[0], input.shape[1], self.head_count, -1))
        conceptual_matrices = []
        summed_positional_preceding_tokens = tf.zeros((input.shape[0], input.shape[-2], input.shape[-1]), dtype=input.dtype)
        # The first matrix is unique in that there are no preceding tokens, which means that it is always equal to 0.
        conceptual_matrices.append(tf.zeros((input.shape[0], input.shape[-2], input.shape[-1], input.shape[-1]), dtype=input.dtype))
        for i in range(1, input.shape[1]):
            summed_positional_preceding_tokens += input[:, i-1]
            summed_positional_preceding_tokens /= 1.2 # This makes the positional factor equal to 1/1.2^d, where d is the distance.
            # This is because the token placed in the list first will be divided over and over again in this loop, giving the geometric pattern of 1/1.2^d.
            conceptual_matrices.append(self.calculate_summed_conceptual_matrix(input[:, i], summed_positional_preceding_tokens))
        conceptual_matrices = tf.stack(conceptual_matrices, axis=1)
        # conceptual_matrices is in the shape (batch_size, sequence_length, head_count, embedding_dimension, embedding_dimension).
        # We want it to be in the shape (batch_size, sequence_length, head_count, 1, embedding_dimension, embedding_dimension) so we can multiply it with the concept map, which is of rank 3.
        conceptual_matrices = tf.expand_dims(conceptual_matrices, axis=-3)
        result = conceptual_matrices * self.concept_map
        print(self.concept_map.shape, result.shape)
        # Now we have to sum along the last two axes to get it back into the shape (batch_size, sequence_length, embedding_dimension).
        result = tf.reduce_sum(result, axis=[-2, -1])
        print(result.shape, input.shape)
        result += input
        # Then we have to reshape to remove the head_count dimension.
        result = tf.reshape(result, (result.shape[0], result.shape[1], -1))
        result = self.normalize(result)
        return self.dropout(result)
