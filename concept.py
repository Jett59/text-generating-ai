import tensorflow as tf
from tensorflow import keras
from keras import layers

class ConceptLayer(layers.Layer):
    def __init__(self, embedding_dimension, dropout_rate):
        super().__init__()
        self.dropout = layers.Dropout(dropout_rate)
        self.normalize = layers.LayerNormalization()
        self.concept_map = self.add_weight(name='concept_map', shape=(embedding_dimension, embedding_dimension, embedding_dimension), trainable=True)

    def apply_positional_encoding(self, token, distance):
        return token / tf.cast(distance ** 2, token.dtype)

    def calculate_summed_conceptual_matrix(self, current_token, preceding_tokens):
        preceding_token_count = preceding_tokens.shape[1]
        if preceding_token_count > 0:
            positioned_preceding_tokens = tf.map_fn(lambda i: self.apply_positional_encoding(preceding_tokens[:, i], preceding_token_count - i), tf.range(preceding_token_count), fn_output_signature=preceding_tokens.dtype)
            # positioned_preceding_tokens has the shape (preceding_token_count, batch_size, embedding_dimension).
            # This is ok, since we are going to sum anyway so we can just sum along the first axis instead of the second one.
        else:
            positioned_preceding_tokens = tf.zeros((0, current_token.shape[0], current_token.shape[1]), current_token.dtype)
        summed_positional_preceding_tokens = tf.reduce_sum(positioned_preceding_tokens, axis=0)
        # We need to add axes between batch and embedding_dimension, otherwise the matrix multiplication will go across the batch.
        current_token = tf.expand_dims(current_token, axis=1)
        summed_positional_preceding_tokens = tf.expand_dims(summed_positional_preceding_tokens, axis=1)
        summed_conceptual_matrix = tf.matmul(current_token, summed_positional_preceding_tokens, transpose_a=True)
        return summed_conceptual_matrix

    def call(self, input):
        conceptual_matrices = []
        for i in range(input.shape[-2]):
            conceptual_matrices.append(self.calculate_summed_conceptual_matrix(input[:, i], input[:, :i]))
        conceptual_matrices = tf.stack(conceptual_matrices, axis=1)
        # concept_matrices is in the shape (batch_size, sequence_length, embedding_dimension, embedding_dimension).
        # We want it to be in the shape (batch_size, sequence_length, 1, embedding_dimension, embedding_dimension) so we can multiply it with the concept map, which has shape (embedding_dimension, embedding_dimension, embedding_dimension).
        conceptual_matrices = tf.expand_dims(conceptual_matrices, axis=2)
        result = conceptual_matrices * self.concept_map
        # Now we have to sum along the last two axes to get it back into the shape (batch_size, sequence_length, embedding_dimension).
        result = tf.reduce_sum(result, axis=[-2, -1])
        # Add and normalize, then we're done.
        result += input
        result = self.normalize(result)
        return self.dropout(result)
