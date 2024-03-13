import tensorflow as tf
from tensorflow import keras
from keras import layers

def angle_between(a, b):
    # The angle between two vectors is defined as cos(theta)=(a.b)/(|a||b|)
    a_magnitude = tf.norm(a, axis=-1)
    b_magnitude = tf.norm(b, axis=-1)
    a_dot_b = tf.reduce_sum(a*b, axis=-1)
    cos_theta = a_dot_b / (a_magnitude * b_magnitude)
    return tf.math.acos(cos_theta)

class QueryLayer(layers.Layer):
    def __init__(self, embedding_dimension, dropout_rate):
        super().__init__()
        self.dropout = layers.Dropout(dropout_rate)
        self.dense = layers.Dense(embedding_dimension)
        self.add = layers.Add()
        self.normalize = layers.LayerNormalization()
        self.query = layers.Dense(embedding_dimension)
        self.mask = layers.Dense(embedding_dimension, activation='sigmoid')
        self.distance_range_width = layers.Dense(1, activation='relu')
        self.distance_range_centre = layers.Dense(1, activation='relu')
        # Maps every pair of embeddings to a new set of embeddings.
        self.embedding_map = self.add_weight(shape=(embedding_dimension, embedding_dimension, embedding_dimension), trainable=True)

    def perform_query(self, preceding_tokens, query, mask, range_width, range_centre):
        # We loop over the preceding tokens and multiply them by the positional encoding and the difference in angle from the query after applying the mask.
        # We sum the result of this and return it.
        preceding_token_count = preceding_tokens.shape[1]
        distances = tf.cast(tf.range(preceding_token_count, 0.0, -1.0), dtype=range_centre.dtype)
        # The sech function is closer to 1 the closer the input is to 0, and approaches 0 on either side.
        # This makes it ideal for our positional encoding, since we can model it such that the positional encoding will be closer to 1 if the distance is closer to the centre, and we can make it flatter using the width parameter.
        # sech = 1/cosh
        positional_encodings = 1/tf.cosh((distances - range_centre)/(range_width + 1e-7))
        # preceding_tokens has shape (batch_size, preceding_token_count, embedding_dimension). We need to reshape our positional encodings to broadcast correctly over this:
        positional_encodings = tf.expand_dims(positional_encodings, axis=-1)
        positionally_encoded_preceding_tokens = preceding_tokens * positional_encodings
        mask = tf.expand_dims(mask, axis=-2)
        masked_positionally_encoded_preceding_tokens = positionally_encoded_preceding_tokens * mask
        query = tf.expand_dims(query, axis=-2)
        # The angle is not defined for vectors of length 0, so we add a small value to ensure that the angle is defined.
        query_similarity = angle_between(masked_positionally_encoded_preceding_tokens + 1e-7, query + 1e-7)
        query_similarity = tf.expand_dims(query_similarity, axis=-1)
        query_results = positionally_encoded_preceding_tokens * query_similarity
        return tf.reduce_sum(query_results, axis=1)

    def call(self, x):
        # We essentially have to go through each query, generate and perform the query, and then pass this through the embeddings map.
        # We then return the result.
        # x has shape (batch_size, sequence_length, embedding_dimension)
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        queries = self.query(x)
        masks = self.mask(x)
        range_widths = self.distance_range_width(x)
        range_centres = self.distance_range_centre(x)
        embedding_matrices = []
        for i in range(sequence_length):
            query = queries[:, i, :]
            mask = masks[:, i, :]
            range_width = range_widths[:, i, :]
            range_centre = range_centres[:, i, :]
            preceding_tokens = x[:, :i, :]
            query_result = self.perform_query(preceding_tokens, query, mask, range_width, range_centre)
            # The embedding matrix is essentially the outer product of the query result and the current token.
            # We then pass this through a dense layer to get the final result.
            query_result = tf.expand_dims(query_result, axis=-2)
            current_token = tf.expand_dims(x[:, i, :], axis=-1)
            embedding_matrix = query_result * current_token
            embedding_matrices.append(embedding_matrix)
        embedding_matrices = tf.stack(embedding_matrices, axis=1)
        # Now we have to multiply by the embeddings map.
        # embedding_matrices has shape (batch_size, sequence_length, embedding_dimension, embedding_dimension).
        # We need to reshape the embedding matrices to make sure it broadcasts properly.
        embedding_matrices = tf.expand_dims(embedding_matrices, axis=-3)
        mapped_embeddings = tf.reduce_sum(embedding_matrices * self.embedding_map, axis=[-2, -1])
        # Add and normalize.
        result = self.add([mapped_embeddings, x])
        result = self.normalize(result)
        return self.dropout(result)
