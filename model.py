import tensorflow as tf
from tensorflow import keras
from keras import layers

class Memory(keras.layers.Layer):
    def __init__(self, units: int):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        memory_shape, input_shape = input_shape
        self.initial_input_transform = self.add_weight(
        shape=(self.units, input_shape[1]),
        initializer="random_normal",
        trainable=True,
    )
        self.contextual_input_transform = self.add_weight(
            shape=(self.units, self.units),
            initializer="random_normal",
            trainable=True,
        )
    
    def call(self, inputs):
        context, initial_input = inputs
        transformed_input = tf.matmul(initial_input, self.initial_input_transform, transpose_b=True)
        contextual_input = context + transformed_input
        return tf.matmul(contextual_input, self.contextual_input_transform)

class TextModel(keras.Model):
    def __init__(self, vocab_size: int, embedding_size: int, memory_units: int):
        super().__init__(self)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.memory_units = memory_units
        self.embedding = layers.Embedding(vocab_size, embedding_size)
        self.memory_layer = Memory(memory_units)
        self.classifier = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        memory, input = inputs
        embeddings = self.embedding(input)
        new_memory = self.memory_layer((memory, embeddings))
        return new_memory, self.classifier(new_memory)

    def train_step(self, data):
        x, y = data
        # x has the shape (batch_size, sequence_length).
        # The way this model is intended to be run is over each input element separately, with the memory held from the previous iteration.
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        with tf.GradientTape() as tape:
            # y_pred needs to be created over time, so we create it here.
            y_pred = []
            memory = tf.zeros((batch_size, self.memory_units,))
            for t in range(sequence_length):
                tokens = x[:, t]
                memory, y_pred_t = self((memory, tokens))
                y_pred.append(y_pred_t)
            y_pred = tf.stack(y_pred, axis=1)
            loss = self.compiled_loss(y, y_pred)
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}
