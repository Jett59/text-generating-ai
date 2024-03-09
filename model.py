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
        shape=(self.units, input_shape[0]),
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
        transformed_input = tf.linalg.matvec(self.initial_input_transform, initial_input)
        contextual_input = context + transformed_input
        return self.contextual_input_transform * contextual_input

class TextModel(keras.Model):
    def __init__(self, vocab_size: int, embedding_size: int, memory_units: int):
        super().__init__(self)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.memory_units = memory_units
        self.embedding = layers.Embedding(vocab_size, embedding_size)
        self.memory_layer = Memory(memory_units)
        self.classifier = layers.Dense(vocab_size)

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
            for batch in tf.range(batch_size):
                memory = tf.zeros((self.memory_units,))
                current_predictions = []
                for t in tf.range(sequence_length):
                    memory, y_pred_t = self((memory, x[batch, t]))
                    current_predictions.append(y_pred_t)
                y_pred.append(tf.stack(current_predictions))
            y_pred = tf.stack(y_pred)
            loss = self.compiled_loss(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Reset the memory after each batch
        self.memory.assign(tf.zeros_like(self.memory))
        return {"loss": loss}
