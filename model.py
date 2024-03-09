import tensorflow as tf
from tensorflow import keras
from keras import layers

class Memory(keras.layers.Layer):
    def __init__(self, units: int):
        super().__init__()
        self.initial_input_transform = layers.Dense(units)
        self.contextual_input_transform = layers.Dense(units)
        self.normalization = layers.LayerNormalization()

    def call(self, inputs):
        context, initial_input = inputs
        input_contribution = self.initial_input_transform(initial_input)
        context_contribution = self.contextual_input_transform(context)
        new_context = input_contribution + context_contribution
        new_context = self.normalization(new_context)
        return new_context

class TextModel(keras.Model):
    def __init__(self, vocab_size: int, embedding_size: int, memory_units: int, dropout_rate: float):
        super().__init__(self)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.memory_units = memory_units
        self.embedding = layers.Embedding(vocab_size, embedding_size)
        self.memory_layer = Memory(memory_units)
        self.classifier = layers.Dense(vocab_size, activation="softmax")
        self.dropout = layers.Dropout(dropout_rate)

    def CREATE_memory(self):
        return tf.zeros((self.memory_units))

    def next_memory(self, memory, input):
        embeddings = self.embedding(input)
        new_memory = self.memory_layer((memory, embeddings))
        new_memory = self.dropout(new_memory)
        return new_memory

    def next_token(self, memory):
        return self.classifier(memory)

    def call(self, inputs):
        memory, input = inputs
        new_memory = self.next_memory(memory, input)
        return new_memory, self.next_token(new_memory)

    def train_step(self, data):
        x, y = data
        # x has the shape (batch_size, sequence_length).
        # The way this model is intended to be run is over each input element separately, with the memory held from the previous iteration.
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        with tf.GradientTape() as tape:
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
