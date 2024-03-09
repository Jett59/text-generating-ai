import tensorflow as tf
import numpy as np
import model
from tensorflow import keras
from keras import mixed_precision

# Enable mixed precision (which allows us to use tensor cores)
mixed_precision.set_global_policy('mixed_float16')

INPUT_SEQUENCE_LENGTH = 128

text = open('input.txt', 'rb').read().decode(encoding='utf-8')

vocabulary = sorted(set(text))

character_to_index = {character:index for index, character in enumerate(vocabulary)}
index_to_character = np.array(vocabulary)

text_indices = np.array([character_to_index[character] for character in text])

text_tensor = tf.data.Dataset.from_tensor_slices(text_indices)

sequences = text_tensor.batch(INPUT_SEQUENCE_LENGTH+1, drop_remainder=True)

def split_into_input_and_target(sequence):
    decoder_text = sequence[:-1]
    target_text = sequence[1:]
    return decoder_text, target_text

dataset = sequences.map(split_into_input_and_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def build_model():
    return model.TextModel(len(vocabulary), 256, 1440, 0.1)

model = build_model()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def train(model):
    model.compile(optimizer=keras.optimizers.Adam(), loss=loss, metrics=['accuracy'])
    model.fit(dataset, epochs=1)
    model.save_weights('model_weights.ckpt')

def generate_text(model, should_print = False):
    prompt = '@START' + input('Enter prompt: ')
    max_length = int(input('Number of characters to generate: '))
    temperature = float(input('Temperature: '))
    memory = model.CREATE_memory()
    for character in prompt:
        memory = model.next_memory(memory, tf.constant([character_to_index[character]]))
    generated_text = prompt
    for _ in range(max_length):
        probabilities = model.next_token(memory)
        probabilities /= temperature
        next_index = tf.random.categorical(probabilities, num_samples=1)[-1,0].numpy()
        next_character = index_to_character[next_index]
        generated_text += next_character
        if should_print:
            print(next_character, end='', flush=True)
        memory = model.next_memory(memory, tf.constant([next_index]))
    if should_print:
        print()
    return generated_text

while True:
    command = input('Enter command: ')
    if command == 'train':
        train(model)    
    elif command == 'load':
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        model.load_weights('model_weights.ckpt')
    elif command == 'generate':
        generate_text(model, should_print=True)
    elif command == 'summary':
        model.summary()
    elif command == 'exit':
        break
    else:
        print('Invalid command')
