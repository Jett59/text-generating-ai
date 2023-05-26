import tensorflow as tf
import numpy as np
import model

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
    return model.TextModel(3, 4, 128, len(vocabulary), 0.001)

model = build_model()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def train(model):
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.fit(dataset, epochs=1)
    model.save_weights('model_weights.ckpt')

def generate_text(model, should_print = False):
    prompt = input('Enter prompt: ')
    max_length = int(input('Number of characters to generate: '))
    temperature = float(input('Temperature: '))
    prompt_indices = [character_to_index[character] for character in "@START" + prompt]
    model_input = tf.expand_dims(prompt_indices, 0)
    generated_text = ''
    for i in range(max_length):
        predictions = model(model_input)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, 1)[-1, 0].numpy()
        character = index_to_character[predicted_id]
        generated_text += character
        # Concattenate the predicted character to the prompt
        prompt_indices = prompt_indices[1:] + [predicted_id]
        model_input = tf.expand_dims(prompt_indices, 0)
        if should_print:
            print(character, end='', flush=True)
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
    elif command == 'exit':
        break
    else:
        print('Invalid command')
