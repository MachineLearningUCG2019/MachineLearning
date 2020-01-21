from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf, numpy as np, os, time

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

filename = input("who are we imitating?: ")
filename = filename
text = open("original_tweets/"+filename+'.txt', 'rb').read().decode(encoding='utf-8')

#get vocabulary, do things with it
vocab = sorted(set(text))
vocab_size = len(vocab)
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

#sequence things
seq_length = 70
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

#RNN config
BATCH_SIZE = 64
BUFFER_SIZE = 10000
embedding_dim = 256
rnn_units = 1024
EPOCHS=50

#checkpoint config
checkpoint_dir = './checkpoints/training_checkpoints_'+filename
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

model.compile(optimizer='adam', loss=loss)

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

print("done!")