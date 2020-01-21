from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf, numpy as np, os, time, json, random

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

def generate_text(model, start_string,length=1000):
  num_generate = length

  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []

  temperature = 1.0
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

filename = input("who are we testing?: ")
text = open("original_tweets/"+filename+'.txt', 'rb').read().decode(encoding='utf-8')

#vocab things
vocab = sorted(set(text))
vocab_size = len(vocab)
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

#RNN config
embedding_dim = 256
rnn_units = 1024

#checkpoint config
checkpoint_dir = './checkpoints/training_checkpoints_'+filename
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

gentweets_file = './generated_tweets/'+filename+'.json'
tweets = []
try:
  while True:
    tweet_length = int(np.random.normal(80,24))
    tweet = generate_text(model, start_string=random.sample(vocab,1)[0],length=tweet_length)
    print(tweet,"\n")
    tweets.append(tweet)
except KeyboardInterrupt:
  pass

f = open(gentweets_file,"w")
f.write(json.dumps(tweets,indent=2))
f.close()
print(len(tweets))
print("done!")