import numpy as np
import tensorflow as tf

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_goemb_size, rate=0.1, **kwargs):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dff = dff
    self.target_goemb_size = target_goemb_size
    self.rate = rate

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    super(Decoder, self).__init__(**kwargs)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

    attention_weights = {}
    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

    return x, attention_weights

  def get_config(self):
    config = super(Decoder,self).get_config().copy()
    config.update({
      'num_layers': self.num_layers,
      "d_model": self.d_model,
      "num_heads":self.num_heads,
      "dff": self.dff,
      "target_goemb_size": self.target_goemb_size,
      "rate": self.rate,
      "dec_layers1": DecoderLayer(self.d_model, self.num_heads, self.dff, self.rate),
      "embedding": tf.keras.layers.Embedding(self.target_goemb_size, self.d_model),
      "droupout":tf.keras.layers.Dropout(self.rate)
          })
    return config

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

def scaled_dot_product_attention(q, k, v, mask):

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.rate = rate

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)


  def call(self, x, enc_output, training,look_ahead_mask, padding_mask):

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      "d_model" : self.d_model,
      "num_heads" : self.num_heads,
      "dff" : self.dff,
      "rate" : self.rate,
      'mha1': MultiHeadAttention(self.d_model, self.num_heads),
      "mha2": MultiHeadAttention(self.d_model, self.num_heads),
      "ffn":point_wise_feed_forward_network(self.d_model, self.dff),
      "layernorm1": tf.keras.layers.LayerNormalization(epsilon=1e-6),
      "layernorm2": tf.keras.layers.LayerNormalization(epsilon=1e-6),
      "layernorm3": tf.keras.layers.LayerNormalization(epsilon=1e-6),
      "droupout1":tf.keras.layers.Dropout(self.rate),
      "droupout2": tf.keras.layers.Dropout(self.rate),
      "droupout3": tf.keras.layers.Dropout(self.rate)
          })
    return config



class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):

    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    scaled_attention, attention_weights = scaled_dot_product_attention( q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      "num_heads" : self.num_heads,
      "d_model" : self.d_model,
      "depth" : self.depth,
      "wq" : tf.keras.layers.Dense(self.d_model),
      "wk": tf.keras.layers.Dense(self.d_model),
      "wv": tf.keras.layers.Dense(self.d_model),
      'dense': tf.keras.layers.Dense(self.d_model)
          })
    return config

