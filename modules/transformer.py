import numpy as np
import tensorflow as tf

#Encoder/Transformer functionality code is build upon https://www.tensorflow.org/tutorials/text/transformer#encoder_layer

#simple feed forward network
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class Transformer_encoder(tf.keras.layers.Layer):
  def __init__(self, inputsize, head_size, num_heads, ff_dim, dropout=0.0):
    super(Transformer_encoder, self).__init__()
    self.head_size = head_size
    self.num_heads = num_heads
    self.ff_dim = ff_dim
    self.dropout = dropout


    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)
    self.dropoutL= tf.keras.layers.Dropout(self.dropout)
    self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.conv1 = tf.keras.layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")
    self.dropoutL2 = tf.keras.layers.Dropout(self.dropout)
    self.conv2 = tf.keras.layers.Conv1D(filters=inputsize, kernel_size=1)


  def build(self, input_shape):
    print('what')
    print(input_shape)
    


  def call(self, inputs, training):

    output, adminSum = inputs
    # Normalization and Attention
    x = self.ln1(output, training=training)
    xF = self.mha(x, x, return_attention_scores=True, training=training)
    x, attention = xF
    x = self.dropoutL(x, training=training)
    res = x + output

    # Feed Forward Part
    x = self.ln2(res, training=training)
    x = self.conv1(x, training=training)
    x = self.dropoutL2(x, training=training)
    x = self.conv2(x, training=training)
    output = x + res

    return output, adminSum, attention

#encoder layer
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1, doMask = False, seed_value = 42):
    super(EncoderLayer, self).__init__()
    self.mha =  tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)#, dropout = rate)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.num_heads = num_heads

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    
    self.profiling = True
    self.doMask = doMask
    

    
    self.dropout1 = tf.keras.layers.Dropout(rate, seed=seed_value)
    self.dropout2 = tf.keras.layers.Dropout(rate, seed=seed_value)
    self.dropout3 = tf.keras.layers.Dropout(rate, seed=seed_value)
    
    self.lstm = tf.keras.layers.LSTM(dff, return_sequences=False)
    self.flatten = tf.keras.layers.Flatten()
    self.preOut = tf.keras.layers.Dense(dff)
    
    # Shape => [batch, time, features]
    self.out = tf.keras.layers.Dense(1)
    
  def build(self, input_shape):
    print(input_shape)
    
  def call(self, x, training):
    
    #print(x1)
    if self.doMask:
        x1, mask, adminSum = x
        print("aaaaa")
        print(mask)
        attn_output, attention = self.mha(
            query=x1,  # Query Q tensor.
            value=x1,  # Value V tensor.
            key=x1,  # Key K tensor.
            return_attention_scores=True,
            attentionMask=mask, # A boolean mask that prevents attention to certain positions.
            training=training # A boolean indicating whether the layer should behave in training mode.
            )

        #attn_output, attention = self.mha([x1, x1, x1], mask = mask)  # (batch_size, input_seq_len, d_model)
    else:
        x1, adminSum = x
        ##attn_output, attention = self.mha([x1, x1, x1])  # (batch_size, input_seq_len, d_model)
        attn_output, attention = self.mha(
            query=x1,  # Query Q tensor.
            value=x1,  # Value V tensor.
            key=x1,  # Key K tensor.
            return_attention_scores=True,
            training=training # A boolean indicating whether the layer should behave in training mode.
            )
    out1 = self.layernorm1(x1 + attn_output, training=training)  # (batch_size, input_seq_len, d_model)

    
    ffn_output = self.ffn(out1, training=training)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1  + ffn_output, training=training)  # (batch_size, input_seq_len, d_model)
    
    return (out2, adminSum, attention)
        
#class which can represent multiple encoder layers with correct input and output handling
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, doPosEnc=True, rate=0.1, input_vocab_size = 10000, maxLen = None, doMask=False, doEmbedding=False, seed_value=42):
    super(Encoder, self).__init__()

    self.num_heads = num_heads
    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, input_length= maxLen)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, doMask=doMask, seed_value=seed_value) for z in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate, seed=seed_value)
    
    self.doMask = doMask
    self.doEmbedding = doEmbedding


    self.num_layers=num_layers
    self.d_model=d_model
    self.num_heads=num_heads
    self.dff=dff
    self.maximum_position_encoding=maximum_position_encoding
    self.rate=rate
    self.input_vocab_size=input_vocab_size
    self.maxLen=maxLen
    self.doMask=doMask
    self.doEmbedding=doEmbedding
    self.seed_value=seed_value
    self.doPosEnc = doPosEnc
    
  def build(self, input_shape):
    if self.doMask:
        self.attention = tf.Variable(tf.ones((self.num_heads, input_shape[0][1], input_shape[0][1])), trainable=False, validate_shape=True, name='attentionMat')
    else: 
        self.attention = tf.Variable(tf.ones((self.num_heads, input_shape[1], input_shape[1])), trainable=False, validate_shape=True, name='attentionMat')
    #self.enc_layers = [Transformer_encoder(input_shape[-1], self.d_model,self. num_heads, self.dff, dropout=self.rate) for z in range(self.num_layers)]

    print(input_shape)
    print('#################')
    self.adminSumer =  0

    
  def call(self, xa, training):
    if self.doMask:
        x, mask = xa
    else:
        x = xa

    seq_len = tf.shape(x)[1]

    if self.doEmbedding:
      x = self.embedding(x)  # (batch_size, input_seq_len, d_model)    
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))    
    if self.doPosEnc:
      x += self.pos_encoding[:, :seq_len, :]
    
    if self.doMask:
        xF = (x, mask, self.adminSumer)
    else:
        xF = (x, self.adminSumer)
    print(x.shape)
    fullAttention = []
    for i in range(self.num_layers):
        xF = self.enc_layers[i](xF, training)
        x, self.adminSumer, attention = xF
        xF = x, self.adminSumer
        fullAttention.append(attention)

    attention = tf.math.reduce_mean(fullAttention, axis=0)
    
    return x, attention, fullAttention  # (batch_size, input_seq_len, d_model)

  def initPhase(self):
    for i in range(self.num_layers):
        #self.sumer = self.enc_layers[i].initPhase(self.sumer)
        self.enc_layers[i].profiling = False

  def get_config(self):

    config = super().get_config().copy()
    config.update({
        'num_layers': self.num_layers,
        'd_model': self.d_model,
        'num_heads': self.num_heads,
        'dff': self.dff,
        'maximum_position_encoding': self.maximum_position_encoding,
        'rate': self.rate,
        'input_vocab_size': self.input_vocab_size,
        'maxLen': self.maxLen,
        'doMask': self.doMask,
        'doEmbedding': self.doEmbedding,
        'seed_value': self.seed_value
    })
    return config
        

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

#safes the "best" model to load it later
# safe is based on highest val acc with the smalles val loss
#probably not optimal but doing fine
class SaveBest(tf.keras.callbacks.Callback):
    
    def __init__(self, weightsName):
        self.loss = -1
        self.weightsName = weightsName
        
    def on_epoch_end(self, epoch, logs=None):
        if (self.loss is -1 or logs['val_loss'] < self.loss):
            print('#########++++##########')
            self.loss = logs['val_loss']
            self.model.save_weights(self.weightsName, overwrite=True)

#custom scheduler with watp up steps
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=10000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def get_config(self):
    config = {
    'd_model': self.d_model.numpy(),
    'warmup_steps': self.warmup_steps
      }
    return config
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
            
            
