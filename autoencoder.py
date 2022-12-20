import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from autoencoder_util import get_batched_dataset


class AutoEncoder:
    def __init__(self, input_dim, hidden_dims, learning_rate, batch_size, num_steps,model_name):
        self.input_dim = input_dim
        self.num_hidden = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.model_name = model_name
        self._build_model()

    def _build_model(self):

        input_seq = tf.keras.layers.Input(shape=(None, self.input_dim), name='seq')
        x = input_seq
        x0 = tf.keras.layers.Dense(self.num_hidden[0], use_bias=True, activation='relu', name='encoder{}'.format(0))(x)
        x1 = tf.keras.layers.Dense(self.num_hidden[1], use_bias=True, activation='relu', name='encoder{}'.format(1))(x0)
        x2 = tf.keras.layers.Dense(self.num_hidden[2], use_bias=True, activation='relu', name='encoder{}'.format(2))(x1)
        decoder_dim = [i for i in reversed(self.num_hidden)]
        y0 = tf.keras.layers.Dense(decoder_dim[0], use_bias=True, activation='relu', name='decoder{}'.format(0))(x2)
        y1 = tf.keras.layers.Dense(decoder_dim[1], use_bias=True, activation='relu', name='decoder{}'.format(1))(y0)
        output_seq = tf.keras.layers.Dense(decoder_dim[2], use_bias=True, activation='relu', name='decoder{}'.format(2))(y1)

        self.model = tf.keras.Model(inputs=input_seq, outputs=output_seq)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.95, beta_2=0.95)
        self.model.compile(optimizer=optimizer, loss='mse')
        print(self.model.summary())

    def train(self, train_data, valid_data):

        batch_train = get_batched_dataset(train_data,
                                          batch_size=self.batch_size
                                          )

        batch_valid = get_batched_dataset(valid_data,
                                          batch_size=self.batch_size
                                          )
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        mc = tf.keras.callbacks.ModelCheckpoint('./trained_model'+self.model_name + '_best_train_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)

        history = self.model.fit(batch_train,
                                 epochs=self.num_steps,
                                 validation_data=batch_valid,
                                 steps_per_epoch= 29902 // self.batch_size,
                                 validation_steps= 3323 // self.batch_size,
                                 callbacks=[es, mc]
                                 )

        self.history = history.history

    def save_model(self):
        self.model.save('./trained_model'+self.model_name + '.h5')




