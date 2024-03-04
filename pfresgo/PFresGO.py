import glob
import tensorflow as tf
from .PFresGO_util import get_batched_dataset, GroupWiseLinear
from .PFresGO_decoder import Decoder
import matplotlib.pyplot as plt



class PFresGO(object):

    def __init__(self, output_dim, n_channels=26, lr=0.0002, drop=0.3, num_heads=2,dff=2048, batch_size=32,
              model_name_prefix=None, hidden_size=64, label_embedding=None,num_hidden_layers=2,train=True,autoencoder_name=None):

        self.output_dim = output_dim
        self.n_channels = n_channels
        self.model_name_prefix = model_name_prefix
        self.hidden_size = hidden_size
        self.label_embedding = label_embedding
        self.num_heads = num_heads

        self.decoder = Decoder(num_hidden_layers, hidden_size, num_heads, dff, output_dim, rate=0.1)
        autoencoder_model = tf.keras.models.load_model(autoencoder_name)
        autoencoder_model = tf.keras.Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer("encoder2").output)
        autoencoder_model.trainable = True

        # build and compile model
        self._build_model(hidden_size,n_channels, output_dim, lr, drop,batch_size,label_embedding = self.label_embedding,train=train,autoencoder_model=autoencoder_model)

    def _build_model(self,hidden_size, n_channels, output_dim, lr, drop, batch_size,label_embedding,train,autoencoder_model):

        input_seq = tf.keras.layers.Input(shape=(None, n_channels), name='seq')
        inputs_padding = tf.keras.layers.Input(shape=(None,), name='padding')
        inputs_padding_mask = tf.cast(tf.equal(inputs_padding, 0), tf.float32)
        inputs_padding_mask = inputs_padding_mask[:, tf.newaxis, tf.newaxis, :]
        res_embed = tf.keras.layers.Input(shape=(None, 1024), name='res_embed')

        label_embedding = tf.expand_dims(label_embedding,axis=0)

        if train:
            label_embedding = tf.tile(label_embedding,[batch_size,1,1])

        x_aa = tf.keras.layers.Dense(hidden_size, use_bias=False, activation='relu',name='AA_embedding')(input_seq)
        x_embed = autoencoder_model(res_embed)
        if train:
            x_embed = tf.keras.layers.Dropout(drop)(x_embed)
        x_en = tf.keras.layers.Add(name='Add_embedding')([x_embed, x_aa])
        print("######################## the shape of encoder is", x_en.shape)

        # Decoding layers
        x_de, attention_weights = self.decoder(x=label_embedding,enc_output=x_en,training=train, padding_mask=inputs_padding_mask,look_ahead_mask=None)
        print("######################## the output of decoder is", x_de.shape)

        # Output layer
        output_layer = GroupWiseLinear(output_dim, self.hidden_size, bias=True)(x_de)
        print("####################### the shape of output layer is",output_layer.shape)

        tf.keras.backend.clear_session()
        self.model = tf.keras.Model(inputs=[input_seq, inputs_padding,res_embed], outputs=output_layer)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.95, beta_2=0.95)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy')
        print (self.model.summary())

    def train(self, train_tfrecord_fn, valid_tfrecord_fn, epochs=100, batch_size=64, pad_len=1000, ont='cc'):
        n_train_records = sum(1 for f in glob.glob(train_tfrecord_fn) for _ in tf.data.TFRecordDataset(f))
        n_valid_records = sum(1 for f in glob.glob(valid_tfrecord_fn) for _ in tf.data.TFRecordDataset(f))
        print ("### Training on: ", n_train_records, "contact maps.")
        print ("### Validating on: ", n_valid_records, "contact maps.")

        # train tfrecords
        batch_train = get_batched_dataset(train_tfrecord_fn,
                                          batch_size=batch_size,
                                          pad_len=pad_len,
                                          n_goterms=self.output_dim,
                                          channels=self.n_channels,
                                          ont=ont)

        # validation tfrecords
        batch_valid = get_batched_dataset(valid_tfrecord_fn,
                                          batch_size=batch_size,
                                          pad_len=pad_len,
                                          n_goterms=self.output_dim,
                                          channels=self.n_channels,
                                          ont=ont)

        # early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        # model checkpoint saving model every epoch
        mc = tf.keras.callbacks.ModelCheckpoint(self.model_name_prefix + '_best_train_model.h5', monitor='val_loss', mode='min', verbose=1,
                                                save_best_only=True, save_weights_only=False)

        # fit model
        history = self.model.fit(batch_train,
                                 epochs=epochs,
                                 validation_data=batch_valid,
                                 steps_per_epoch=n_train_records//batch_size,
                                 validation_steps=n_valid_records//batch_size,
                                 callbacks=[es, mc])

        self.history = history.history



    def predict(self, input_data):
        return self.model(input_data, training=False).numpy().reshape(-1)

    def plot_losses(self):
        plt.figure()
        plt.plot(self.history['loss'], '-')
        plt.plot(self.history['val_loss'], '-')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.model_name_prefix + '_model_loss.png', bbox_inches='tight')

        plt.figure()
        plt.plot(self.history['acc'], '-')
        plt.plot(self.history['val_acc'], '-')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.model_name_prefix + '_model_accuracy.png', bbox_inches='tight')


    def load_weights(self,path):
        self.model.load_weights(path)
        #self.model.load_weights('../trained_model/'+self.model_name_prefix + '_best_train_model.h5')


    def save_model(self,path):
        #self.model.save('../trained_model/'+self.model_name_prefix + '.h5')
        self.model.save(path)



