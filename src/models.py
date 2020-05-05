import numpy as np
import tensorflow as tf

import keras
import config


def softmax_cross_entropy_with_logits(y_true, y_pred):

    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss


class GeneralModel:
    def __init__(self, reg_constant, learning_rate, input_dimension, output_dimension):
        self.reg_constant = reg_constant
        self.learning_rate = learning_rate
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split, batch_size=batch_size)

    def write(self, version):
        self.model.save(config.RUN_FOLDER + 'models/version' + "{0:0>4}".format(version) + '.h5')

    def read(self, version):
        return keras.models.load_model(
            config.RUN_ARCHIVE_FOLDER + "models/version" + "{0:0>4}".format(
                version) + '.h5',
            custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})


class residual_CNN(GeneralModel):
    def __init__(self, reg_constant, learning_rate, input_dimension, output_dimension, hidden_layers):
        GeneralModel.__init__(self, reg_constant, learning_rate, input_dimension, output_dimension)
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.model = self._build_model()

    def conv_layer(self, layer, filters, kernel_size):
        convolution = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            data_format='channels_first',
            use_bias=False,
            activation='linear',
            kernel_regularizer=keras.regularizers.l2(self.reg_constant)
        )
        layer = convolution(layer)
        layer = keras.layers.BatchNormalization(axis=1)(layer)
        layer = keras.layers.LeakyReLU()(layer)
        return layer

    def residual_layer(self, inp, filters, kernel_size):
        layer = self.conv_layer(inp, filters, kernel_size)
        layer = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format='channels_first',
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=keras.regularizers.l2(self.reg_constant)
        )(layer)
        layer = keras.layers.BatchNormalization(axis=1)(layer)
        layer = keras.layers.Add()([inp, layer])
        layer = keras.layers.LeakyReLU()(layer)
        return layer

    def policy_head(self, layer):
        layer = keras.layers.Conv2D(
            filters=2,
            kernel_size=(1, 1),
            data_format='channels_first',
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=keras.regularizers.l2(self.reg_constant)
        )(layer)
        layer = keras.layers.BatchNormalization(axis=1)(layer)
        layer = keras.layers.LeakyReLU()(layer)
        layer = keras.layers.Flatten()(layer)
        layer = keras.layers.Dense(
            self.output_dimension,
            use_bias=False,
            activation='linear',
            kernel_regularizer=keras.regularizers.l2(self.reg_constant),
            name='policy_head'
        )(layer)
        return layer

    def value_head(self, layer):
        layer = keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            data_format='channels_first',
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=keras.regularizers.l2(self.reg_constant)
        )(layer)
        layer = keras.layers.BatchNormalization(axis=1)(layer)
        layer = keras.layers.LeakyReLU()(layer)
        layer = keras.layers.Flatten()(layer)
        layer = keras.layers.Dense(
            20,
            use_bias=False,
            activation='linear',
            kernel_regularizer=keras.regularizers.l2(self.reg_constant),
        )(layer)
        layer = keras.layers.LeakyReLU()(layer)
        layer = keras.layers.Dense(
            1,
            use_bias=False,
            activation='tanh',
            kernel_regularizer=keras.regularizers.l2(self.reg_constant),
            name='value_head'
        )(layer)
        return layer

    def _build_model(self):
        input_layer = keras.engine.input_layer.Input(shape=self.input_dimension, name='input')
        properties = self.hidden_layers[0]
        layer = self.conv_layer(input_layer, properties['filters'], properties['kernel_size'])

        if len(self.hidden_layers) > 1:
            for properties in self.hidden_layers[1:]:
                layer = self.residual_layer(layer, properties['filters'], properties['kernel_size'])

        policy_head = self.policy_head(layer)
        value_head = self.value_head(layer)

        model = keras.models.Model(inputs=input_layer, outputs=[policy_head, value_head])
        optimizer = keras.optimizers.SGD(lr=self.learning_rate, momentum=config.MOMENTUM)
        model.compile(
            optimizer=optimizer,
            loss={
                'policy_head': softmax_cross_entropy_with_logits,
                'value_head': 'mean_squared_error'
                },
            loss_weights={
                'policy_head': 0.5,
                'value_head': 0.5
                }
            )
        return model

    def convert_to_input(self, state):
        input_to_model = state.binary
        input_to_model = np.reshape(input_to_model, self.input_dimension)
        return input_to_model
