import tensorflow as tf


class BasicRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
            #todo: implement the new_state calculation given inputs and state
            if not hasattr(self, '_w'):
                self._w = tf.get_variable('basic_rnn_w', shape=[inputs.get_shape()[-1] + state.get_shape()[-1], self._num_units], dtype=tf.float32, initializer=tf.orthogonal_initializer())
                self._b = tf.get_variable('basic_rnn_b', shape=[self._num_units], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            new_state = self._activation(tf.matmul(tf.concat([state, inputs], 1), self._w) + self._b)

        return new_state, new_state


class GRUCell(tf.contrib.rnn.RNNCell):
    '''Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078).'''

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            #We start with bias of 1.0 to not reset and not update.
            #todo: implement the new_h calculation given inputs and state
            if not hasattr(self, '_wz'):
                self._wz = tf.get_variable('gru_cell_wz', shape=[inputs.get_shape()[-1] + state.get_shape()[-1], self._num_units], dtype=tf.float32, initializer=tf.orthogonal_initializer())
                self._bz = tf.get_variable('gru_cell_bz', shape=[self._num_units], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
                self._wr = tf.get_variable('gru_cell_wr', shape=[inputs.get_shape()[-1] + state.get_shape()[-1], self._num_units], dtype=tf.float32, initializer=tf.orthogonal_initializer())
                self._br = tf.get_variable('gru_cell_br', shape=[self._num_units], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
                self._wc = tf.get_variable('gru_cell_wc', shape=[inputs.get_shape()[-1] + state.get_shape()[-1], self._num_units], dtype=tf.float32, initializer=tf.orthogonal_initializer())
                self._bc = tf.get_variable('gru_cell_bc', shape=[self._num_units], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            z = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, state], 1), self._wz) + self._bz)
            r = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, state], 1), self._wr) + self._br)
            _h = self._activation(tf.matmul(tf.concat([inputs, r * state], 1), self._wc) + self._bc)
            new_h = (1.0 - z) * state + z * _h
            
        return new_h, new_h


class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    '''Basic LSTM cell (http://arxiv.org/abs/1409.2329).'''

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
            c, h = state
            #For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            #todo: implement the new_c, new_h calculation given inputs and state (c, h)
            if not hasattr(self, '_wi'):
                self._wi = tf.get_variable('basic_lstm_cell_wi', dtype=tf.float32, shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self._num_units], initializer=tf.orthogonal_initializer())
                self._bi = tf.get_variable('basic_lstm_cell_bi', dtype=tf.float32, shape=[self._num_units], initializer=tf.constant_initializer(0.0))
                self._wo = tf.get_variable('basic_lstm_cell_wo', dtype=tf.float32, shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self._num_units], initializer=tf.orthogonal_initializer())
                self._bo = tf.get_variable('basic_lstm_cell_bo', dtype=tf.float32, shape=[self._num_units], initializer=tf.constant_initializer(0.0))
                self._wf = tf.get_variable('basic_lstm_cell_wf', dtype=tf.float32, shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self._num_units], initializer=tf.orthogonal_initializer())
                self._bf = tf.get_variable('basic_lstm_cell_bf', dtype=tf.float32, shape=[self._num_units], initializer=tf.constant_initializer(1.0))
                self._wc = tf.get_variable('basic_lstm_cell_wc', dtype=tf.float32, shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self._num_units], initializer=tf.orthogonal_initializer())
                self._bc = tf.get_variable('basic_lstm_cell_bc', dtype=tf.float32, shape=[self._num_units], initializer=tf.constant_initializer(0.0))
            i = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, h], 1), self._wi) + self._bi)
            o = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, h], 1), self._wo) + self._bo)
            f = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, h], 1), self._wf) + self._bf)
            _c = self._activation(tf.matmul(tf.concat([inputs, h], 1), self._wc) + self._bc)
            new_c = f * c + i * _c
            new_h = o * self._activation(new_c)

            return new_h, (new_c, new_h)
