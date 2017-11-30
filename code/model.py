import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
from cell import GRUCell, BasicLSTMCell, BasicRNNCell

PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']


class RNN(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            num_labels,
            embed,
            learning_rate=0.001,
            max_gradient_norm=5.0,
            learning_rate_decay_factor=0.1):
        #todo: implement placeholders
        # PROBLEMS REMAIN
        self.texts = tf.placeholder(dtype=tf.string, shape=[None, None])  # shape: batch*len
        self.texts_length = tf.placeholder(dtype=tf.int32, shape=[None])  # shape: batch
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None])  # shape: batch
        self.output_keep_prob = tf.placeholder(dtype=tf.float32, shape=[])
        
        self.symbol2index = MutableHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=UNK_ID,
                shared_name="in_table",
                name="in_table",
                checkpoint=True)
        # build the vocab table (string to index)
        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_update_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)

        self.index_input = self.symbol2index.lookup(self.texts)  # batch*len
        
        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

        self.embed_input = tf.nn.embedding_lookup(self.embed, self.index_input) #batch*len*embed_unit

        if num_layers == 1:
            # basic rnn
            # cell = BasicRNNCell(num_units)
            # outputs, states = dynamic_rnn(cell, self.embed_input, self.texts_length, dtype=tf.float32, scope="rnn")

            # gru
            # cell = GRUCell(num_units)
            # outputs, states = dynamic_rnn(cell, self.embed_input, self.texts_length, dtype=tf.float32, scope="rnn")

            # lstm
            # cell = BasicLSTMCell(num_units)
            # outputs, states = dynamic_rnn(cell, self.embed_input, self.texts_length, dtype=tf.float32, scope="rnn")
            # states = states[1]

            # final model
            cell = tf.nn.rnn_cell.DropoutWrapper(BasicLSTMCell(num_units), output_keep_prob=self.output_keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(BasicLSTMCell(num_units), output_keep_prob=self.output_keep_prob)
            outputs, states = bidirectional_dynamic_rnn(cell, cell_bw, self.embed_input, self.texts_length,
                                                                            dtype=tf.float32, scope="rnn")
            states = states[0][1] + states[1][1]
        else:
            cells = []
            cells_bw = []
            for _ in range(num_layers):
                cell = tf.nn.rnn_cell.DropoutWrapper(GRUCell(num_units), output_keep_prob=output_keep_prob)
                cells.append(cell)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(GRUCell(num_units), output_keep_prob=output_keep_prob)
                cells_bw.append(cell_bw)
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.MultiRNNCell(cells_bw, state_is_tuple=True)
            outputs, states = bidirectional_dynamic_rnn(cell, cell_bw, self.embed_input, self.texts_length,
                                                    dtype=tf.float32, scope="stacked_rnn")
            states = states[0][num_layers - 1] + states[1][num_layers - 1]

        #todo: implement unfinished networks
        self.w1 = tf.Variable(tf.random_normal(shape=[num_units, num_labels], stddev=tf.sqrt(2.0 / (num_units + num_labels))))
        self.b1 = tf.Variable(tf.constant(0.0, shape=[num_labels]))
        logits = tf.matmul(states, self.w1) + self.b1

        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits), name='loss')
        mean_loss = self.loss / tf.cast(tf.shape(self.labels)[0], dtype=tf.float32)
        self.predict_labels = tf.argmax(logits, 1, 'predict_labels', output_type=tf.int32)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.labels, self.predict_labels), tf.int32), name='accuracy')

        self.params = tf.trainable_variables()
            
        # calculate the gradient of parameters
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(mean_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)

        tf.summary.scalar('loss/step', self.loss)
        for each in tf.trainable_variables():
            tf.summary.histogram(each.name, each)

        self.merged_summary_op = tf.summary.merge_all()
        
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
    
    def train_step(self, session, data, summary=False):
        input_feed = {self.texts: data['texts'],
                self.texts_length: data['texts_length'],
                self.labels: data['labels'], self.output_keep_prob: 0.5}
        output_feed = [self.loss, self.accuracy, self.gradient_norm, self.update]
        if summary:
            output_feed.append(self.merged_summary_op)
        return session.run(output_feed, input_feed)
