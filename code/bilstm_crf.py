# -*- coding: utf-8 -*
import numpy as np
import tensorflow as tf

#未进行精细调参
class Model:
    def __init__(self,config,embedding_pretrained,dropout_keep=1):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.embedding_size = config["embedding_size"]
        self.embedding_dim = config["embedding_dim"]
        self.sen_len = config["sen_len"]
        self.tag_size = config["tag_size"]
        self.pretrained = config["pretrained"]
        self.dropout_keep = dropout_keep
        self.embedding_pretrained = embedding_pretrained
        self.input_data = tf.placeholder(tf.int32, shape=[self.batch_size,self.sen_len], name="input_data")
        self.labels = tf.placeholder(tf.int32,shape=[self.batch_size,self.sen_len], name="labels")
        self.embedding_placeholder = tf.placeholder(tf.float32,shape=[self.embedding_size,self.embedding_dim], name="embedding_placeholder")
        with tf.variable_scope("bilstm_crf") as scope:
            self._build_net()
    def _build_net(self):
        word_embeddings = tf.get_variable("word_embeddings",[self.embedding_size, self.embedding_dim])
        if self.pretrained:
            embeddings_init = word_embeddings.assign(self.embedding_pretrained)

        input_embedded = tf.nn.embedding_lookup(word_embeddings, self.input_data)#提取张量里相应索引所对应的元素
        input_embedded = tf.nn.dropout(input_embedded,self.dropout_keep)#防止过拟合，dropout指神经元被选中的概率

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)#n_hidden表示神经元的个数，forget_bias就是LSTM们的忘记系数，如果等于1，就是不会忘记任何信息。如果等于0，就都忘记。state_is_tuple默认就是True，官方建议用True，就是表示返回的状态用一个元祖表示。
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, # 前向RNN
                                                                         lstm_bw_cell, # 后向RNN
                                                                         input_embedded,# 输入
                                                                         dtype=tf.float32,
                                                                         time_major=False,
                                                                         scope=None)

        bilstm_out = tf.concat([output_fw, output_bw], axis=2)


        # Fully connected layer.
        W = tf.get_variable(name="W", shape=[self.batch_size,2 * self.embedding_dim, self.tag_size],
                        dtype=tf.float32)

        b = tf.get_variable(name="b", shape=[self.batch_size, self.sen_len, self.tag_size], dtype=tf.float32,
                        initializer=tf.zeros_initializer())

        bilstm_out = tf.tanh(tf.matmul(bilstm_out, W) + b)

        # Linear-CRF.
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(bilstm_out, self.labels, tf.tile(np.array([self.sen_len]),np.array([self.batch_size])))

        loss = tf.reduce_mean(-log_likelihood)

        # Compute the viterbi sequence and score (used for prediction and test time).
        #self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(bilstm_out, self.transition_params,tf.tile(np.array([self.sen_len]),np.array([self.batch_size])))
        self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(bilstm_out, self.transition_params,tf.cast(tf.tile(np.array([self.sen_len]),np.array([self.batch_size])),tf.int32))

        # Training ops.
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(loss)
