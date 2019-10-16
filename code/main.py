import tensorflow as tf
from Batch import BatchGenerator
from bilstm_crf import Model
from utils import *
from dataProcessing import *
from resultProcess import *

filename1="F:/天池/糖尿病文本分析/DiabetesKG/data/processedData/pkl/"
filename2='F:/天池/糖尿病文本分析/DiabetesKG/data/ruijin_round1_test_b_20181112/ruijin_round1_test_b_20181112/'

def train():
    with open(os.path.join(filename1, "train1.pkl"), 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)

    data_train = BatchGenerator(x_train, y_train, shuffle=True)
    data_valid = BatchGenerator(x_valid, y_valid, shuffle=False)
    data_test = BatchGenerator(x_test, y_test, shuffle=False)
    epochs = 31
    batch_size = 32

    config = {}
    config["lr"] = 0.001
    config["embedding_dim"] = 100
    config["sen_len"] = len(x_train[0])
    config["batch_size"] = batch_size
    config["embedding_size"] = len(word2id) + 1
    config["tag_size"] = len(tag2id)
    config["pretrained"] = False

    embedding_pre = []
    #训练模型
    print("begin to train...")
    model = Model(config, embedding_pre, dropout_keep=0.5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train(model, sess, saver, epochs, batch_size, data_train, data_valid, id2word, id2tag)

def test():
    with open(os.path.join(filename1, "train1.pkl"), 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)

    epochs = 31
    batch_size = 32

    config = {}
    config["lr"] = 0.001
    config["embedding_dim"] = 100
    config["sen_len"] = len(x_train[0])
    config["batch_size"] = batch_size
    config["embedding_size"] = len(word2id) + 1
    config["tag_size"] = len(tag2id)
    config["pretrained"] = False

    embedding_pre = []
    #利用训练完成的模型进行测试
    print("begin to extraction...")
    model = Model(config, embedding_pre, dropout_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt is None:
            print('Model not found, please train your model first')
        else:
            path = ckpt.model_checkpoint_path
            print('loading pre-trained model from %s.....' % path)
            saver.restore(sess, path)
            pathDir = os.listdir(filename2)
            for i in range(len(pathDir)):
                a = os.path.join(filename2, pathDir[i])
                b = os.path.join('F:/天池/糖尿病文本分析/DiabetesKG/data/result3/', pathDir[i])
                extraction(a, b, model, sess, word2id, id2tag, batch_size)
if __name__=="__main__":
    #1.数据预处理
    dataProcess()
    # 2.训练模型
    train()
    # 3.模型测试
    test()
    #4.结果输出
    resultPro()