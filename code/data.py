import numpy as np
import os
import random
import pickle
import codecs
import re
filename1 = 'F:/天池/糖尿病文本分析/data/ruijin_round2_train/'
filename3 = "F:/天池/糖尿病文本分析/data/tag"
filename4 = "F:/天池/糖尿病文本分析/data"
trainTxtArr = []
trainTxtArr1 = []
trainAnnArr = []
# 遍历指定目录，显示目录下的所有文件名
def eachTrainFile(filepath):
    pathDir = os.listdir(filepath)
    for i in range(len(pathDir)):
        a = pathDir[i][-3:len(pathDir[i])]
        if a == "txt":
            trainTxtArr.append(os.path.join(filename1, pathDir[i]))
            trainTxtArr1.append(pathDir[i])
        else:
            trainAnnArr.append(os.path.join(filename1, pathDir[i]))

tag2label = {"O": 0,
             "Disease": 1, "I-Disease": 2,
             "Reason": 3, "I-Reason": 4,
             "Symptom": 5, "I-Symptom": 6,
             "Test": 7, "I-Test": 8,
             "Test_Value": 9, "I-Test_Value": 10,
             "Drug": 11, "I-Drug": 12,
             "Frequency": 13, "I-Frequency": 14,
             "Amount": 15, "I-Amount": 16,
             "Method": 17, "I-Method": 18,
             "Treatment": 19, "I-Treatment": 20,
             "Operation": 21, "I-Operation": 22,
             "Anatomy": 23, "I-Anatomy": 24,
             "SideEff": 25, "I-SideEff": 26,
             "Level": 27, "I-Level": 28,
             "Duration": 29, "I-Duration": 30
             }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            if len(line.strip().split())==2:
                [char, label] = line.strip().split()
                if char != "," and char != "，" and char != "。" and char != "、" and char != "[" and char != "【" and char != "】" and char != "]" and char != "{" and char != "}" and char != "(" and char != ")" and char != "（" and char != "）" and char != "“" and char != "”" and char != "‘" and char != "’" and char != "：" and char != "；" and char != ':' and char != ";":
                    sent_.append(char)
                    tag_.append(label)
                else:
                    if (sent_ != []) & (tag_ != []):
                        data.append((sent_, tag_))
                    sent_, tag_ = [], []
                #sent_.append(char)
                #tag_.append(label)
        else:
            if (sent_!=[])&(tag_!=[]):
                data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            '''if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
                '''
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

# 通过抛出异常
def is_num_by_except(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

# 将原始数据做标注（B I）
def origin2tag(number):
    f = open(trainTxtArr[number],encoding='utf-8')
    s=trainTxtArr[number][0:len(trainTxtArr[number])-3]+"ann"
    f1 = open(s, encoding='utf-8')
    txt=f.read()
    lines = f1.readlines()
    value=[]
    index=[]#起始位置和终止位置
    tag=[]
    for line in lines:
        if line[0]=='T':
            for db in re.split(";| |!|\t|\n|,", line):
                value.append(db)
            for i in range(4):
                if i<2:
                    if is_num_by_except(value[2+i]):
                        index.append(int(value[2+i]))
                        tag.append(value[1])
                else:
                    if is_num_by_except(value[4])&is_num_by_except(value[5]):
                        index.append(int(value[2+i]))
                        tag.append(value[1])
            value.clear()
    f1.close()
    str=""
    index1=[]#便于后续清洗数据
    for i in range(len(index)):
        start=index[i]
        if i%2==0:
            index1.append(start)
            start=start+1
            while start<index[i+1]:
                index1.append(start)
                start = start + 1

    for i in range(len(txt)):
        if txt[i]!="\n":
            if i in index1:
                if i in index:
                    a=index.index(i)
                    if a%2==0:
                        # 双数为起始位置
                        if (txt[i]==" ")|(txt[i]=="")|(txt[i]=="\t")|(txt[i]=="\n"):
                            str +=  "\n"
                        else:
                            str += txt[i]
                            str += "\t"+"B-"+tag[a]+"\n"
                    else:
                        if (txt[i]==" ")|(txt[i]=="")|(txt[i]=="\t")|(txt[i]=="\n"):
                            str +=  "\n"
                        else:
                            str += txt[i]
                            str += "\t"+"O"+"\n"
                else:
                    b=index1.index(i)
                    if i>=len(txt)-1:
                        if (txt[i]==" ")|(txt[i]=="")|(txt[i]=="\t")|(txt[i]=="\n"):
                            str +=  "\n"
                        else:
                            str += txt[i]
                            str += "\t" + "I-" + tag[a] + "\n"
                    else:
                        if b>=len(index1)-1:
                            if (txt[i] == " ") | (txt[i] == "") | (txt[i] == "\t") | (txt[i] == "\n"):
                                str += "\n"
                            else:
                                str += txt[i]
                                str += "\t" + "I-" + tag[a] + "\n"
                        elif (abs(index1[b+1]-index1[b])>1):
                            if (txt[i] == " ") | (txt[i] == "") | (txt[i] == "\t") | (txt[i] == "\n"):
                                str += "\n"
                            else:
                                str += txt[i]
                                str += "\t" + "I-" + tag[a] + "\n"
                        else:
                            if (txt[i] == " ") | (txt[i] == "") | (txt[i] == "\t") | (txt[i] == "\n"):
                                str += "\n"
                            else:
                                str += txt[i]
                                str += "\t" + "I-" + tag[a] + "\n"
            else:
                if (txt[i] == " ") | (txt[i] == "") | (txt[i] == "\t") | (txt[i] == "\n"):
                    str += "\n"
                else:
                    str+=txt[i]
                    str += "\t" + "O" + "\n"
    f.close()

    str1=os.path.join(filename3, trainTxtArr1[number])
    output_data = codecs.open(str1, 'w', 'utf-8')
    output_data.write(str)
    output_data.close()

def getTrainData():
    str1 = ""
    pathDir = os.listdir(filename3)
    for i in range(len(pathDir)):
        str = os.path.join(filename3, pathDir[i])
        f = open(str, encoding='utf-8')
        txt = f.read()
        str1 += txt
        str1 += '\n'
        f.close()
    output_data = codecs.open(os.path.join(filename4, "train_data.txt"), 'w', 'utf-8')
    output_data.write(str1)
    output_data.close()

if __name__ == '__main__':
    pathDir = os.listdir(filename1)
    for i in range(len(pathDir)):
        str = os.path.join(filename1, pathDir[i])
        # 1.获得训练数据集
    eachTrainFile(filename1)
    for i in range(len(trainTxtArr)):
        origin2tag(i)
    getTrainData()
    # 2.获得word2id
    vocab_build("F:/pythonProgram/BiLSTM_CRF/data_path/word2id.pkl",
                    "F:/pythonProgram/BiLSTM_CRF/data_path/train_data.txt", 5)