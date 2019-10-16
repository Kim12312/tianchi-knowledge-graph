#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 文本数据预处理
import codecs
import pandas as pd
import numpy as np
import re
import collections
import os
import pickle
filename1='/home/lingang/Kim/DiabetesKG/data/ruijin_round1_train_20181022/ruijin_round1_train2_20181022'
filename2="/home/lingang/Kim/DiabetesKG/data/processData/"
filename3="/home/lingang/Kim/DiabetesKG/data/processData/wordtagspilt/"
filename4="/home/lingang/Kim/DiabetesKG/data/processData/pkl/"
filename5="/home/lingang/Kim/DiabetesKG/data/processedData/"
filename6="/home/lingang/Kim/DiabetesKG/data/processedData/pkl/"
trainTxtArr = []
trainTxtArr1 = []
trainAnnArr = []
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, type("s")):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

# 遍历指定目录，显示目录下的所有文件名
def eachTrainFile(filepath):
    pathDir = os.listdir(filepath)
    for i in range(len(pathDir)):
        a=pathDir[i][-3:len(pathDir[i])]
        if a=="txt":
            trainTxtArr.append(os.path.join(filename1, pathDir[i]))
            trainTxtArr1.append(pathDir[i])
        else:
            trainAnnArr.append(os.path.join(filename1, pathDir[i]))


# pkl文件是python里面保存文件的一种格式，如果直接打开会显示一堆序列化的东西。
def data2pkl1():
    datas = list()
    labels = list()
    tags = set()
    str1 = os.path.join(filename5, "train11.txt")
    str2 = os.path.join(filename6, "train1.pkl")
    input_data = codecs.open(str1, 'r', 'utf-8')
    for line in input_data.readlines():
        line = line.split()
        linedata = []
        linelabel = []
        numNotO = 0
        for word in line:
            word = word.split(':')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
            if word[1] != 'O':
                numNotO += 1
        if numNotO != 0:
            datas.append(linedata)
            labels.append(linelabel)

    input_data.close()
    all_words = flatten(datas)  # compiler.flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)

    tags = [i for i in tags]
    tag_ids = range(len(tags))
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)

    word2id["unknow"] = len(word2id)+1

    max_len = 60
    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:  
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len: 
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids
    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))

    from sklearn.model_selection import train_test_split
    x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=43)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.1, random_state=43)

    with open(str2, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(x_valid, outp)
        pickle.dump(y_valid, outp)

#通过抛出异常
def is_num_by_except(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

# 将原始数据做标注（B M E）
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
        #else:
            #index1.append(start)

    for i in range(len(txt)):
        if txt[i]!="\n":
            if i in index1:
                if i in index:
                    a=index.index(i)
                    if a%2==0:
                        # 双数为起始位置
                        str += txt[i]
                        str += ":B_"+tag[a]+" "
                    else:
                        str += txt[i]
                        str += ":O"+" "
                else:
                    b=index1.index(i)
                    if i>=len(txt)-1:
                        str += txt[i]
                        str += ":E_" + tag[a]+" "
                    else:
                        if b>=len(index1)-1:
                            str += txt[i]
                            str += ":E_" + tag[a]+" "
                        elif (abs(index1[b+1]-index1[b])>1):
                            str += txt[i]
                            str += ":E_" + tag[a]+" "
                        else:
                            str += txt[i]
                            str += ":M_" + tag[a]+" "
            else:
                str+=txt[i]
                str += ":O"+" "
    f.close()

    str1=os.path.join(filename2, trainTxtArr1[number])
    output_data = codecs.open(str1, 'w', 'utf-8')
    output_data.write(str)
    output_data.close()

# 将标注文本进行分解，每个实体为一行
def tagsplit(number):
    str1 = os.path.join(filename2, trainTxtArr1[number])
    str2 = os.path.join(filename3, trainTxtArr1[number])
    with open(str1, 'rb') as inp:
        texts = inp.read().decode('utf-8')
    sentences = re.split('[，。！？、‘’“”（）]/[O]', texts)
    output_data = codecs.open(str2, 'w', 'utf-8')
    for sentence in sentences:
        if sentence != " ":
            output_data.write(sentence.strip() + '\n')
    output_data.close()

#if __name__=='__main__':
def dataProcess():
    eachTrainFile(filename1)
    for i in range(len(trainTxtArr)):
        origin2tag(i)
        tagsplit(i)

    str1=""
    pathDir = os.listdir(filename3)
    for i in range(len(pathDir)):
        str=os.path.join(filename3, pathDir[i])
        f = open(str,encoding='utf-8')
        txt=f.read()
        str1+=txt
        str1+='\n'
        f.close()
    output_data = codecs.open(os.path.join(filename5, "train1.txt"), 'w', 'utf-8')
    output_data.write(str1)
    output_data.close()
    
    str2 = os.path.join(filename5, "train11.txt")
    with open(os.path.join(filename5, "train1.txt"), 'rb') as inp:
        texts = inp.read().decode('utf-8')
    sentences = re.split('[，。！？、‘’“”（）]', texts)
    output_data = codecs.open(str2, 'w', 'utf-8')
    for sentence in sentences:
        if sentence != " ":
            output_data.write(sentence.strip() + '\n')
    output_data.close()
    data2pkl1()
                
    

