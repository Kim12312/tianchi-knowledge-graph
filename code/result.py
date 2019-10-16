#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 文本数据预处理
import codecs
import pandas as pd
import numpy as np
import re
import os
import math
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
filename1="F:/天池/糖尿病文本分析/DiabetesKG/data/submit2/"
max_len = 60

if __name__=="__main__":
    pathDir = os.listdir(filename1)
    text=[]
    num=[]
    for i in range(len(pathDir)):
        num.clear()
        a=os.path.join(filename1, pathDir[i])
        with codecs.open(a, 'rb', 'utf8') as inp:
            lines=inp.readlines()
            for line in lines:
                line1 = re.split('[\t]',line)
                line3 = re.split('[ ;]',line1[1])
                line2=[]
                for j in range(len(line3)):
                    if j>0:
                        line2.append(int(line3[j]))
                text.append(line2)
                number=int(line1[0][1:len(line1[0])])
                if num==[]:
                    num.append(number)
                else:
                    if num[len(num)-1]+1==number:
                        num.append(number)
                    else:
                        print(a+"序号有错")
    print(text[1])
