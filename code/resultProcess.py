#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 文本数据预处理
import codecs
import re
import os
filename1="F:/天池/糖尿病文本分析/DiabetesKG/data/result3/"
filename2='F:/天池/糖尿病文本分析/DiabetesKG/data/ruijin_round1_test_b_20181112/ruijin_round1_test_b_20181112/'
filename3="F:/天池/糖尿病文本分析/DiabetesKG/data/submit2/"
max_len = 60
        
def padding_word(sen):
    if len(sen) >= max_len:
        return sen[:max_len]
    else:
        return sen

#if __name__=="__main__":
def resultPro():
    pathDir = os.listdir(filename1)
    num=[]#首字符序号
    for i in range(len(pathDir)):
        num.clear()
        number=1
        a=os.path.join(filename2, pathDir[i])
        b=os.path.join(filename1, pathDir[i])
        f = open(a, encoding='utf-8')
        txt=f.read()
        f.close
        text = []#标注数组
        text1=[] #原文

        with codecs.open(a, 'rb', 'utf8') as inp:
            lines=inp.readlines()
            for line in lines:
                text1.append(line)

        with codecs.open(b, 'rb', 'utf8') as inp:
            lines=inp.readlines()
            for line in lines:
                if line[0]=='$':
                    continue
                else:
                    text.append(line)
        output_path = os.path.join(filename3, pathDir[i][0:len(pathDir[i])-4]+".ann")
        with codecs.open(output_path, 'a', 'utf-8') as outp:          
            for j in range(len(text)):
                flag=text[j].isspace()
                if flag==False:
                    line = re.split('[$]', text[j].strip())
                    l1=line[len(line)-1].isspace()
                    l2=line[len(line)-1]
                    L=['',' ','\n']
                    l3=l2 in L
                    if l3:
                        length=(len(line)-1)/2
                    else:
                        length=(len(line))/2
                    for k in range(int(length)):
                        for m1 in range(len(text1[j])):
                            if text1[j][m1]==line[2*k+1][0]:
                                if m1+len(line[2*k+1])<=len(text1[j])-1:
                                    sum=0
                                    if j>0:
                                        for m2 in range(j):
                                            sum+=(len(text1[m2]))
                                    if (sum+m1) not in num:
                                        str1='T'+str(number)
                                        outp.write(str1+"\t"+line[2*k]+" "+str(sum+m1)+" "+str(sum+m1+len(line[2*k+1]))+"\t"+line[2*k+1])
                                        outp.write('\n')
                                        number+=1
                                        for mm in range(len(line[2*k+1])):
                                            c=sum+mm+m1
                                            num.append(c)
                                    else:
                                        continue
                                else:
                                    sum=0
                                    if j>0:
                                        for m2 in range(j):
                                            sum+=(len(text1[m2]))
                                    if (sum+m1) not in num:
                                        str1='T'+str(number)
                                        outp.write(str1+"\t"+line[2*k]+" "+str(sum+m1)+" "+str(sum+len(text1[j])-1)+";"+str(sum+len(text1[j]))+" "+str(sum+m1+len(line[2*k+1])+1)+"\t"+txt[sum+m1:sum+len(text1[j])-1]+' '+txt[sum+len(text1[j]):sum+m1+len(line[2*k+1])+1])
                                        outp.write('\n')
                                        number+=1
                                        for mm in range(len(line[2*k+1])+1):
                                            c=sum+mm+m1
                                            num.append(c)
                                    else:
                                        continue
                                break
