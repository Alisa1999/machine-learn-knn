# coding=utf-8
# @Time   : 2020/5/30 22:17:49
# @File   :Spam sorting.py

# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Date     : 2017-05-09 09:29:13
# @Author   : Alan Lau (rlalan@outlook.com)
# @Language : Python3.5
# @EditTime : 2018-04-09 13:04:13
# @Editor   : Galo
# @Function : 1.遍历包含50条数据的email文件夹，获取文件列表
#             2.使用random.shuffle()函数打乱列表
#             3.截取乱序后的文件列表前10个文件路径，并转移到test文件夹下，作为测试集。
#             4.Bayes 垃圾邮件分类及100次结果及分析统计图表


# from fwalker import fun
# from reader import readtxt
import os
import shutil                           # 移动文件
import random                           # 随机化抽取文件
import numpy as np                      # 画图
import matplotlib.pyplot as plt         # 画图
from nltk.corpus import stopwords       # 去停用词
import jieba
from string import digits

cachedStopWords = stopwords.words("english")    # 选用英文停用词词典


def fileWalker(path):
    # 遍历语料目录，将所有语料文件绝对路径存入列表fileArray
    fileArray = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            eachpath = str(root+'\\'+fn)
            fileArray.append(eachpath)
    return fileArray


def test_set_select():
    # 从spam和ham集中随机选10封移动到test集中作为测试集
    # filepath = r'.\email'
    # testpath = r'.\email\test'
    # files = fileWalker(filepath)
    # random.shuffle(files)
    # top10 = files[:10]
    # for ech in top10:
    #     ech_name = testpath+'\\'+('_'.join(ech.split('\\')[-2:]))  # 取分割后的后两项用_拼接
    #     shutil.move(ech, testpath)  # 把ech移动到testpath文件夹下
    #     os.rename(testpath+'\\'+ech.split('\\')[-1], ech_name)  # 把ech更名为ech_name,其实可以和上一步合并
    #     # print('%s moved' % ech_name)
    # return
    pass


def test_set_clear():
    # 移动test测试集中文件回spam和ham中，等待重新抽取测试集
    filepath = r'..\email'
    testpath = r'..\email\test'
    files = fileWalker(testpath)
    for ech in files:
        ech_initial = filepath + '\\' + '\\'.join(' '.join(ech.split('\\')[-1:]).split('_'))  # 分析出文件移入测试集前的目录及名称
        ech_move = filepath + '\\' + (' '.join(ech.split('\\')[-1:]).split('_'))[0]  # 分析出文件移入测试集前的目录
        shutil.move(ech, ech_move)  # 把ech移动到ech_move文件夹下
        os.rename(ech_move+'\\'+' '.join(ech.split('\\')[-1:]), ech_initial)  # 恢复原名称
        # print('%s moved' % ech)
    return


def readtxt(path, encoding):
    # 按encoding方式按行读取path路径文件所有行，返回行列表lines
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    return lines


def email_parser(email_path):
    # 去特殊字符标点符号，返回纯单词列表clean_word
    punctuations = """,.:<>()*&^%$#@!'";~`[]{}|、\\/~+_-=?"""
    content_list = readtxt(email_path, 'gbk')
    content = (' '.join(content_list)).replace('\r\n', ' ').replace('\t', ' ').replace('\n', ' ')
    clean_word = []
    for punctuation in punctuations:
        content = (' '.join(content.split(punctuation))).replace('  ', ' ')
        clean_word = [word.lower()
                      for word in content.split(' ') if word.lower() not in cachedStopWords and len(word) > 2]
        # 此处去了停用词，可不去，影响不大
    s = []
    remove_digits = str.maketrans('', '', digits)
    for con in clean_word:
        con = seg_sentence(con)
        for co in con.split(' '):
            co = co.translate(remove_digits)
            if co != '' and len(co) > 1:
                s.append(co)
    return s


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('./words_stop.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

def get_word(email_file):
    # 获取email_file路径下所有文件的总单词列表，append入word_list，extend入word_set并去重转为set
    word_list = []
    word_set = []
    email_paths = fileWalker(email_file)
    for email_path in email_paths:
        clean_word = email_parser(email_path)
        word_list.append(clean_word)
        word_set.extend(clean_word)
        # print(set(word_set))
    return word_list, set(word_set)


def count_word_prob(email_list, union_set):
    # 返回训练集词频字典word_prob
    word_prob = {}
    email_list.append(list(union_set))
    union_set = union_set
    for word in union_set:
        counter = 0
        for email in email_list:
            for word_1 in email:
                if word == word_1:
                    counter += 1
                else:
                    continue
        word_prob[word] = counter
    return word_prob


def filter(ham_word_pro, spam_word_pro, test_file):
    # 进行一次对测试集(10封邮件)的测试，输出对测试集的判断结果
    # 并返回准确率right_rate，以及把spam误判成ham和总误判次数对应情况
    right = 0
    wrong = 0
    wrong_spam = 0
    test_paths = fileWalker(test_file)
    for test_path in test_paths:
        # 贝叶斯推断计算与判别实现
        email_spam_prob = 0.0
        spam_prob = 0.5  # 假设P(spam) = 0.5
        ham_prob = 0.5  # P(ham) = 0.5
        file_name = test_path.split('\\')[-1]
        prob_dict = ham_word_pro
        prob_dict_1 = spam_word_pro
        count = 0
        word_dict = {}
        word_list = email_parser(test_path)
        words = set(word_list)
        no_word = []
        for word in words:
            if word not in prob_dict:
                no_word.append(word)
        for k, v in prob_dict.items():
            prob_dict[k] += len(no_word)
        for k, v in prob_dict_1.items():
            prob_dict_1[k] += len(no_word)
        for word in no_word:
            prob_dict[word] = 1
            prob_dict_1[word] = 1
        for word in words:
            for word_1 in word_list:
                if word == word_1:
                    count += 1
            word_dict[word] = count
            count = 0
        psw = 10**100
        phw = 10**100
        sum = 0
        for k, v in prob_dict.items():  # 统计测试集所出现单词word的P(spam|word)
            sum += v
        for k, v in prob_dict.items():  # 统计测试集所出现单词word的P(spam|word)
            prob_dict[k] = v/sum
        sum = 0
        for k, v in word_dict.items():  # 统计测试集所出现单词word的P(spam|word)
            for i in range(v):
                phw *= prob_dict[k]
        for k, v in prob_dict_1.items():  # 统计测试集所出现单词word的P(spam|word)
            sum += v
        for k, v in prob_dict_1.items():  # 统计测试集所出现单词word的P(spam|word)
            prob_dict_1[k] = v/sum
        for k, v in word_dict.items():  # 统计测试集所出现单词word的P(spam|word)
            for i in range(v):
                psw *= prob_dict_1[k]
        psw *= spam_prob
        phw *= ham_prob
        right = 0
        wrong = 0
        if psw > phw:  # P(spam|word1word2…wordn) > 0.9 认为是spam垃圾邮件
            print(file_name, 'spam', email_spam_prob)
            if file_name.split('_')[0] == 'spam':  # 记录是否判断准确
                right += 1
            else:
                wrong += 1
                print('***********************Wrong Prediction***********************')
        else:
            print(file_name, 'ham', email_spam_prob)
            if file_name.split('_')[0] == 'ham':  # 记录是否判断准确
                right += 1
            else:
                wrong += 1
                print('***********************Wrong Prediction***********************')
    right_rate = right/(right+wrong)  # 计算一个测试集的准确率
    return right_rate


def main():
    # 主函数
    right_rate_list = []
    wrong_spam_rate_list = []
    ham_file = r'.\email\ham'
    spam_file = r'.\email\spam'
    test_file = r'.\email\test'
    for i in range(1):
        # 进行100次抽取测试集，测试并记录准确率，注意训练集应不包含测试集
        test_set_select()  # 构造测试集
        ham_list, ham_set = get_word(ham_file)
        spam_list, spam_set = get_word(spam_file)
        union_set = ham_set | spam_set  # 合并纯单词集合
        ham_word_pro = count_word_prob(ham_list, union_set)  # 单词在ham中的出现频率字典
        spam_word_pro = count_word_prob(spam_list, union_set)  # 单词在spam里的出现频率字典
        rig = filter(ham_word_pro, spam_word_pro, test_file)
        right_rate_list.append(rig)  # 返回正确率
        test_set_clear()  # 还原测试集
    # 画出100次判别的准确率散点图
    x = range(1)
    y = right_rate_list
    plt.scatter(x, y)
    plt.title('Correct Rate of 100 Times')
    plt.show()
    return


if __name__ == '__main__':
    main()




