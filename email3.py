# coding=utf-8
# @Time   : 2020/6/1 9:35:03
# @File   :K-Folds.py

import os
import shutil                           # 移动文件
import random


def readtxt(path, encoding):
    # 按encoding方式按行读取path路径文件所有行，返回行列表lines
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    return lines


def fileWalker(path):
    # 遍历语料目录，将所有语料文件绝对路径存入列表fileArray
    fileArray = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            eachpath = str(root+'\\'+fn)
            fileArray.append(eachpath)
    return fileArray


def test_set_select():
    road = readtxt('.\\trec06c\\full\\index', 'gbk')
    roads = {}
    for i in road:
        roads[i.split('/')[-2]+i.split('/')[-1][:3]] = i.split('/')[0].split(' ')[0]
    # 从spam和ham集中随机选10封移动到test集中作为测试集
    ham_filepath = r'.\email\ham'
    spam_filepath = r'.\email\spam'
    file_path = r'.\trec06c\data'
    files = fileWalker(file_path)
    # print(files)
    random.shuffle(files)
    for ech in files:
        ech_name = ech.split('\\')[-2]+ech.split('\\')[-1]
        if len(fileWalker(r'.\email')) < 6000:
            print(len(fileWalker(r'.\email')))
            if roads[ech_name] == 'ham':
                shutil.move(ech, ham_filepath)  # 把ech移动到testpath文件夹下
                os.rename(ham_filepath+'\\'+ech.split('\\')[-1], ham_filepath+'\\'+ech.split('\\')[-2]+ech.split('\\')[-1]+'.txt')  # 把ech更名为ech_name,其实可以和上一步合并
            else:
                shutil.move(ech, spam_filepath)  # 把ech移动到testpath文件夹下
                os.rename(spam_filepath + '\\' + ech.split('\\')[-1], spam_filepath+'\\'+ech.split('\\')[-2] + ech.split('\\')[-1] + '.txt')
        else:
            return


def test_set_clear():
    file_path = r'.\email'
    load_file = r'.\trec06c\data'
    files = fileWalker(file_path)
    for file in files:
        if len(fileWalker(r'.\email')) > 0:
            print(len(fileWalker(r'.\email')))
            if len(file.split('\\')[-1]) == 10:
                path_1 = file.split('\\')[-1][:3]
                path_2 = file.split('\\')[-1][3:6]
                load_file_1 = load_file + '\\' + path_1
                shutil.move(file, load_file_1)
                os.rename(load_file_1 + '\\' + path_1 + path_2 + '.txt', load_file_1 + '\\' + path_2)
            else:
                path_1 = file.split('\\')[-1][-10:-7]
                path_2 = file.split('\\')[-1][-7:-4]
                load_file_1 = load_file + '\\' + path_1
                shutil.move(file, load_file_1)
                os.rename(load_file_1 + '\\' + file.split('\\')[-1], load_file_1 + '\\' + path_2)
        else:
            return


if __name__ == '__main__':
    print(len(fileWalker(r'.\email')))
    test_set_clear()
