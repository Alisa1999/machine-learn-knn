# coding=utf-8
# @Time   : 2020/5/26 21:02:58
# @File   :KNN2.py

import numpy as np
import gzip
import matplotlib.pyplot as plt


#按32位读取，主要为读校验码、图片数量、尺寸准备的
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#抽取图片，并按照需求，可将图片中的灰度值二值化，按照需求，可将二值化后的数据存成矩阵或者张量
def extract_images(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
#        print(magic, num_images, rows, cols)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows*cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data


#抽取标签
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]//10 # shape[0] stands for the num of row
    init_shape = newInput.shape[0]
    newInput = newInput.reshape(1, init_shape)
    diff = np.tile(newInput, (numSamples, 1)) - dataSet[:numSamples] # Subtract element-wise
    squaredDiff = diff ** 2 # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis = 1) # sum is performed by row
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)
    classCount = {} # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]


        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    ## step 5: the max voted class will return
    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


def misclass_show(i, k):
    test_x = extract_images('D:/360Downloads/机器学习-knn/data/mnist/test_images', True, True)
    test_y = extract_labels('D:/360Downloads/机器学习-knn/data/mnist/test_labels')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    im = np.array(test_x[i])
    im = im.reshape(28, 28)
    fig = plt.figure()
    fig.add_subplot(111)
    plt.imshow(im, cmap='gray')
    plt.title("数字%d错分为%d" % (test_y[i], k))
    plt.show()


def magic_show(i):
    test_x = extract_images('D:/360Downloads/机器学习-knn/data/mnist/test_images', True, True)
    test_y = extract_labels('D:/360Downloads/机器学习-knn/data/mnist/test_labels')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    im = np.array(test_x[i])
    im = im.reshape(28, 28)
    fig = plt.figure()
    fig.add_subplot(111)
    plt.imshow(im, cmap='gray')
    plt.title("训练集图片：数字%d" % test_y[i])
    plt.show()

# 主函数，先读图片，然后用于测试手写数字
def testHandWritingClass(k):
    ## step 1: load data
    print("step 1: load data...")
    train_x = extract_images('D:/360Downloads/机器学习-knn/data/mnist/train_images', True, True)
    train_y = extract_labels('D:/360Downloads/机器学习-knn/data/mnist/train_labels')
    test_x = extract_images('D:/360Downloads/机器学习-knn/data/mnist/test_images', True, True)
    test_y = extract_labels('D:/360Downloads/机器学习-knn/data/mnist/test_labels')

    ## step 2: training...
    print("step 2: training...")
    pass

    ## step 3: testing
    print("step 3: testing...")
    numTestSamples = test_x.shape[0]
    matchCount = 0
    test_num = numTestSamples//10
    for i in range(test_num):
        predict = kNNClassify(test_x[i], train_x, train_y, k)
        if predict == test_y[i]:
            matchCount += 1
#        else:
#            misclass_show(i, predict)
        if i % 100 == 0:
            print("完成%d张图片"%(i+1))
            print("图片正确数量：%d张"%(matchCount))
    accuracy = float(matchCount) / test_num
    return accuracy


def testHandWritingClass_1(k):
    ## step 1: load data
    print("step 1: load data...")
    train_x = extract_images('D:/360Downloads/机器学习-knn/data/mnist/train_images', True, True)
    train_y = extract_labels('D:/360Downloads/机器学习-knn/data/mnist/train_labels')
    test_x = extract_images('D:/360Downloads/机器学习-knn/data/mnist/test_images', True, True)
    test_y = extract_labels('D:/360Downloads/机器学习-knn/data/mnist/test_labels')

    ## step 2: training...
    print("step 2: training...")
    pass

    ## step 3: testing
    print("step 3: testing...")
    numTestSamples = test_x.shape[0]
    matchCount = 0
    test_num = numTestSamples//10
    for i in range(test_num):
        predict = kNNClassify(train_x[i], train_x, train_y, k)
        if predict == train_y[i]:
            matchCount += 1
#        else:
#            misclass_show(i, predict)
        if i % 100 == 0:
            print("完成%d张图片"%(i+1))
            print("图片正确数量：%d张"%(matchCount))
    accuracy = float(matchCount) / test_num
    return accuracy


if __name__ == '__main__':
    rang = [1, 2, 3, 4, 5, 10, 20, 40, 80, 100, 120]
    x = []
    y = []
    for i in rang:
        x.append(1-testHandWritingClass(i))
        y.append(1-testHandWritingClass_1(i))
fig = plt.figure()
plt.plot(rang, x, label='test', linestyle='--')
plt.plot(rang, y, label='train', linestyle='dotted')
plt.xlabel('K')
plt.ylabel('misclassification rate')
plt.legend(loc='upper right')
plt.savefig("knn_1.png")
plt.show()

