from os import listdir

from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np

#将32*32的图像转换为1024大小图像
def img2vector(filename):
    vect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            vect[0,32*i+j] = int(line_str[j])
    return vect

#手写数据集测试
def handwriting_class_test():
    #训练集标签
    train_labels = []
    train_file_list = listdir('dataset/digitis_recognize/trainingDigits')
    n = len(train_file_list)
    train_mat = np.zeros((n,1024))
    for i in range(n):
        file_name = train_file_list[i]
        class_number = int(file_name.split('_')[0])
        train_labels.append(class_number)
        train_mat[i,:] = img2vector('dataset/digitis_recognize/trainingDigits/%s' % file_name)
    neigh = KNN(n_neighbors=3 , algorithm='auto')
    neigh.fit(train_mat, train_labels)

    #测试集计数
    test_file_list = listdir('dataset/digitis_recognize/testDigits')
    error_count = 0.0
    m = len(test_file_list)
    for i in range(m):
        file_name = test_file_list[i]
        class_number = int(file_name.split('_')[0])
        test_mat = img2vector('dataset/digitis_recognize/testDigits/%s' % file_name)
        class_result = neigh.predict(test_mat)
        print('the class_result is %d, the real class_number is %d' % (class_result, class_number))
        if class_result != class_number:
            error_count += 1.0
    print('the total error rate is %f' % (error_count/m))

if __name__ == '__main__':
    handwriting_class_test()