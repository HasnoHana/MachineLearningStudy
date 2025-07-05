import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
def file2matrix(filename):
    fr = open(filename)
    array_all_lines = fr.readlines()
    array_len = len(array_all_lines)
    array_mat = np.zeros((array_len,3))
    class_label_list = []
    idx = 0
    for line in array_all_lines:
        line = line.strip()
        line_list = line.split('\t')
        array_mat[idx,:] = line_list[0:3]
        #1为不喜欢、2为一般、3为喜欢
        if line_list[-1] == 'didntLike':
            class_label_list.append(1)
        elif line_list[-1] == 'smallDoses':
            class_label_list.append(2)
        elif line_list[-1] == 'largeDoses':
            class_label_list.append(3)
        idx += 1
    return array_mat, class_label_list
#数据可视化
def show_datas(dataset, data_labels):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    fig, axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8))
    len_labels = len(data_labels)
    labels_color = []
    for i in data_labels:
        if i == 1:
            labels_color.append('black')
        elif i == 2:
            labels_color.append('orange')
        elif i == 3:
            labels_color.append('red')
    axs[0][0].scatter(x=dataset[:,0],y=dataset[:,1],color=labels_color,s=15,alpha=.5)
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', fontproperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', fontproperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', fontproperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    axs[0][1].scatter(x=dataset[:, 0], y=dataset[:, 2], color=labels_color, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', fontproperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', fontproperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', fontproperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=dataset[:, 1], y=dataset[:, 2], color=labels_color, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', fontproperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', fontproperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', fontproperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    didntlike_line = mlines.Line2D([],[],color='black',marker='.',
                                   markersize=6,label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')

    axs[0][0].legend(handles=[didntlike_line, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntlike_line, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntlike_line, smallDoses, largeDoses])

    plt.show()

#数据归一化
def autoNorm(dataset):
    #计算每一列的最小值
    min_val = dataset.min(0)
    max_val = dataset.max(0)
    ranges = max_val - min_val
    n = dataset.shape[0]
    norm_dataset = dataset - np.tile(min_val, (n,1))
    norm_dataset = norm_dataset / np.tile(ranges, (n,1))
    return norm_dataset, ranges, min_val

def classify(in_x, dataset, labels, k):
    n = dataset.shape[0]
    diff_mat = np.tile(in_x, (n,1)) - dataset
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_distances_idx = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distances_idx[i]]
        class_count[vote_label] = class_count.get(vote_label,0) + 1
    sorted_class_count = sorted(class_count.items(), key=lambda item:item[1], reverse=True)
    return sorted_class_count[0][0]

def test_classify():
    filename = 'dataset/datingTestSet.txt'
    dataset , data_labels = file2matrix(filename)
    test_rate = 0.10
    norm_dataset, ranges, min_val = autoNorm(dataset)
    n = norm_dataset.shape[0]
    test_num = int(n * test_rate)
    error_count = 0.0
    for i in range(test_num):
        classify_result = classify(norm_dataset[i,:],norm_dataset[test_num:n,:],data_labels[test_num:n],4)
        if classify_result != data_labels[i]:
            error_count += 1.0
    error_rate = (error_count / float(test_num))*100
    print('错误率：%f%%' % error_rate)

def classify_person():
    result_list = ['dontlike','smalllike','verylike']
    percent_game = float(input('玩视频游戏所耗时间百分比：'))
    ice_cream = float(input('每周消费的冰激淋公升数：'))
    fly_distance = float(input('每年获得的飞行常客里程数：'))
    filename = 'dataset/datingTestSet.txt'
    dataset , data_labels = file2matrix(filename)
    norm_dataset, ranges, min_val = autoNorm(dataset)
    in_x = np.array([fly_distance,percent_game,ice_cream])
    norm_in_arr = (in_x - min_val) / ranges
    classify_result = classify(norm_in_arr,norm_dataset,data_labels,3)
    print(' classified as : %s' % result_list[classify_result-1])

if __name__ == '__main__':
    classify_person()