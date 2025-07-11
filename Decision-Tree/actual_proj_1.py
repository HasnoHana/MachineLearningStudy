from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    fr = open('dataset/lenses.txt')
    #这里的数据集是字符串 ，需要转换成数字才能拟合
    lenses = []
    for line in fr.readlines():
        lense = line.strip().split('\t')
        lenses.append(lense)
    lenses_train = []
    for line in lenses:
        lenses_train.append(line[-1])
    print(lenses)
    lenses_labels = [
        'age',
        'prescript',
        'astigmatic',
        'tearRate'
    ]
    lenses_list = []
    lenses_dict = {}
    for each_label in lenses_labels:
        for each in lenses:
            lenses_list.append(each[lenses_labels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    #数据序列化
    mid_data = LabelEncoder()
    #对每一列，即为 年龄、处方、镜片是否为近视、泪滴率
    for col in lenses_pd.columns:
        lenses_pd[col] = mid_data.fit_transform(lenses_pd[col])
    print(lenses_pd)
    #构建决策树
    classify = tree.DecisionTreeClassifier(max_depth=4)
    res = classify.fit(lenses_pd.values.tolist(), lenses_train)

    print(classify.predict([[1, 1, 1, 0]]))