import csv
import numpy as np
import jieba
import pyprind
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
import sklearn.svm as svm

filename_data = "dataset.csv"
filename_stopwords = ["baidu_stopwords.txt", "cn_stopwords.txt", "hit_stopwords.txt", "scu_stopwords.txt", "my_stopwords.txt"]
extract_top_k = 30


def text_handle(old_text):
    old_text = old_text.replace('\n', '').replace('\t', '').replace(' ', '')
    new_text = ''.join([i for i in old_text if not i.encode('UTF-8').isalnum()])
    return new_text


def seg_text(file_path):
    # read csv file
    print('读入csv文件', file_path)
    product_category = []
    product_name = []
    products = []
    with open(file_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            product_category.append(row[0])
            words = " ".join(jieba.cut(text_handle(row[1])))
            product_name.append(words)
            products.append([row[0], words])
    return product_category, product_name, products


def get_stop_word(file_paths):
    print('读入停用词文件: ', ','.join(file_paths))
    stop_words = list()
    for file_path in file_paths:
        with open(file_path, 'r', errors='ignore', encoding='utf-8') as f:
            words = f.read().splitlines()
            stop_words.extend(words)
    return stop_words


def get_tf_idf_mat(train_data, train_label, stop_word_list_):
    class_list = np.unique(train_label)
    print('共有商品类别 %d 个，分别为: %s' % (len(class_list), ' '.join(class_list)))
    train_text_dict = dict()
    for product_class in class_list:
        train_text_dict[product_class] = ''
    assert (len(train_label) == len(train_data)), 'the length of train_label and train_data should be the same'
    for num in range(len(train_label)):
        train_text_dict[train_label[num]] = train_text_dict[train_label[num]] + train_data[num]
    train = []
    for each_train_text in train_text_dict:
        train.append(train_text_dict[each_train_text])
    print('统计词频，计算TF-IDF值')
    tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_word_list_, max_df=0.8)        # 该类会将文本中的词语转换为词频矩阵，再统计每个词语的tf-idf权值
    cipin_tf_idf = tf_idf_vectorizer.fit_transform(train)
    # print('词频统计表为：', tf_idf_vectorizer.vocabulary_)
    # train a svm model
    print('训练分类svm模型')
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    train_cipin_arr = tf_idf_vectorizer.transform(train_data)
    clf_model = model.fit(train_cipin_arr, train_label)
    return class_list, tf_idf_vectorizer, clf_model


def classify(product_to_classify, tf_idf_vectorizer, clf_model):
    sentence_in = [' '.join(jieba.cut(product_to_classify[1]))]
    vec_in = tf_idf_vectorizer.transform(sentence_in)
    prd = clf_model.predict(vec_in)
    return prd[0]


if __name__ == '__main__':

    prod_category, prod_name, prods = seg_text(filename_data)
    stop_word_list = get_stop_word(filename_stopwords)
    c_list, tf_idf_vec, clf = get_tf_idf_mat(prod_name, prod_category, stop_word_list)

    # initial categories corrections
    corrects = 0
    correct_dict = dict()
    for c in c_list:
        correct_dict[c] = {'correct': 0, 'wrong': 0}

    # classify every product using keywords
    products_classify = []
    progress_bar = pyprind.ProgBar(len(prods), title='正在进行商品分类...')
    for product in prods:
        category = product[0]
        category_classify = classify(product, tf_idf_vec, clf)
        products_classify.append([product[0], product[1], category_classify])
        if category == category_classify:
            corrects += 1
            correct_dict[category]['correct'] += 1
        else:
            correct_dict[category]['wrong'] += 1
        progress_bar.update()

    accuracy = corrects/len(prods)
    print('平均预测准确率： %.2f%%' % (accuracy*100))

    print('各类别预测准确率：')
    for c in correct_dict:
        accuracy = correct_dict[c]['correct']/(correct_dict[c]['correct'] + correct_dict[c]['wrong'])
        print('%s ： %.2f%%' % (c, accuracy * 100))
