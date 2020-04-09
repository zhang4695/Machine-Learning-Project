import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os
from random import shuffle
import sklearn
import sklearn.model_selection as model_selection
import nltk
from nltk.stem import WordNetLemmatizer
import contractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import heapq
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier as knn
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# ##loading packages

'''
The following 2 functions are used to load data and do train_test_split
'''
# ## 0--------ham
# ## 1--------spam
def data_mix():
    path1 = os.scandir(r'enron1\ham')
    path2 = os.scandir(r'enron1\spam')
    entries1 = []
    entries2 = []
    for x in path1:
        entries1.append(x.name)
    for x in path2:
        entries2.append(x.name)

    df = pd.DataFrame(columns=['email_name', 'content', 'type'])
    for entry, ind in zip(entries1, range(3672)):
        df = df.append({'email_name': entry, 'content': read_file(entry), 'type': 0}, ignore_index=True)
        if ind % 500 == 0:
            print(ind)
    for entry in entries2:
        df = df.append({'email_name': entry, 'content': read_file(entry, True), 'type': 1}, ignore_index=True)

    temp_ind = df.index.tolist()
    shuffle(temp_ind)
    df = df.iloc[temp_ind]  # .reset_index(drop=True)
    temp = model_selection.train_test_split(df, test_size=0.1)
    return temp[0].reset_index(drop=True), temp[1].reset_index(drop=True)

def read_file(filename, spam=False):
    if spam==False:
        file = open('enron1\\ham\\'+filename,'r',encoding='latin-1')
        content = file.read()
    else:
        file = open('enron1\\spam\\'+filename,'r',encoding='latin-1')
        content = file.read()
    return content

'''
The following 3 functions are used to do word preprocessing, bag of words and vectorize our sample
'''
def process_message(message, stem=True, no_num=True, stop_words=True, gram=2):
    message = message.lower()
    message = contractions.fix(message)
    temp1 = word_tokenize(message)

    if stop_words:
        sword = stopwords.words('english')
        temp2 = [x for x in temp1 if x not in sword]
    if stem:
        stemmer = PorterStemmer()
        temp3 = [stemmer.stem(x) for x in temp2]
    temp4 = [x for x in temp3 if x.isalpha()]
    words = [x for x in temp4 if len(x) > 2]
    return words

def bag_of_words(dataset, num_feature):
    wordfreq = {}
    for ind in range(len(dataset)):
        for x in dataset.content[ind]:
            if x not in wordfreq.keys():
                wordfreq[x] = 1
            else:
                wordfreq[x] +=1

    most_freq = heapq.nlargest(num_feature, wordfreq, key=wordfreq.get)
    return most_freq

def vectorize(dataset, most_freq):
    sentence_vec = []
    for sentence in dataset.content:
        sent_vec = []
        for token in most_freq:
            if token in sentence:
                sent_vec.append(sentence.count(token))
            else:
                sent_vec.append(0)
        sentence_vec.append(sent_vec)

    dataset.content = pd.Series(sentence_vec)
    dataset = dataset.reset_index(drop=True)
    dataset = dataset[['email_name', 'content', 'type']]
    mat = np.asarray(sentence_vec)

    return dataset, mat

'''
KNN classifier
'''
class KNNClassifier():
    def __init__(self, train, train_vec):
        self.train_result = np.array(train)[:, 2]
        self.train = train_vec
        self.scaler = sklearn.preprocessing.MinMaxScaler()
        self.scaler.fit(self.train)
        self.train = self.scaler.transform(self.train)

    def norm(self, row1, row2, L):
        return np.linalg.norm(row1 - row2, L)

    def get_neighbors(self, test_row, num_nei, L):
        distance = []
        for idx in range(len(self.train)):
            distance.append((self.train_result[idx], self.norm(test_row, self.train[idx], L)))
        distance.sort(key=lambda tup: tup[1])

        neighbors = []
        for idx in range(num_nei):
            neighbors.append(distance[idx][0])
        return max(neighbors, key=neighbors.count)

    def operate(self, test, test_vec, num_nei=4, metric=None):

        test_vec = self.scaler.fit_transform(test_vec)
        pred_res = []
        if metric == 2 or metric == None:
            L = None
        elif metric == 1:
            L = 1
        else:
            L = np.inf
        for each, ind in zip(test_vec, range(len(test_vec))):
            pred_res.append(self.get_neighbors(each, num_nei, L))
            if ind % 500 == 0:
                print('in processing --->', ind, '\n')
        return pred_res

'''
the operate function that calls our preprocessing, bag of words, vectorize functions, then do KNN testing
'''
def operate_knn(set1, set2, neighbor=5, num_feature=350, metric=2):
    train_copy = set1.copy()
    test_copy = set2.copy()

    for dataset in [train_copy, test_copy]:
        token = []
        for x in range(len(dataset)):
            token.append(process_message(dataset.content[x]))
        dataset.content = pd.Series(token)

    voc_set = bag_of_words(train_copy, num_feature)
    train, train_vec = vectorize(train_copy, voc_set)
    test, test_vec = vectorize(test_copy, voc_set)

    clf = KNNClassifier(train, train_vec)
    res = clf.operate(test, test_vec)
    knn_score = metrics.accuracy_score(test.type.tolist(), res)
    return knn_score

'''
The Naive Bayes classifier and also with its operate function in the following
'''
class NaiveBayes():
    def __init__(self, train, train_vec):
        self.train = train
        self.result = self.train.type.tolist()
        self.train_vec = train_vec
        self.y_0 = self.train[self.train.type == 0].index.tolist()
        self.y_1 = self.train[self.train.type == 1].index.tolist()
        self.y_0_prior = len(self.y_0) / len(self.result)
        self.y_1_prior = 1 - self.y_0_prior

    def predict(self, test):
        pred = []
        for row in test:
            cond_0 = 1
            cond_1 = 1
            for ind in range(len(row)):
                cond_0 *= (list(self.train_vec[self.y_0, ind]).count(row[ind]) + 1) / len(self.y_0)
                cond_1 *= (list(self.train_vec[self.y_1, ind]).count(row[ind]) + 1) / len(self.y_1)
            if cond_0 * self.y_0_prior >= cond_1 * self.y_1_prior:
                pred.append(0)
            else:
                pred.append(1)

        return pred


def operate_nb(set1, set2, num_feature=350):
    train_copy = set1.copy()
    test_copy = set2.copy()

    for dataset in [train_copy, test_copy]:
        token = []
        for x in range(len(dataset)):
            token.append(process_message(dataset.content[x]))
        dataset.content = pd.Series(token)

    voc_set = bag_of_words(train_copy, num_feature)
    train, train_vec = vectorize(train_copy, voc_set)
    test, test_vec = vectorize(test_copy, voc_set)

    clf = NaiveBayes(train, train_vec)
    res = clf.predict(test_vec)
    nb_score = metrics.accuracy_score(test.type.tolist(), res)

    return nb_score


'''
Decision Tree Classifier with Node class, and also the operate function in the following
'''
class Node():
    def __init__(self, predicted_class):
        self.left = None
        self.right = None
        self.threshold = None
        self.index = None
        self.predicted_class = predicted_class


class DecisionTreeClassifier2:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def calculate_gini(self, X, y):
        m = len(y)
        if m <= 1:
            return None, None

        email_distribution = [np.sum(y == n) for n in range(self.n_types)]
        best_gini = 1 - sum((n / m) ** 2 for n in email_distribution)
        best_idx, best_thr = None, None

        for idx in range(self.n_features):
            threshold, types = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_types
            num_right = email_distribution.copy()

            for i in range(1, m):
                email_result = types[i - 1]
                num_left[email_result] += 1
                num_right[email_result] -= 1

                gini_left = 1 - sum((num_left[x] / i) ** 2 for x in range(self.n_types))
                gini_right = 1 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_types))
                gini = (i * gini_left + (m - i) * gini_right) / m

                if threshold[i] == threshold[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (threshold[i] + threshold[i - 1]) / 2
        print(best_idx, best_thr)
        return best_idx, best_thr

    def fit(self, X, y):
        self.n_types = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        sample_per_type = [np.sum(y == n) for n in range(self.n_types)]
        predicted_type = np.argmax(sample_per_type)
        node = Node(predicted_class=predicted_type)

        if depth < self.max_depth:
            idx, thr = self.calculate_gini(X, y)
            #             print(idx, thr)
            if idx is not None:
                node.index = idx
                node.threshold = thr
                left_part_idx = X[:, idx] < thr
                X_left, y_left = X[left_part_idx], y[left_part_idx]
                X_right, y_right = X[~left_part_idx], y[~left_part_idx]
                node.left = self.grow_tree(X_left, y_left, depth + 1)
                node.right = self.grow_tree(X_right, y_right, depth + 1)
        return node

    def predict(self, X_test):
        return [self.predict_process(x) for x in X_test]

    def predict_process(self, test_row):
        node = self.tree
        while node.left:
            if test_row[node.index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


def operate_dc(set1, set2, num_feature=350, max_d=10):
    train_copy = set1.copy()
    test_copy = set2.copy()

    for dataset in [train_copy, test_copy]:
        token = []
        for x in range(len(dataset)):
            token.append(process_message(dataset.content[x]))
        dataset.content = pd.Series(token)

    voc_set = bag_of_words(train_copy, num_feature)
    train, train_vec = vectorize(train_copy, voc_set)
    test, test_vec = vectorize(test_copy, voc_set)

    X, y = train_vec, np.array(train.type.tolist())
    clf = DecisionTreeClassifier2(max_depth=max_d)
    clf.fit(X, y)
    res = clf.predict(test_vec)

    sknn_score = metrics.accuracy_score(test.type.tolist(), res)
    #     nb_confusion_mat = metrics.confusion_matrix(test.type.tolist(),skres)

    return sknn_score


'''
When we want to run our classifiers:
'''
temp1, temp2 = data_mix()
knn_res = operate_knn(temp1, temp2)
nb_res = operate_nb(temp1, temp2)
dc_res = operate_dc(temp1, temp2)



