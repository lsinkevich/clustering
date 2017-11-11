import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import Counter
from nltk.corpus import stopwords


def features_collection(data):
    counter = Counter()
    stop_words = stopwords.words("english")
	
    for symbol in data:
        if symbol not in stop_words:
            counter[symbol] = 1
			
    return counter

	
def shuffle_collection(data):
    sh_keys = list(data.keys())
    sh_collection = collections.OrderedDict()
    random.seed(36)
    random.shuffle(sh_keys)
	
    for k in sh_keys:
        sh_collection[k] = data[k]

    return sh_collection


def train(data, features, labels):
    train_list = []
    weight1 = np.zeros(len(features))
    weight2 = np.zeros(len(features))
    avg_weight1 = []
    avg_weight2 = []
    increment_variable = 1
	
    values = collections.OrderedDict()
    values["positive"] = 1
    values["negative"] = -1

    for i in range(100):
        # for computing errors
        right_count = 0

        data = shuffle_collection(data)

        for d in data:
            count1 = 0
	        for i in b[d]:
		    count1 += weight1[i] * data[d][i]			
            count2 = 0
	        for i in b[d]:
		    count1 += weight2[i] * data[d][i]

            predict_label = 1
            if (count1 > count2):
                predict_label = -1
            y = values[labels[d]]

            # update weights
            if predict_label != y:
                if (predict_label == -1):
                    for i in data[d]:
                        weight1[i] -= data[d][i]
                    for i in data[d]:
                        weight2[i] += data[d][i]
                else: 
                    for i in data[d]:
                        weight1[i] += data[d][i]
                    for i in data[d]:
                        weight2[i] -= data[d][i]
            else:
                right_count += 1
				
        increment_variable += 1
		
        # training accuracy
        train_accuracy = right_count / len(data) * 100
        train_list.append(train_accuracy)

        # weight vectors
        avg_weight2.append(weight2)
        avg_weight1.append(weight1)

        if (train_accuracy == 100):
            avg_w1 = np.sum(avg_weight1 ,axis=0) / increment_variable
            avg_w2 = np.sum(avg_weight2, axis=0) / increment_variable
            return avg_w1, avg_w2, values

    avg_w1 = np.sum(avg_weight1 ,axis=0) / increment_variable
    avg_w2 = np.sum(avg_weight2, axis=0) / increment_variable

    # plot about the progress
    plt.plot(train_list)
    plt.title('train progress')
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.savefig('train_progress.png')
    return avg_w1, avg_w2, values


def test(data, weight1, weight2, values, labels):
    # for computing errors    
    right_count = 0
	
    negative_events = 0
    positive_events = 0
    positive_as_negative_events = 0 
    negative_as_positive_events = 0

    data = shuffle_collection(data)
    for d in data:
        count1 = 0
	    for i in b[d]:
	        count1 += weight1[i] * data[d][i]

        count2 = 0
	    for i in b[d]:
	        count1 += weight2[i] * data[d][i]

        predict_label = 1
        if (count1 > count2):
            predict_label = -1 

        y = values[labels[d]]
        if (y == predict_label):
            right_count += 1 

        # update positive_events, positive_as_negative_events and negative_as_positive_events
        if (y == predict_label and y == 1):
            positive_events += 1
        if (y == 1 and predict_label == 0):
            positive_as_negative_events += 1
        if (y == -1 and predict_label == 1):
            negative_as_positive_events += 1

    print("Testing accuracy: ", right_count / len(data) * 100)
    print("Precision: ", positive_events / (positive_events + negative_as_positive_events))
    print("Recall: ", positive_events / (positive_events + positive_as_negative_events))


if __name__ == '__main__':
    features = []
    feature_count = 0
	
    counts = Counter()	
    docs = collections.OrderedDict()
    docs_tmp = collections.OrderedDict()
    labels = collections.OrderedDict()
    hash_collection = collections.OrderedDict()

    for l in ["positive", "negative"]:
        subcatalog = os.path.join("./", l)
        for s in os.listdir(subcatalog): 
            file_path = os.path.join(subcatalog, s)
            data = open(file_path).read().lower()  
            symbols = data.split(" ")
            counter = Counter()
            counter += features_collection(symbols)

            movement = Counter()
            movement["bias"] = 1
            counter += movement
            counts += counter
			
            docs[s] = counter
            labels[s] = l
    
    for (w, c) in counts.most_common(1000000):
        hash_collection[w] = feature_count
        features.append(w)
        feature_count += 1

    for d in docs:
        dt = collections.OrderedDict()
        for c in docs[d]:
            if c in hash_collection:
                dt[hash_collection[c]] = 1
        docs_tmp[d] = dt


    # split the dataset : for training and for testing
    notrain_pos = 0
    train_pos = 0
    notrain_neg = 0
    for_training = collections.OrderedDict()
    for_testing = collections.OrderedDict()

    for d in docs_tmp:
        if (labels[d] == "positive"):
            if (notrain_pos < 1000 and train_pos < 2000):
                for_training[d] = docs_tmp[d]
                notrain_pos += 1
                train_pos += 1
            else:
                for_testing[d] = docs_tmp[d]
        else:
            if (notrain_neg < 1000 and train_pos < 2000):
                for_training[d] = docs_tmp[d]
                notrain_pos += 1
                train_pos += 1
            else:
                for_testing[d] = docs_tmp[d]

    # training + testing
    weight1, weight2, vals = train(for_training, features, labels)
    test(for_testing, weight1, weight2, vals, labels)

    # positive class
    zipped = list(zip(weight2, features))
    zipped.sort(key = lambda t: t[0], reverse=True) 
    print ("Positive class:")
    for (weight, word) in zipped[:10]:
        print ("word: {}, weight: {}".format(weight, word))
 
    # negative class
    zipped = list(zip(weight1, features))
    zipped.sort(key = lambda t: t[0], reverse=True)
    print ("Negative class:")
    for (weight, word) in zipped[:10]:
        print ("word: {}, weight: {}".format(weight, word))
