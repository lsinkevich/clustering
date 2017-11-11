# text_classification_nltk

Get the stopwords list
```python
def features_collection(data):
    counter = Counter()
    stop_words = stopwords.words("english")
	
    for symbol in data:
        if symbol not in stop_words:
            counter[symbol] = 1
			
    return counter
```
Split the dataset : for training and for testing
```python
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
```

Plotting the training progress
```python
    plt.plot(train_list)
    plt.title('train progress')
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.savefig('train_progress.png')
```

Report testing accuracy (based on TP, FN and FP)
```python
    print("Testing accuracy: ", right_count / len(data) * 100)
    print("Precision: ", positive_events / (positive_events + negative_as_positive_events))
    print("Recall: ", positive_events / (positive_events + positive_as_negative_events))
```

Reporting the weights for positive and negative classes

```python
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
```
