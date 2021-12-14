"""
Running a Naive Bayes Sentiment Analysis model.
Created by Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""
from sentiment_analysis_naive_bayes1 import SentimentAnalyzer
import csv
import random
import sys
import math
def obtain_training_data(filename):
    """
    Obtain the training data from the given file.
    """
    data = []
    with open(filename, 'r', encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        for line in reader:
            tweet = line[1].strip()
            label = line[0].strip()
            data.append((tweet, label))
    return data

def split_training_test(data):
    """
    Split the training data into training and test data.
    """
    random.shuffle(data)
    training_data = data[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):]
    return training_data, test_data


def evaluate_model(test_data):
    """
    Return the accuracy of the model
    Preconditions:
        - len(test_data) > 0
    """
    count = 0
    correct = 0
    for tweet, label in test_data:
        if analyzer.classify(tweet)[0] == label:
            correct += 1
        count += 1
        sys.stdout.write('\r'+"progress: " + "{:.4f}".format(round(count / len(test_data) * 100, 4)) + "%"+ " "+ "accuracy: " + "{:.4f}".format(round(correct / count * 100, 4)) + "%")
        sys.stdout.flush()
    return correct / len(test_data)

if __name__ == '__main__':
    print("initializing...")
    analyzer = SentimentAnalyzer(pretrained=True)
    # print("loading training data...")
    # training_data = obtain_training_data('data.csv')
    # training_data, test_data = split_training_test(training_data)
    # print("finished loading training data.")
    # print("training model...")
    # analyzer.train(training_data)
    # print("finished training model.")
    # analyzer.export_trained_data('exports.json')
    # print("exported data", analyzer.priors)
    # print("evaluating model...")
    # evluation_accuracy = evaluate_model(test_data)
    # print("evaluation accuracy:", evluation_accuracy)
    classified, result = analyzer.classify("I love hate hate hate hate hate hate this movie!")
    mean = (abs(result["0"]) + abs(result["1"])) / 2
    result = {"0": -abs(result["0"] + mean), "1": result["1"] - mean}
    print(math.tanh(result["0"]), math.tanh(result["1"]))

    