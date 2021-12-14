"""
Running a Naive Bayes Sentiment Analysis model.
Created by Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""
from sentiment_analysis_naive_bayes import SentimentAnalyzer
from logistic_regression import SentimentAnalysisLogisticRegression
import csv
import random
import sys
def obtain_training_data(filename: str) -> list[tuple[str, str]]:
    """
    Obtain the training data from the given file.
    """
    data = []
    with open(filename, 'r', encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        for line in reader:
            tweet = line[5].strip()
            label = line[0].strip()
            data.append((tweet, label))
    return data

def split_training_test(data: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Split the training data into training and test data.
    """
    random.shuffle(data)
    training_data = data[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):]
    return training_data, test_data


def evaluate_model(test_data: list[tuple[str, str]], analyzer: SentimentAnalysisLogisticRegression) -> float:
    """
    Return the accuracy of the model
    Preconditions:
        - len(test_data) > 0
    """
    count = 0
    correct = 0
    for tweet, label in test_data:
        if analyzer.classify(tweet) == label:
            correct += 1
        # elif float(analyzer.predict(tweet)) < 0.5 and label == '0':
        #     count += 1
        count += 1
        sys.stdout.write('\r'+"progress: " + "{:.4f}".format(round(count / len(test_data) * 100, 4)) + "%"+ " "+ "accuracy: " + "{:.4f}".format(round(100  * correct / count, 4)) + "%")
        sys.stdout.flush()
    return correct / len(test_data)

if __name__ == '__main__':
    print("initializing...")
    analyzer = SentimentAnalyzer()
    print("loading training data...")
    training_data = obtain_training_data('training.csv')
    training_data, test_data = split_training_test(training_data)
    print("finished loading training data.")
    print("training model...")
    analyzer.train(training_data)
    print("finished training model.")
    analyzer.export_trained_data('exports.json')
    print("exported data", analyzer.priors)
    print("evaluating model...")
    evluation_accuracy = evaluate_model(test_data, analyzer)
    print("evaluation accuracy:", evluation_accuracy)
    # analyzer = SentimentAnalysisLogisticRegression()
    # print(analyzer.weights)
    # print("loading training data...")
    # training_data = obtain_training_data('data.csv')
    # training_data, test_data = split_training_test(training_data)
    # print("finished loading training data.")
    # print("training model...")
    # analyzer.count_words_in_class(training_data)
    # X, labels = analyzer.process_to_matrix(training_data)
    # analyzer.normal_eqn(X, labels)#.gradient_descent(X, labels, 1000, 0.001)
    # print("finished training model.")
    # print("evaluating model...")
    # evluation_accuracy = evaluate_model(test_data, analyzer)
    # print("evaluation accuracy:", evluation_accuracy)

    