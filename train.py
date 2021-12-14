"""
Running a Naive Bayes Sentiment Analysis model.
Copyright Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""

import csv
import sys
import random

from sentiment_analysis_naive_bayes import SentimentAnalyzer


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


def split_training_test(data: list[tuple[str, str]])\
        -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Split the training data into training and test data.
    """
    random.shuffle(data)
    training_data = data[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):]
    return training_data, test_data


def evaluate_model(test_data: list[tuple[str, str]], analyzer: SentimentAnalyzer) -> float:
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
        sys.stdout.write('\rprogress: ' + "{:.4f}".format(round(count / len(test_data) * 100, 4))
                         + "% accuracy: "
                         + "{:.4f}".format(round(100 * correct / count, 4)) + "%")
        sys.stdout.flush()
    return correct / len(test_data)


if __name__ == '__main__':
    import python_ta.contracts

    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'extra-imports': [
            'csv', 'random', 'sys', 'sentiment_analysis_naive_bayes'
        ],
        'allowed-io': ['obtain_training_data'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    print("initializing...")
    sentiment_analyzer = SentimentAnalyzer()
    print("loading training data...")
    train_data = obtain_training_data('training.csv')
    train_data, test = split_training_test(train_data)
    print("finished loading training data.")
    print("training model...")
    sentiment_analyzer.train(train_data)
    print("finished training model.")
    sentiment_analyzer.export_trained_data('exports.json')
    print("exported data", sentiment_analyzer.priors)
    print(sentiment_analyzer.classify("I hate this movie"))
    print("evaluating model...")
    evluation_accuracy = evaluate_model(test, sentiment_analyzer)
    print("", end="\n")
    print("evaluation accuracy:", evluation_accuracy)
