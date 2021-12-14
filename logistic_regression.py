import numpy as np
import sys
class SentimentAnalysisLogisticRegression:
    """
    Sentiment Analysis using logistic regression
    """
    count_label_to_word_count: dict[str, dict[str, int]]

    def __init__(self, alpha = 0.01, num_iters = 1000):
        self.count_label_to_word_count = {}
        self.alpha = alpha
        self.num_iters = num_iters
        self.weights = np.zeros(shape=(1, 2))
    
    def count_words_in_class(self, train_data: list[tuple[str, str]]):
        """Counts the number of times each word occurs in the given reviews for given sentiment class
        Preconditions:
            - sentiment_class is a valid sentiment class
            - strings in review are stripped off whitespaces
        """
        words_dict = {}
        classes = list(zip(*train_data))[1]
        for sentiment in set(classes):
            word_counts = {}
            for data in train_data:
                if data[1] == sentiment:
                    words = data[0].strip().split()
                    for word in words:
                        word_counts[word] = word_counts.get(word, 0) + 1
            words_dict[sentiment] = word_counts
        self.count_label_to_word_count = words_dict

    def process_to_matrix(self, train_data: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Process the given data to a matrix of weights
        """
        X = np.zeros(shape=(len(train_data), len(self.count_label_to_word_count)))
        y = np.zeros(shape=(len(train_data), 1))
        sorted_classes = sorted(self.count_label_to_word_count.keys()) #assuming that they are integers or strings of integers
        for i, data in enumerate(train_data):
            x = np.zeros(shape=(1, len(self.count_label_to_word_count)))
            for word in data[0].strip().split():
                for j, sentiment in enumerate(sorted_classes):
                        x[0,j] = self.count_label_to_word_count[sentiment].get(word, 0)
            y[i] = sorted_classes.index(data[1])
            X[i] = x
        return X, y


    def gradient_descent(self, X, y, num_iters, learning_rate):
        """
        Perform gradient descent
        """
        count = 0
        for _ in range(num_iters):
            z = np.dot(self.weights, X.T)#np.dot(X, self.weights)
            y_pred = 1/(1+np.exp(-z))
            cost =  np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))/(-X.shape[0])
            self.weights = self.weights - (learning_rate/X.shape[0]) * np.dot((y_pred - y), X)
            count += 1
            sys.stdout.write('\r'+"progress: " + "{:.4f}".format(round(count / num_iters, 4))+ "%")
            sys.stdout.flush()
        return cost, self.weights
    
    def normal_eqn(self, X, y):
        """
        Predict the sentiment of a given review
        """
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        return self.weights


    def predict(self, X):
        """
        Predict the sentiment of a given review
        """
        return 1/(1+np.exp(-np.dot(X, self.weights)))
