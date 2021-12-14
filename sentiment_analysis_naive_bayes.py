"""
Class Implementation of Naive Bayes Sentiment Analysis model.
Created by Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""
import math
import json


class SentimentAnalyzer:
    """
    Sentiment Analysis model object to both train on and classify datasets

    Instance Attributes:
        - priors: mapping of sentiment category to its probability of occurence
          and log of probability of occurence in training data
        - class_to_word_to_count: mapping of each sentiment category to a mapping of each word that
          occurs in training data to the number of times it appears in sample trained data
        - vocabulary: set of all words that occur in training data
    """
    vocabulary: set[str]
    priors: dict[str: tuple[float, float]]
    class_to_word_to_count: dict[str, dict[str, int]]
    sum_denom: dict[str, float]

    def __init__(self, pretrained: bool = False) -> None:
        """ Initialize a new classifier based on external training data if available """
        
        self.vocabulary = set()
        self.priors = {}
        self.class_to_word_to_count = {}
        self.sum_denom = {}

        if pretrained:
            self.load_pretrained("exports.json")

    def load_pretrained(self, filename: str) -> None:
        """Loads pretrained model data from exports.json
        Preconditions:
            - given file is a json file and exists
            - json file is formatted correctly in the format outputted in export_trained_data()
        """
        with open(filename, "r") as json_file:
            json_data = json.load(json_file)
            self.priors = json_data["priors"]
            self.class_to_word_to_count = json_data["class_word_weights"]
            self.vocabulary = json_data["vocabulary"]
            self.sum_denom = json_data["sum_denom"]

    def create_vocabulary(self, sentences: list[str]) -> None:
        """Creates vocabulary from all setences in training set
        >>> x = SentimentAnalyzer()
        >>> x.create_vocabulary(["I love this movie", "I hate this movie"])
        >>> x.vocabulary == {"I", "love", "this", "movie", "hate", "this", "movie"}
        True
        """
        for sentence in sentences:
            for word in sentence.split():
                self.vocabulary.add(word)

    def train(self, train_set_data: list[tuple[str, str]]) -> None:
        """
        Trains the model on the given training set.
        """
        classes = list(zip(*train_set_data))[1]
        self.create_vocabulary(list(list(zip(*train_set_data))[0]))
        for sentiment in set(classes):
            prior = classes.count(sentiment) / len(train_set_data)
            self.priors[sentiment] = (prior, math.log(prior))
            sentences = [sentence[0] for sentence in train_set_data if sentence[1] == sentiment]
            word_counts = count_words_in_sentence(sentences)
            self.class_to_word_to_count[sentiment] = word_counts
            denom = sum(self.class_to_word_to_count[sentiment].values()) + len(self.vocabulary)
            self.sum_denom[sentiment] = denom

    def export_trained_data(self, filename: str = "exports.json") -> None:
        """
        Exports the trained model data to a json file
        Precoditions:
            - filename is a valid json file
            - len(self.priors) > 0
            - len(self.class_to_word_to_count) > 0
            - len(self.vocabulary) > 0
            - self should already be trained
        """
        output_dict = {"priors": self.priors, "class_word_weights": self.class_to_word_to_count,
                       "vocabulary": list(self.vocabulary), "sum_denom": self.sum_denom}
        with open(filename, "w") as json_file:
            json.dump(output_dict, json_file)

    def classify(self, sentence: str) -> tuple[str, float, dict[str, tuple[float, float]]]:
        """
        Classifies the given sentence based on the trained model
        Preconditions:
            - len(self.priors) > 0
            - len(self.class_to_word_to_count) > 0
            - len(self.vocabulary) > 0
            - model should be trained
        """
        result_dict = {}
        for sentiment in self.priors:
            result_dict[sentiment] = self.priors[sentiment][1]
            for word in sentence.strip().split():
                num = self.class_to_word_to_count[sentiment].get(word, 0) + 1
                likelihood = math.log(num / self.sum_denom[sentiment])
                result_dict[sentiment] += likelihood

        max_val = max(result_dict, key=result_dict.get)
        maxi = max(result_dict.values())
        # further computations to convert the value to a probability
        sentiment_probs = {}
        sentiment_sum = sum([math.exp(result_dict[category] - maxi) for category in result_dict])
        for sentiment in self.priors:
            numerator = result_dict[sentiment]
            denominator = maxi + math.log(sentiment_sum)
            log_prob = numerator - denominator
            prob = math.exp(log_prob)
            sentiment_probs[sentiment] = (prob, log_prob)

        return max_val, maxi, sentiment_probs

# Helper Functions:


def count_words_in_sentence(sentences: list[str]) -> dict[str, int]:
    """Counts the number of times each word occurs in the given sentences for given sentiment
    class
    Preconditions:
        - sentiment_class is a valid sentiment class
        - strings in sentences are stripped off whitespaces

    >>> expected = {'I': 2, 'love': 1, 'this': 2, 'movie': 2, 'hate': 1}
    >>> count_words_in_sentence(["I love this movie", "I hate this movie"]) == expected
    True
    """
    word_counts = {}
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts


if __name__ == '__main__':
    import python_ta.contracts
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'math', 'json'
        ],
        'allowed-io': ['export_trained_data', 'load_pretrained'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
