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
        - priors: mapping of each sentiment category to its probability of occurence and log of probability of occurence in training data
        - class_to_word_to_count: mapping of each sentiment category to a mapping of each word that occurs in training data to the number of times it appears in sample trained data
        - vocabulary: set of all words that occur in training data
    """

    priors: dict[str, tuple[float, float]]
    class_to_word_to_count: dict[str, dict[str, int]]
    vocabulary: set[str]

    def __init__(self, pretrained=False):
        """ Initialize a new classifier based on external training data if available """
        if pretrained:
            self.load_pretrained("exports.json")
        else:
            self.vocabulary = set()
            self.priors = {}
            self.class_to_word_to_count = {}
    
    def load_pretrained(self, filename):
        """Loads pretrained model data from exports.json
        Preconditions:
            - given file is a json file and exists
            - json file is formatted correctly in the format outputted in export_trained_data()
        """
        with open(filename, "r") as jsonFile:
            json_data = json.load(jsonFile)
            self.priors = json_data["priors"]
            self.class_to_word_to_count = json_data["class_word_weights"]
            self.vocabulary = json_data["vocabulary"]

    def create_vocabulary(self, reviews: list[str]):
        """Creates vocabulary from all reviews in training set
        
        >>> reviews = ["I love this movie", "I hate this movie"]
        >>> create_vocabulary(reviews)
        >>> self.vocabulary == {"I", "love", "this", "movie", "hate", "this", "movie"}
        True
        """
        for review in reviews:
            for word in review.split():
                self.vocabulary.add(word)
    
    def count_words_in_class(self, sentiment_class: str, reviews: list[str]) -> dict[str, int]:
        """Counts the number of times each word occurs in the given reviews for given sentiment class
        Preconditions:
            - sentiment_class is a valid sentiment class
            - strings in review are stripped off whitespaces
        """
        word_counts = {}
        for review in reviews:
            words = review.split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        return word_counts

    def train(self, train_set_data: list[tuple[str, str]]):
        """
        Trains the model on the given training set. 
        """
        classes = list(zip(*train_set_data))[1]
        self.create_vocabulary(list(zip(*train_set_data))[0])
        for sentiment in set(classes):
            prior = classes.count(sentiment)/len(train_set_data)
            self.priors[sentiment] = (prior, math.log(prior))
            reviews = [review[0] for review in train_set_data if review[1] == sentiment]
            word_counts = self.count_words_in_class(sentiment, reviews)
            self.class_to_word_to_count[sentiment] = word_counts    
    
    def export_trained_data(self, filename="exports.json"):
        """
        Exports the trained model data to a json file
        Precoditions:
            - filename is a valid json file
            - len(self.priors) > 0
            - len(self.class_to_word_to_count) > 0
            - len(self.vocabulary) > 0
            - self should already be trained
        """
        output_dict = {"priors": self.priors, "class_word_weights": self.class_to_word_to_count, "vocabulary": list(self.vocabulary)}
        with open(filename, "w") as jsonFile:
            json.dump(output_dict, jsonFile)

    def classify(self, review: str) -> tuple[str, dict[str, float]]:
        """
        Classifies the given review based on the trained model
        Preconditions:
            - len(self.priors) > 0
            - len(self.class_to_word_to_count) > 0
            - len(self.vocabulary) > 0
            - model should be trained 
        """
        result_dict = {}
        for sentiment in self.priors:
            result_dict[sentiment] = self.priors[sentiment][1]
            for word in review.strip().split():
                likelihood = (self.class_to_word_to_count[sentiment].get(word, 0) + 1)/(sum(self.class_to_word_to_count[sentiment].values()) + len(self.vocabulary))
                result_dict[sentiment] += math.log(likelihood)
        return max(result_dict, key=result_dict.get), result_dict