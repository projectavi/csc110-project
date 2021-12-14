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

    def __init__(self, pretrained=False):
        """ Initialize a new classifier based on external training data if available """
        if pretrained:
            self.load_pretrained("exports.json")
        else:
            self.vocabulary = set()
            self.priors = {}
            self.class_to_word_to_count = {}
            self.sum_denom = 0
    
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
            self.sum_denom = json_data["sum_denom"]

    def create_vocabulary(self, sentences):
        """Creates vocabulary from all setences in training set"""
        for sentence in sentences:
            for word in sentence.split():
                self.vocabulary.add(word)
    
    def count_words_in_class(self, sentences):
        """Counts the number of times each word occurs in the given sentences for given sentiment class
        Preconditions:
            - sentiment_class is a valid sentiment class
            - strings in sentences are stripped off whitespaces
        """
        word_counts = {}
        for sentence in sentences:
            words = sentence.split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        return word_counts

    def train(self, train_set_data):
        """
        Trains the model on the given training set. 
        """
        classes = list(zip(*train_set_data))[1]
        self.create_vocabulary(list(zip(*train_set_data))[0])
        for sentiment in set(classes):
            prior = classes.count(sentiment)/len(train_set_data)
            self.priors[sentiment] = (prior, math.log(prior))
            sentences = [sentence[0] for sentence in train_set_data if sentence[1] == sentiment]
            word_counts = self.count_words_in_class(sentences)
            self.class_to_word_to_count[sentiment] = word_counts  
        self.sum_denom = ((sum(self.class_to_word_to_count[sentiment].values()) + len(self.vocabulary)))  
    
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
        output_dict = {"priors": self.priors, "class_word_weights": self.class_to_word_to_count, "vocabulary": list(self.vocabulary), "sum_denom": self.sum_denom}
        with open(filename, "w") as jsonFile:
            json.dump(output_dict, jsonFile)

    def classify(self, sentence):
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
                likelihood = math.log((self.class_to_word_to_count[sentiment].get(word, 0) + 1)/self.sum_denom)
                result_dict[sentiment] += (likelihood)

        max_val = max(result_dict, key=result_dict.get)
        maxi = max(result_dict.values())
        #further computations to convert the value to a probability
        sentiment_probs = {}
        sentiment_sum = sum([math.exp(result_dict[sentiment] - maxi) for sentiment in result_dict])
        for sentiment in self.priors:
            numerator = result_dict[sentiment]
            denominator = maxi + math.log(sentiment_sum)
            log_prob = numerator - denominator
            prob = math.exp(log_prob)
            sentiment_probs[sentiment] = (prob, log_prob)

        return max_val, maxi, sentiment_probs