"""
Simulate the spread of sentiment amongst a population from a comment
Copyright Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""

import random
import plotly
import plotly.express as px
import pandas as pd
import numpy as np
from sentiment_analysis_naive_bayes import SentimentAnalyzer


class PopulationSentimentSimulation:
    """
    This abstract class represents a single simulation instance for a population
    Instance Attributes:
        - comment: The comment who's effect will be simulated
        - comment_sentiment: The sentiment of the comment
        - population: The size of the population to be simulated
        - population_sentiment: The sentiments of each person in the population
    Representation Invariants:
        - self.population > 0
    """
    comment: str
    comment_sentiment: float
    population: int
    population_sentiment: list[float]

    def compute_comment_sentiment(self, comment: str, analyzer: SentimentAnalyzer) -> None:
        """
        Utilises the Sentiment Analysis model to compute the sentiment of the comment
        """
        raise NotImplementedError

    def generate_responses_sentiment(self, sentiment: float) -> None:
        """
        Simulate the impact of the comment sentiment on the sentiment of the public
        """
        raise NotImplementedError

    def calc_impact(self, sentiment: float, index: int) -> list[float]:
        """
        Calculates the sentiment impact based on the sentiment value
        """
        raise NotImplementedError

    def generate_sentiment_impact(self) -> list[float]:
        """
        Generate the sentiment impact using the past sentiment values
        """
        raise NotImplementedError

    def run_simulation(self, comment_raised: str, analyzer: SentimentAnalyzer) -> None:
        """
        Manage the simulation and run it on the comment raised and generate the visualisation
        """
        raise NotImplementedError


class OpinionSimulation(PopulationSentimentSimulation):
    """
    This class represents a single simulation instance for a population for Opinion spread
    simulation
    Instance Attributes:
        - comment: The comment who's effect will be simulated
        - comment_sentiment: The sentiment of the comment
        - population: The size of the population to be simulated
        - population_sentiment: The sentiments of each person in the population
    Representation Invariants:
        - self.population > 0
        - self.j_max > 0
    """
    comment: str
    comment_sentiment: float
    population: int
    population_sentiment: list[float]
    past_values: list[float]
    j_max: int

    def __init__(self, population: int) -> None:
        self.population = population
        self.population_sentiment = [0.0 for _ in range(population)]
        self.comment = ""
        self.comment_sentiment = 0
        self.past_values = list(self.population_sentiment)
        self.j_max = 0

    def compute_comment_sentiment(self, comment: str, analyzer: SentimentAnalyzer) -> None:
        """
        Utilises the Sentiment Analysis model to compute the sentiment of the comment
        """
        self.comment = comment
        probability_positive = analyzer.classify(self.comment)[2]["0"][0] - 0.5
        self.comment_sentiment = probability_positive * -100
        print(self.comment_sentiment)

    def generate_responses_sentiment(self, sentiment: float) -> None:
        """
        Simulate the impact of the comment sentiment on the sentiment of the public
        """
        sentiment_impact = [self.calc_impact(sentiment, 0)]
        already_impacted = []
        not_impacted = list(range(self.population))

        for sentiments in sentiment_impact:
            j = sentiment_impact.index(sentiments)
            impacted = []
            if not_impacted == []:
                already_impacted = []
                not_impacted = list(range(self.population))
            for _ in range(((self.population * 2) // 5) - j):
                if not_impacted == []:
                    already_impacted = []
                    not_impacted = list(range(self.population))
                temp = random.choice(not_impacted)
                impacted.append(temp)
                not_impacted.remove(temp)
                already_impacted.append(temp)
            if sum([abs(x) for x in sentiment_impact[j]]) <= 0.04 or j >= self.population * 3:
                if j <= self.population:
                    sentiment_impact[j] = self.generate_sentiment_impact()
                else:
                    self.j_max += j
                    return
            for i in range(((self.population * 2) // 5) - j):
                self.population_sentiment[impacted[i]] += sentiment_impact[j][i]
                sentiment_impact.append(self.calc_impact(self.population_sentiment[impacted[i]], j))

            self.past_values += self.population_sentiment
            sentiment_impact.remove(sentiment_impact[j])

    def calc_impact(self, sentiment: float, index: int) -> list[float]:
        """
        Calculates the sentiment impact based on the sentiment value
        """
        sentiment_impact = []
        if sentiment == 0:
            sentiment_impact = [random.uniform(-abs(sentiment / 2), abs(sentiment / 2))
                                for _ in range(((self.population * 2) // 5) - index)]
        elif sentiment < 0:
            sentiment_impact = [random.uniform(-abs(sentiment / 2), abs(sentiment / 4))
                                for _ in range(((self.population * 2) // 5) - index)]
        elif sentiment > 0:
            sentiment_impact = [random.uniform(-abs(sentiment / 4), abs(sentiment / 2))
                                for _ in range(((self.population * 2) // 5) - index)]
        return sentiment_impact

    def generate_sentiment_impact(self) -> list[float]:
        """
        Generate the sentiment impact using the past sentiment values
        """
        return ([random.uniform(-abs(max([abs(x) for x in [max(self.past_values),
                                                           min(self.past_values)]]) / 2),
                                abs(max([abs(x)
                                         for x in [max(self.past_values),
                                                   min(self.past_values)]]) / 2)) for _ in
                 range((self.population * 2) // 5)])

    def run_simulation(self, comment_raised: str, analyzer: SentimentAnalyzer) -> None:
        """
        Manage the simulation and run it on the comment raised and generate the visualisation
        """
        self.compute_comment_sentiment(comment_raised, analyzer)
        self.generate_responses_sentiment(self.comment_sentiment)
        temps = []
        for i in range(self.j_max + 1):
            temps += [i] * self.population
        dict_temp = {"Population": np.array(
            list(range(1, self.population + 1))
            * (len(self.past_values) // self.population)),
            "Sentiment": np.array(self.past_values), "j": np.array(temps)}
        df = pd.DataFrame.from_dict(dict_temp, orient="index")
        df = df.transpose()
        population = df['Population']
        sentiment = df['Sentiment']
        iteration = df['j']
        fig = px.bar(df, x=population, y=sentiment, animation_frame=iteration,
                     animation_group=population, range_y=(min(self.past_values),
                                                          max(self.past_values)))

        name = 'simulation' + str(id(self))  # input("Filename to save as: ")

        plotly.offline.plot(fig, filename=name + '.html')


class SimulationManager:
    """
    This abstract class represents a single simulation manager for multiple instances
    Instance Attributes:
        - instances: Stores the instances of the simulations
        - comments: Stores the lists of comments for each simulation instance
        - results: Stores the results of the population sentiment returns
    """
    instances: list[OpinionSimulation]
    comments: list[list[str]]
    results: list[list[float]]

    def add_sim_instance(self, population: int) -> None:
        """
        Adds a simulation instance to the list of simulations managed by the class
        """
        raise NotImplementedError

    def add_comments(self) -> None:
        """
        Adds comments to simulate the effect of for each simulation instance
        """
        raise NotImplementedError

    def run_simulation(self) -> None:
        """
        Manages and runs the simulations on all the instances for all their comments in the order
        they were entered
        """
        raise NotImplementedError


class OpinionSimulationManager(SimulationManager):
    """
    This class represents a single simulation manager for multiple instances of Opinion Effect
    Simulation
    Instance Attributes:
        - instances: Stores the instances of the simulations
        - comments: Stores the lists of comments for each simulation instance
        - results: Stores the results of the population sentiment returns
    """
    instances: list[OpinionSimulation]
    comments: list[list[str]]
    results: list[list[float]]

    def __init__(self) -> None:
        self.instances = []
        self.comments = []
        self.results = []

    def add_sim_instance(self, population: int) -> None:
        """
        Adds a simulation instance to the list of simulations managed by the class
        """
        self.instances.append(OpinionSimulation(population))

    def add_comments(self) -> None:
        """
        Adds comments to simulate the effect of for each simulation instance
        """
        for i in range(0, len(self.instances)):
            print("Entering comments for simulation with "
                  "population " + str(self.instances[i].population))
            print("Continue to enter comments to run the simulation on, type 'FIN' to end")
            flag_done = False
            temp_list = []
            while not flag_done:
                string_in = input("Comment: ")
                if string_in == "FIN":
                    flag_done = True
                else:
                    temp_list.append(string_in)
            self.comments.append(temp_list)

    def run_simulation(self) -> None:
        """
        Manages and runs the simulations on all the instances for all their comments in the order
        they were entered
        """
        for i in range(0, len(self.instances)):
            print(i)
            for j in range(0, len(self.comments[i])):
                print(j)
                self.instances[i].run_simulation(self.comments[i][j])
            self.results.append(self.instances[i].population_sentiment)


if __name__ == '__main__':
    import python_ta.contracts

    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'extra-imports': ["doctest", "python_ta", "random", "plotly",
                          "plotly.express", "pandas", "numpy", "sentiment_analysis_naive_bayes"],
        # the names (strs) of imported modules
        'allowed-io': ["add_comments", "run_simulation", "compute_comment_sentiment"],
        # the names (strs) of functions that call print/open/input
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
