import random
import pprint

import plotly
import plotly.graph_objects as go
import plotly.express as px
import numpy
from raceplotly.plots import barplot

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


class PopulationSentimentSimulation():
    comment: str
    comment_sentiment: float
    population: int
    population_sentiment: list[float]
    comment_responses_sentiment: list[float]

    def compute_comment_sentiment(self, comment: str):
        raise NotImplementedError

    def generate_comment_responses_sentiment(self, sentiment: float):
        raise NotImplementedError

    def shift_visualisation(self, rects, histogramSeries, i, fig):
        raise NotImplementedError

    def run_simulation(self, comment_raised: str):
        raise NotImplementedError

class OpinionSimulation(PopulationSentimentSimulation):
    comment: str
    comment_sentiment: float
    population: int
    population_sentiment: list[float]
    comment_responses_sentiment: list[float]
    past_values: list[float]
    j_values: list[int]


    def __init__(self, population: int):
        self.population = population
        self.population_sentiment = [0.0 for person in range(population)]
        print(len(self.population_sentiment))

        self.comment = ""
        self.comment_responses_sentiment = []

        self.past_values = [x for x in self.population_sentiment]
        self.j_values = [0 for i in range(self.population)]

        self.j_max = 0

    def compute_comment_sentiment(self, comment: str):
        "Use mishaals sentiment computer"
        self.comment = comment

        self.comment_sentiment = random.uniform(-50, 50) #Comment this out and set self.comment_sentiment equal to the return of the sentiment from your analyser

    def generate_comment_responses_sentiment(self, sentiment: float):

        print(sentiment)
        sentiment_impact = [[random.uniform(-abs(sentiment/2), abs(sentiment/2)) for i in range(
            (self.population * 2) // 5)]]
        FLAG_not_done = True
        already_impacted = []
        not_impacted = [x for x in range(self.population)]

        for sentiments in sentiment_impact:
            j = sentiment_impact.index(sentiments)
            #print(j)
            impacted = []
            if not_impacted == []:
                already_impacted = []
                not_impacted = [x for x in range(self.population)]
            for i in range((self.population * 2) // 5):
                if not_impacted == []:
                    already_impacted = []
                    not_impacted = [x for x in range(self.population)]
                temp = random.choice(not_impacted)
                impacted.append(temp)
                not_impacted.remove(temp)
                already_impacted.append(temp)
            if sum([abs(x) for x in sentiment_impact[j]]) <= 0.04 or j >= self.population * 10:
                if j <= self.population:
                    sentiment_impact[j] = ([random.uniform(-abs(max([abs(x) for x in [max(self.past_values), min(self.past_values)]]) / 2),
                                                           abs(max([abs(x) for x in [max(self.past_values),
                                                                                     min(self.past_values)]]) / 2)) for i in range(
                        (self.population * 2) // 5)])
                else:
                    self.j_max += j
                    return
            self.j_values += [(j+1) for i in range(self.population)]
            #print(len(self.population_sentiment))
            for i in range((self.population * 2) // 5):

                self.population_sentiment[impacted[i]] += sentiment_impact[j][i]
                sentiment_impact.append([random.uniform(-abs(self.population_sentiment[impacted[i]] / 2), abs(
                    self.population_sentiment[impacted[i]] / 2)) for i in range((self.population * 2) // 5)])

            self.past_values += self.population_sentiment
            # print(len(self.population_sentiment))
            # print(len(self.past_values))
            sentiment_impact.remove(sentiment_impact[j])



        # if sum([abs(i) for i in sentiment_impact]) <= 0.04:
        #     return
        # else:
        #     impacted = [random.randint(0, self.population-1) for i in range(4)]
        #     print(impacted)
        #     for i in range(3):
        #         self.population_sentiment[impacted[i]] = self.population_sentiment[impacted[i]] + sentiment_impact[i]
        #         self.generate_comment_responses_sentiment(self.population_sentiment[impacted[i]])
        #         #self.shift_visualisation()
        #     return


    def shift_visualisation(self, rects, histogramSeries, i, fig):
        "generate the new visualisation"
        for rect, h in zip(rects, histogramSeries[i, :]):
            rect.set_height(h)
        fig.canvas.draw()
        plt.pause(0.001)

    def run_simulation(self, comment_raised: str):
        print(self.population_sentiment)
        self.compute_comment_sentiment(comment_raised)
        self.generate_comment_responses_sentiment(self.comment_sentiment)
        print("Done")

        print(len([x for x in range(1, self.population + 1)] * (len(self.past_values) // self.population)))
        print(len(np.array(self.past_values)))
        temps = []
        for i in range(self.j_max+1):
            temps += [i] * self.population
        print(len(np.array(temps)))
        #print((self.population_sentiment))
        dict_temp = {"Population": np.array([x for x in range(1, self.population + 1)] * ((len(self.past_values) // self.population))), "Sentiment": np.array(self.past_values), "j": np.array(temps)}
        df = pd.DataFrame.from_dict(dict_temp, orient="index")
        df = df.transpose()
        print(df)
        Population = df['Population']
        Sentiment = df['Sentiment']
        Iteration = df['j']
        fig = px.bar(df, x=Population, y=Sentiment, animation_frame=Iteration, animation_group=Population, range_y=(min(self.past_values), max(self.past_values)))

        name = input("Filename to save as: ")

        plotly.offline.plot(fig, filename=name + '.html')
        #fig.show(renderer="browser")


        # my_raceplot = barplot(df, item_column='Population', value_column='Sentiments', time_column='j')
        # my_raceplot.plot(item_label='Population', value_label='Sentiment', frame_duration=(self.j_max))


class SimulationManager():
    instances: list[OpinionSimulation]
    comments: list[list[str]]
    results: list[list[list[float]]]

    def add_sim_instance(self, population: int):
        raise NotImplementedError

    def add_comments(self):
        raise NotImplementedError

    def run_simulation(self):
        raise NotImplementedError


class PopulationOpinionSimulationManager(SimulationManager):
    instances: list[OpinionSimulation]
    comments: list[list[str]]
    results: list[list[float]]

    def __init__(self):
        self.instances = []
        self.comments = []
        self.results = []

    def add_sim_instance(self, population: int):
        self.instances.append(OpinionSimulation(population))

    def add_comments(self):
        for i in range(0, len(self.instances)):
            print("Entering comments for simulation with population " + str(self.instances[i].population))
            print("Continue to enter comments to run the simulation on, type 'FIN' to end")
            FLAG_done = False
            temp_list = []
            while not FLAG_done:
                string_in = input("Comment: ")
                if string_in == "FIN":
                    FLAG_done = True
                else:
                    temp_list.append(string_in)
            self.comments.append(temp_list)

    def run_simulation(self):
        for i in range(0, len(self.instances)):
            print(i)
            for j in range(0, len(self.comments[i])):
                print(j)
                self.instances[i].run_simulation(self.comments[i][j])
            self.results.append(self.instances[i].population_sentiment)
        pprint.pprint(self.results)
