import random
import pprint

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

    def shift_visualisation(self):
        raise NotImplementedError

    def run_simulation(self, comment_raised: str):
        raise NotImplementedError

class OpinionSimulation(PopulationSentimentSimulation):
    comment: str
    comment_sentiment: float
    population: int
    population_sentiment: list[float]
    comment_responses_sentiment: list[float]

    def __init__(self, population: int):
        self.population = population
        self.population_sentiment = [0.0 for person in range(population)]

        self.comment = ""
        self.comment_responses_sentiment = []

    def compute_comment_sentiment(self, comment: str):
        "Use mishaals sentiment computer"
        self.comment = comment

        self.comment_sentiment = random.uniform(-5, 5)

    def generate_comment_responses_sentiment(self, sentiment: float):

        print(sentiment)
        sentiment_impact = [[random.uniform(-abs(sentiment/3), abs(sentiment/3)) for i in range(4)]]
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
            for i in range(4):
                if not_impacted == []:
                    already_impacted = []
                    not_impacted = [x for x in range(self.population)]
                temp = random.choice(not_impacted)
                impacted.append(temp)
                not_impacted.remove(temp)
                already_impacted.append(temp)
            if sum([abs(x) for x in sentiment_impact[j]]) <= 0.05 or j >= self.population * 2:
                print(j)
                return
            for i in range(4):
                self.population_sentiment[impacted[i]] = self.population_sentiment[impacted[i]] + sentiment_impact[j][i]
                sentiment_impact.append([random.uniform(-abs(self.population_sentiment[impacted[i]] / 3), abs(
                    self.population_sentiment[impacted[i]] / 3)) for i in range(4)])
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


    def shift_visualisation(self):
        "generate the new visualisation"

        raise NotImplementedError

    def run_simulation(self, comment_raised: str):
        print(self.population_sentiment)
        self.compute_comment_sentiment(comment_raised)
        self.generate_comment_responses_sentiment(self.comment_sentiment)
        print("Done")


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
