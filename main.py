"""
Sentiment analysis on public opinion regarding governments is conducted.
Further statistical analysis is conducted on scored sentiments from the model
and a stunning visual simulation based on sentiments is presented.
Copyright Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""
import datetime
import pandas

from stats import StatisticsCommentInfo, graph_raw, process_average_comments, filter_comments
from sentiment_analysis_naive_bayes import SentimentAnalyzer
from simulation import OpinionSimulationManager


def load_and_graph(analyzer: SentimentAnalyzer,
                   path: str, title: str, keys: tuple[str, str],
                   date_range: tuple[datetime.datetime, datetime.datetime]) -> None:
    """
    Loads the CSV file at `path`, analyzes comment text with `analyzer`
    and graphs the results in a graph named `title`.

    Only comments between `date_range[0]` and `date_range[1]` are graphed.

    The CSV file at `path` is assumed to have two columns
    named after the values of `keys[0]` and `keys[1]` respectively.

    `keys[0]` is the name of the column that contains the Unix Timestamp.
    `keys[1]` is the name of the column that contains the comment body.
    """

    # Grab relevant data from CSVs.
    data = pandas.read_csv(path)

    # Go through each comment, turn it into a data point for graphing.
    comments = [
        StatisticsCommentInfo(
            date=datetime.datetime.utcfromtimestamp(int(data[keys[0]][i])),
            sentiment=analyzer.classify(data[keys[1]][i])[2]["4"][0] * 2 - 1
        )

        for i in range(len(data))
    ]

    comments = filter_comments(comments, date_range[0], date_range[1])

    data = process_average_comments(comments)

    graph_raw(data, title)


if __name__ == '__main__':
    import python_ta.contracts
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'stats', 'sentiment_analysis_naive_bayes',
            'simulation', 'pandas', 'datetime'
        ],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    # Parameters
    analyzer_source_model = 'datasets/exports.json'
    simulation_seed_comment = 'I really like fortnite.'

    # Some of the CSV files include sentiment data 3 days before the sentiment model was pushed.
    # I'm just going to assume the sentiment data in the pandas model is faulty.

    # Set up the sentiment analyzer.
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.load_pretrained(analyzer_source_model)

    load_and_graph(sentiment_analyzer,
                   'datasets/pre-covid-us.csv',
                   'Comment Sentiment Over Time (Pre Covid US)',
                   ('timestamp', 'body'),
                   (datetime.datetime(2019, 4, 1), datetime.datetime(2021, 2, 28)))

    load_and_graph(sentiment_analyzer,
                   'datasets/post-covid-us.csv',
                   'Comment Sentiment Over Time (Post Covid US)',
                   ('created_utc', 'body'),
                   (datetime.datetime(2021, 2, 28), datetime.datetime.now()))

    simulation = OpinionSimulationManager()
    simulation.add_sim_instance(100)
    simulation.comments.append([simulation_seed_comment])
    simulation.run_simulation(sentiment_analyzer)
