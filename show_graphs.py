"""
ABC
"""

from final.stats import StatisticsCommentInfo, graph_all, process_average_comments
from sentiment_analysis_naive_bayes import SentimentAnalyzer

import numpy
import pandas
import datetime

count = 0
total = 0


def classify(s: SentimentAnalyzer, text: str):
    global count
    count += 1

    print(f'Analyzing {count}/{total}...')

    return +1 if s.classify(text)[0] == '4' else -1


# Not that I will import this elsewhere...
if __name__ == '__main__':
    graph_title = 'Comment Sentiment Over Time (Filtered)'
    source_model = 'exports.json'
    source_csv = 'final/cleaned-data/dataset_filtered_pandas.csv'

    # Some of the CSV files include sentiment data 3 days before the sentiment model was pushed.
    # I'm just going to assume the sentiment data in the pandas model is faulty.

    # Set up the sentiment analyzer.
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.load_pretrained(source_model)

    # I'm not entirely sure where to grab the data from...
    data = pandas.read_csv(source_csv)

    total = len(data)

    comments = [
        StatisticsCommentInfo(
            date=datetime.datetime.utcfromtimestamp(data['created_utc'][i]),
            sentiment=classify(sentiment_analyzer, data['body'][i])
        )

        for i in range(len(data))
    ]

    comments = process_average_comments(comments)

    graph_all(comments, graph_title)
