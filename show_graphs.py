"""
ABC
"""

from final.stats import StatisticsCommentInfo, graph_all
from sentiment_analysis_naive_bayes import SentimentAnalyzer

import numpy
import pandas
import datetime

# Not that I will import this elsewhere...
if __name__ == '__main__':
    graph_title = 'Comment Sentiment Over Time (Filtered)'
    source_model = 'exports.json'
    source_csv = 'data/filtered_data_us.csv'

    # Some of the CSV files include sentiment data 3 days before the sentiment model was pushed.
    # I'm just going to assume the sentiment data in the pandas model is faulty.

    # Set up the sentiment analyzer.
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.load_pretrained(source_model)

    # I'm not entirely sure where to grab the data from...
    data = pandas.read_csv(source_csv)

    comments = [
        StatisticsCommentInfo(
            date=datetime.datetime.utcfromtimestamp(data['created_utc'][i]),
            sentiment=data['sentiment'][i]
        )

        for i in range(len(data))
    ]

    graph_all(comments, graph_title)
