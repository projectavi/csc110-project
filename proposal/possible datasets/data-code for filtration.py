import pandas as pd

data = pd.read_csv('./the-reddit-covid-dataset-comments.csv', nrows = 2000000)
data = data.loc[data["subreddit.name"] == "politics"]
print(data)
data.to_csv("filtered_data.csv")