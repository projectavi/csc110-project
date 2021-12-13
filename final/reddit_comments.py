"""
Reddit API Scraper to gather comments from before Covid (Pre 2020)
Created by Mishaal Kandapath, Taylor Whatley, Aviraj Newatia, and Rudraksh Monga.
"""
import datetime
import praw

reddit = praw.Reddit(
    client_id="ZTrdortwe2H2cqKIfBVb7g",
    client_secret="21wRV5suixf9FO1UtNMzLTvIFUxprQ",
    password="4theproject",
    user_agent="csc110-project by u/fakebot3",
    username="csc110-proj",
)


def get_comments_pre(subreddit: str, limit: int = None) -> list[dict[str]]:
    """
    Gathers a number (limit) of the comments from the given subreddit with timestamps before
    1st January 2020

    >>> x = (get_comments_pre('american_politics', 2))
    >>> len(x)
    2
    """
    list_of_comments = []
    for submission in reddit.subreddit(subreddit).new():
        if len(list_of_comments) == limit:
            return list_of_comments
        elif submission.created_utc < 1577817000:
            submission.comments.replace_more()
            for comment in submission.comments.list():
                utc = comment.created_utc
                list_of_comments.append({"id": comment.id, "body": comment.body,
                                         "author": comment.author,
                                         "timestamp": comment.created_utc,
                                         "time": datetime.datetime.fromtimestamp(utc),
                                         "link": comment.permalink})
        else:
            continue
    return list_of_comments


def get_comments_subreddits(subreddits: list[str], limit: int) -> list[dict[str]]:
    """
    Initialises the reddit object from praw and gathers precovid reddit comments based on an
    input of subreddits from the users

    >>> x = get_comments_subreddits(["american_politics"], 2)
    >>> len(x)
    2
    """
    comments = []
    if limit == 0:
        limit = None
    for subreddit in subreddits:
        # ["uspolitics", "ultimateuspolitics", "american_politics", "americanpolitics"]
        # before = len(comments)
        comments = comments + get_comments_pre(subreddit, limit)
        # print(subreddit + ": " + str(len(comments) - before))

    return comments


if __name__ == '__main__':
    import python_ta.contracts
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ["doctest", "python_ta", "datetime", "praw"],
        # the names (strs) of imported modules
        'allowed-io': ["get_comments_subreddits"],
        # the names (strs) of functions that call print/open/input
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
