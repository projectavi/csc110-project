import praw
import pprint

reddit = praw.Reddit(
    client_id="ZTrdortwe2H2cqKIfBVb7g",
    client_secret="21wRV5suixf9FO1UtNMzLTvIFUxprQ",
    password="4theproject",
    user_agent="csc110-project by u/fakebot3",
    username="csc110-proj",
)

print(reddit.user.me())

def get_comments(subreddit: str, limit: int=None) -> list[dict[str]]:
    list_of_comments = []
    for comment in reddit.subreddit(subreddit).comments(limit=limit):
        list_of_comments.append({"id": comment.id, "body": comment.body, "author": comment.author, "timestamp": comment.created_utc, "link": comment.permalink})
    return list_of_comments

sub = input("Subreddit: ")
limit = int(input("Limit: "))
if limit == 0:
    limit = None

comments = get_comments(sub, limit)
pprint.pprint(len(comments))