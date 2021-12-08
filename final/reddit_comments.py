import praw
import pprint
import datetime

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
    for submission in reddit.subreddit(subreddit).new():
        if len(list_of_comments) == limit:
            return list_of_comments
        else:
            if submission.created_utc < 1577817000:
                submission.comments.replace_more();
                for comment in submission.comments.list():
                    list_of_comments.append({"id": comment.id, "body": comment.body, "author": comment.author, "timestamp": comment.created_utc, "time": datetime.datetime.fromtimestamp(comment.created_utc), "link": comment.permalink})
            else:
                continue
    return list_of_comments

comments = []
limit = int(input("Limit: "))
if limit == 0:
    limit = None
for subreddit in ["uspolitics", "ultimateuspolitics", "american_politics", "americanpolitics"]: #"usapolitics",
    before = len(comments)
    comments = comments + get_comments(subreddit, limit//4)
    print(subreddit +": " + str(len(comments) - before))
    

pprint.pprint((comments))

# sub = input("Subreddit: ")
# limit = int(input("Limit: "))
# if limit == 0:
#     limit = None

# comments = get_comments(sub, limit)
# pprint.pprint(len(comments))