import praw
import matplotlib.pyplot as plt
from dataset import load_dataset
import pandas as pd
from credentials import get_cred

#top secret data
cred = get_cred()

reddit = praw.Reddit(client_id=cred[0], \
                     client_secret=cred[1], \
                     user_agent='My Example App', \
                     username=cred[2], \
                     password=cred[3])
print(reddit)

# for submission in reddit.front.hot():
#     print(submission)

orig_dataset = load_dataset('reddit_train.csv', ',')

subreddits = orig_dataset.subreddits.unique()
print(subreddits)
index = 70000

# subreddit = reddit.subreddit(subredditname)
# print(subreddit)
# top_subreddit = subreddit.top()
# print(top_subreddit)
count = 0
max = 10000
print('success')
words = []
wordCount = {}
commonWords = {'that','this','and','of','the','for','I','it','has','in',
'you','to','was','but','have','they','a','is','','be','on','are','an','or',
'at','as','do','if','your','not','can','my','their','them','they','with',
'at','about','would','like','there','You','from','get','just','more','so',
'me','more','out','up','some','will','how','one','what',"don't",'should',
'could','did','no','know','were','did',"it's",'This','he','The','we',
'all','when','had','see','his','him','who','by','her','she','our','thing','-',
'now','what','going','been','we',"I'm",'than','any','because','We','even',
'said','only','want','other','into','He','what','i','That','thought',
'think',"that's",'Is','much'}
# rows_list = []

for i, sub in enumerate(subreddits):
    subreddit = reddit.subreddit(sub)
    comment_count = 0
    rows_list = []
    for submission in subreddit.hot(limit=100):
        print(submission, f'Subreddit: { sub } ({ i + 1 }/{ len(subreddits)})')
        submission.comments.replace_more(limit=0)
        # comment_count = 0
        for top_level_comment in submission.comments:
            comment_count += 1
            # dict1 = {}
            # dict1.update({'comments': top_level_comment.body, 'subreddits': sub})
            # rows_list.append(dict1)
            orig_dataset.loc[index] = [index, top_level_comment.body, sub]
            index += 1
            # print(top_level_comment.body)
            if(count == max):
                break
            # word = ""
            # for letter in top_level_comment.body:
            #     if(letter == ' '):
            #         if(word and not word[-1].isalnum()):
            #             word = word[:-1]
            #         if not word in commonWords:
            #             words.append(word)
            #         word = ""
            #     else:
            #         word += letter
        if(count == max):
                break
    print(comment_count)
    new_rows = pd.DataFrame(rows_list)
    orig_dataset.append(new_rows)
    orig_dataset.to_csv('reddit_train_updated.csv')
for word in words:
    if word in wordCount:
        wordCount[word] += 1
    else:
        wordCount[word] = 1

sortedList = sorted(wordCount, key = wordCount.get, reverse = True)

keyWords = []
keyCount = []
amount = 0

for entry in sortedList:
    keyWords.append(entry)
    keyCount.append(wordCount[entry])
    amount += 1
    if (amount == 10):
        break

labels = keyWords
sizes = keyCount
# explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.title('Top comments for: r/' + subredditname)
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
