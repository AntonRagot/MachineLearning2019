from utils import *


PATH_TWEETS_VAL = '../data/human_val.txt'
tweets = load_tweets(PATH_TWEETS_VAL)

label_tweet = []
for t in tweets:
    if '__label__1' in t:
        label_tweet.append([1,t[11:]])
    else:
        label_tweet.append([0,t[12:]])

print("Welcome to our human classifier !")
print("Can you guess if the following tweets contained positive or negative smileys ? \n")
print("If you do not understand the sentence below, press 0.")
shift = int(input("Do you want to shift the indices of the tweets to answer by a certain number (press that certain number) or not (press 0) ? \n"))
num_tweets = int(input("How many tweets do you want to label ? \n"))

print("Label the following tweets as positive (press 1) or negative (press 0):\n")
correct = 0
for i in range(shift, shift + num_tweets):
    answer = int(input(label_tweet[i][1]))
    if answer == int(label_tweet[i][0]):
        print("Correct ! \n")
        correct += 1
    else:
        print("Wrong ! \n")

    if (i - shift + 1) % 10 == 0:
        print("Your intermediate score:")
        print(str(correct) + " answers correct")
        print(str(i - shift + 1 - correct) + " answers wrong")
        print("accuracy is " + str(correct / (i - shift + 1)) + "\n")

print("Your score: \n")
print(str(correct) + " answers correct")
print(str(num_tweets - correct) + " answers wrong")
print("accuracy is " + str(correct/num_tweets))