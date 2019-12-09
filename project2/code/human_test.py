from utils import *

tweets = load_tweets('../data/ft_val.txt')

label_tweet = []
for t in tweets:
    if '__label__1' in t:
        label_tweet.append([1,t[11:]])
    else:
        label_tweet.append([0,t[12:]])

starting_index = {1:0, 2:len(label_tweet)//3, 3:(len(label_tweet)*2)//3}

name = int(input("Are you Peter (enter 1), Anton (2) or Robin (3) ? \n"))
print("If you do not understand the sentence below, press 0")
shift = int(input("Do you want to shift the tweets to answer by a certain number (press that certain number) or not (press 0) ? \n"))
num_tweets = int(input("How many tweets do you want to label ? \n"))

starting_index = starting_index[name] + shift

print("Label the following tweets as positive (1) or negative (0):\n")
correct = 0
for i in range(starting_index, starting_index + num_tweets):
    answer = int(input(label_tweet[i][1]))
    if answer==int(label_tweet[i][0]):
        print("Correct ! \n")
        correct += 1
    else:
        print("Wrong ! \n")

    if (i - starting_index + 1) % 10 == 0:
        print("Your intermediate score:")
        print(str(correct) + " answers correct")
        print(str(i - starting_index + 1 - correct) + " answers wrong")
        print("accuracy is " + str(correct / (i - starting_index + 1)) + "\n")

print("Your score: \n")
print(str(correct) + " answers correct")
print(str(num_tweets - correct) + " answers wrong")
print("accuracy is " + str(correct/num_tweets))



