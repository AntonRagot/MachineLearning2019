# Machine Learning project 2: Twitter Sentiment Analysis
_Robin Zbinden, Anton Ragot, Peter Krcmar (team RAP)_

In this project, we aim to obtain a classifier that predicts if a given tweet message contains a positive :) or negative :( smiley, by considering only the remaining part of the tweet.

## Folder Structure

```
├───code     (python scripts)
├───data     (train and test sets)
├───models   (pretrained models)
└───out      (predictions)

```

## Dependencies

To be able to run `run.py`, you need the following dependencies:

 - [NLTK](https://www.nltk.org/) In order to have the lemmatizer for preprocessing.
 
 First, you need to make sure you have `nltk` installed:
 ```
 pip3 install nltk
 ```
 Then, you need to download the `WordNetLemmatizer` we use:
 ```
 python3 code/initialize.py
 ```
  
 - [Symspellpy](https://pypi.org/project/symspellpy/) Used for preprocessing.
 ```
 pip3 install -U symspellpy
 ```
 
 - [Fasttext](https://fasttext.cc/) Used for the embeddings.
 ```
 pip3 install fasttext
 ```
 
- [Keras](https://keras.io/) USED For the RNN model
```
pip3 install keras
```

## Reproduce our score

### Required files

To be able to reproduced our score, you first need to place the data files in the correct place. You can download the dataset on the ML course [repo](https://github.com/epfml/ML_course/tree/master/projects/project2/project_text_classification/Datasets).

The dataset needs to be placed inside the `data` folder, unzipped.

You also need our trained models. Our models are located in the `models` folder, you should only check that they are at the correct place with the correct name:
- Embedding : `models/...`
- Classifier : `models/...`

### Start predicting

Run the script `run.py` located in the `code` folder with: 

```python run.py```

The generated predictions are saved to output/predictions.csv.


## Test yourself!

Feel free to run the file `human_classifier.py` to try classifying random tweets to positive or negative. Compare your accuracy to ours and the one from our model. You will see that it is not a trivial task.
