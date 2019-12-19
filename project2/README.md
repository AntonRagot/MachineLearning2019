# Machine Learning project 2: Twitter Sentiment Analysis
_Robin Zbinden, Anton Ragot, Peter Krcmar (team RAP)_

In this project, we aim to obtain a classifier that predicts if a given tweet message contains a positive :) or negative :( smiley, by considering only the remaining part of the tweet.

## Folder Structure

```
├───code          (python scripts)
├───data          (train and test sets)
│    └───out      (clean train and test sets)
├───model         (pretrained models)
└───out           (predictions)

```

## Dependencies

To be able to run `run.py`, you need the following dependencies:

 - [NLTK](https://www.nltk.org/) In order to have the lemmatizer for preprocessing.
 
 First, you need to make sure you have `nltk` installed:
 ```
 pip3 install nltk
 ```
 Then, you need to download the `WordNetLemmatizer` we use. Download it using:
 ```
 python3 code/initialize.py
 ```
  
 - [Symspellpy](https://pypi.org/project/symspellpy/), used for preprocessing.
 ```
 pip3 install -U symspellpy
 ```
 
 - [Tensorflow](https://www.tensorflow.org/), used for RNN model.
 ```
 pip3 install tensorflow
 ```
 
- [Keras](https://keras.io/), used for RNN model.
```
pip3 install keras
```

- [Sckit](https://scikit-learn.org/stable/index.html), used for other models.
```
pip install -U scikit-learn
```

## Reproduce our score

### Required files

To be able to reproduced our score, you first need to place the data files in the correct place. You can download the dataset on the ML course [repository](https://github.com/epfml/ML_course/tree/master/projects/project2/project_text_classification/Datasets).

The dataset needs to be placed inside the `data` folder, unzipped. You should have the following files: `train_pos_full.txt`, `train_neg_full.txt` and `test_data.txt`.

You also need the weight of our best model. Our weights are located in the `models` folder, you should only check that they are at the correct place with the correct name:
- Classifier's weights : `model/pretrained_model_weights.hdf5`

### Start predicting

Run the script `run.py` located in the `code` folder with: 

```python run.py```

The generated predictions are saved to `output/predictions.csv`.


## Test yourself!

Feel free to run the file `human_classifier.py` to try classifying random tweets to positive or negative. Compare your accuracy to ours and the one from our model. You will see that it is not a trivial task.
