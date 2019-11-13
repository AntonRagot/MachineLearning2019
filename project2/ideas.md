# qques idées

## preprocessing

- enlever tous les duplicats
- enlever ponctuation `-`, `)`, `!`, ...
- enlever mots sans signification (stopwords) `a`, `by`, `the`, ...
- enlever chiffres (?)
- enlever tokens `<url>`, `<user>`, ...
- [lemmatizer](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python) les mots
- garder les hashtag
- garder mots seuls mais aussi paires voire triplets de mots (pour naive bayes)

## models

- Naive Bayes Classifier (bcp utilisé pour text p.ex. spam dans email)
- RNN LSTM (short term memory, peut deviner le contexte / lien entre les mots d'une phrase)


## libraries

- [nltk](https://www.nltk.org/) (natural language)
- [pandas](https://pandas.pydata.org/) (loading / analyzing data)
- [scikit-learn](https://scikit-learn.org/stable/) (nice ML functions)
- [tensorflow](https://www.tensorflow.org/) (RNN LSTM)
- [keras](https://keras.io/) (RNN LSTM)
