# qques idées

## preprocessing

- enlever tous les duplicats
- enlever ponctuation `-`, `)`, `!`, ...
- enlever mots sans signification (stopwords) `a`, `by`, `the`, ...
- enlever chiffres (?)
- enlever tokens `<url>`, `<user>`, ...
- tout mettre en minuscules
- [lemmatizer](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python) les mots
- modifier les mots qui ont des lettres qui se répètent (e.g., bonjourrrrr)
- garder les hashtags/transformer les hashtags
- garder mots seuls mais aussi paires voire triplets de mots (pour naive bayes)

## features

- bag of words + enlever tous les mots qui apparaissent qu'une fois
- number of positive / negative occurences in predefined [lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)
- ratio of capital letters
- New binary features (1 and -1): has_exclamation_mark, has_question_mark, has_number, has_repeating_letters, has_repeated_exclamation_mark, has_hashtags, 

## models

- Naive Bayes Classifier (bcp utilisé pour text p.ex. spam dans email)
- RNN LSTM (short term memory, peut deviner le contexte / lien entre les mots d'une phrase)
- Decision trees

## libraries

- [nltk](https://www.nltk.org/) (natural language)
- [pandas](https://pandas.pydata.org/) (loading / analyzing data)
- [scikit-learn](https://scikit-learn.org/stable/) (nice ML functions)
- [tensorflow](https://www.tensorflow.org/) (RNN LSTM)
- [keras](https://keras.io/) (RNN LSTM)
