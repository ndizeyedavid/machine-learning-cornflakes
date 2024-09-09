from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import string

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# stop words language we are using
stop_words = set(stopwords.words('english'))

text = "write an essay describing my school is better than there's!"

# to lowercase
text = text.lower()

# remove punctuation
text_no_punctuation = text.translate(str.maketrans('', '', string.punctuation))


# tokenise the text
tokens = word_tokenize(text_no_punctuation)

# remove stop words
filtered_tokens = [word for word in tokens if word not in stop_words]

# return words to their original form
stemmed_words = [ps.stem(word) for word in filtered_tokens]

# remove any adjectives conversion
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in stemmed_words]


# print(lemmatized_words)


corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())