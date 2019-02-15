import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

corpus = ["The first time you see The Second Renaissance it may look boring.",
          "Look at it at least twice and definitely watch part 2.",
          "It will change your view of the matrix.",
          "Are the human people the ones who started the war?",
          "Is AI a bad thing ?"]

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


vect = CountVectorizer(tokenizer=tokenize)
x = vect.fit_transform(corpus)
print(x)

x.toarray()

print(vect.vocabulary_)

transformer = TfidfTransformer(smooth_idf=False)

tfidf = transformer.fit_transform(x)
print(tfidf.toarray())