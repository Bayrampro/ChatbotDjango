import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from joblib import dump
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer  # Добавлен импорт для лемматизатора
import string
from nltk.stem.snowball import SnowballStemmer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import pymorphy2

morph = pymorphy2.MorphAnalyzer()
wordnet_lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Приведение всех слов к нижнему регистру
    tokens = word_tokenize(text)
    # Разделение текста на слова
    tokens = [token.lower() for token in tokens]

    tokens = [token for token in tokens if token not in string.punctuation]
    # Лемматизация каждого слова

    stop_words_en = set(stopwords.words('english'))

    stop_words_ru = set(stopwords.words('russian'))

    stop_words = stop_words_en.union(stop_words_ru)

    words_to_remove = ['what', 'doing', 'can']

    for word in words_to_remove:
        if word in stop_words:
            stop_words.remove(word)

    tokens = [token for token in tokens if token not in stop_words]

    tokens_ru = [morph.parse(token)[0].normal_form for token in tokens]

    tokens_en = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    # Сборка текста обратно из лемматизированных слов

    tokens = tokens_en + tokens_ru
    text = ' '.join(tokens)

    return text


# Загрузка данных
data = pandas.read_csv('../data/dataset.csv')

data.dropna(inplace=True)

# Preprocess data
data['question'] = data['question'].apply(preprocess_text)

# Initialize and train TfidfVectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(data['question'])

# Initialize and train LinearSVC classifier
model = LinearSVC()
model.fit(X_vectorized, data['answer'])

# Save trained model and vectorizer
dump(model, 'model.joblib')
dump(vectorizer, 'vectorizer.joblib')



