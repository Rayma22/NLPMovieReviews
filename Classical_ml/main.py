import datasets
from datasets import load_dataset
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.base import TransformerMixin
from nltk.stem import WordNetLemmatizer
import nltk
import re
import string
from bs4 import BeautifulSoup

nltk.download("punkt")
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class LinguisticPreprocessor(TransformerMixin):
    def __init__(self, ):
        super().__init__()
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = [BeautifulSoup(x, 'html.parser').get_text() for x in X]
        X = [re.sub('[%s]' % re.escape(string.punctuation), '', x) for x in X]
        X = [re.sub(" +", " ", x) for x in X]
        X = [" ".join([self.lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)]) for text in X]
        return X

def main():
    dataset = load_dataset("rotten_tomatoes")
    imdb = load_dataset('imdb')
    combined_train_df = pd.DataFrame({
        'text': dataset['train']['text'] + imdb['train']['text'],
        'label': dataset['train']['label'] + imdb['train']['label']
    })
    combined_train_df.drop_duplicates(subset=['text'], inplace=True)
    X_train = combined_train_df['text'].tolist()
    y_train = combined_train_df['label'].tolist()
    X_test = dataset['test']['text']
    y_test = dataset['test']['label']
    pipeline = Pipeline(
        steps=[
            ("processor", LinguisticPreprocessor()),
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
            ("model", SGDClassifier(loss="log_loss", n_jobs=-1, alpha=0.000001, penalty='elasticnet'))
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1}")

if __name__ == "__main__":
    main()