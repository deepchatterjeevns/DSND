import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import string
import numpy as np
import joblib
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
import sqlalchemy as db
import string
import pickle

def load_data(database_filepath):
    """
    Load data from SQLite database and split into features and target

    Args:
    database_filepath: string. Filename for SQLite database containing cleaned message data.

    Returns:
    X: dataframe. Dataframe containing features dataset.
    y: dataframe. Dataframe containing labels dataset.
    category_names: list of strings. List containing category names.
    """
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('Messages', con=engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Tokenize function which will be used in CountVectorizer;
    Steps:
    (1) lower the case
    (2) replace url
    (3) remove punctuations
    (4) tokenize
    (5) remove stopwords
    (6) reduce words to their root form
    Output is a list of cleaned tokens

    Args:
    text: string.

    Returns:
    clean_tokens: list of strings.
    """
    text = str(text)
    detected_urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))

    fil_tokens = [w for w in tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in fil_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build model with a pipeline

    Args:
    None

    Returns:
    cv: Gridsearchcv object. Model with the optimal parameters.
    """

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=2)))
    ])
    parameters = {
              'clf__estimator__n_estimators': [50, 60, 70],
              #'clf__estimator__min_samples_split': [2, 3, 4],
              #'clf__estimator__criterion': ['entropy', 'gini']
            }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Print classification report for the model

    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """
    Pickle model to designated file

    Args:
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved

    """
    joblib.dump(model, model_filepath)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
