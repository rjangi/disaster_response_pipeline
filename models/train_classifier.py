import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    Loads data from SQLite database as a dataframe
    Input:
        database_filepath: File path of database
    Output:
        X: Feature data (messages)
        y: Target variable (categories)
        category_names: List of labels for each category
    '''
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    
    # Assign feature target variables to X and y
    X = df['message']
    y = df.iloc[:, 4:]
    
    # Create category_names list from column headers
    category_names = list(df.columns[4:])
    
    return X, y, category_names


def tokenize(text):
    '''
    Normalizes, tokenizes, and lemmatizes text
    Input:
        text: message
    Output:
        clean_tokens: processed text
    '''
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize and normalize tokens 
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens 


def build_model():
    '''
    Builds an ML pipeline
    Input:
        NONE
    Output:
        cv: GridSearchCV results
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'clf__estimator__n_estimators': [10],
                  'clf__estimator__min_samples_split': [2, 3]
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model based on test data performance
    Input:
        model: trained model from training data
        X_test: Features (test)
        Y_test: Target (test)
        category_names: category labels
    Output:
        print statement with accuracy and classification report
    '''
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: evaluated model
        model_filepath: path of the output file
    Output:
        A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))


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