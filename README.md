# Disaster Response Pipeline - Data Scientist Nanodegree Project 3
## Radhika Jangi
An ETL and ML pipeline plus web app to predict danger category from disaster messages
### 1. Libraries
The libraries I used for this project were:
- sys
- pandas
- numpy
- sqlalchemy
- nltk
- re
- pickle
- sklearn
- json
- plotly
- flask
### 2. Project Motivation
The motivation behind this project is to potentially build an app to classify real disaster messages. I processed real message data with NLP techniques and general cleaning. I then used that data to train a machine learning pipeline to build a model that can be used to classify test messages through a Flask web app.
### 3. File Descriptions
1. data:
    - disaster_categories.csv: A csv file containing the 36 different message categories
    - disaster_messages.csv: A csv file containing the disaster messages
    - process_data.py: ETL pipeline to process messages and categories into a single SQLite database
    - DisasterResponse.db: SQLite database that contains both messages and categories
2. model:
   - train_classifier.py: ML pipeline to build, train, evaluate, and save a classifer
   - classifier.pkl: Pick file of trained model
3. app:
    - run.py: Runs the Flask web app
    - templates: HTML files for web app
### 4. Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### 5. Acknowledgements
Thanks to Mrudula, my Udacity mentor, for her timely help
Thank you to the entire Udacity team

