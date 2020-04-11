# Disaster Response Pipeline Project

In this project, we have given a dataset provided by Figure Eight to Udacity. This is a Disaster Response dataset on which we are required to build a classifier to classify the disaster messages for appropriate and fast action from the response team.

## Project Workflow

1. Data Cleaning and storage of cleaned data
   - Read messages.csv and categories.csv files
   - Merged both the datasets, perform cleaning of data and storing of cleaned data into a sql database for quick querying
   - Used process_data.py
     
  
  
2. Training the model for classification
   - First I load the data and then use tokenization process to covert the text into a tokens format.
   - Then I usea a machine learning pipeline to train the data.First I vectorized the input token from earlier step.Then, I have use RandomForestClassifier from Scikit-Learn Package and also use GridSearchCV algorithm to find the best parameters for the ML algorithm to perform well on the data.
   -Final step is to evaluate the data on unseen test data to check for generalization error.
 
    
3. Web-app
   - The home page of the web-app has a simple interface showing an input box and some data visualization of the dataset.
   - The user simmpy have to enter request message. The ML model will classify the message and highlighted one of 36 important categories for the response team to proceed accordingly.
   
## File Description
    |   ETL Pipeline Preparation.ipynb
    |   ML Pipeline Preparation.ipynb
    |   new.txt
    |   README.md
    |   
    +---.ipynb_checkpoints
    |       ETL Pipeline Preparation-checkpoint.ipynb
    |       ML Pipeline Preparation-checkpoint.ipynb
    |       
    +---App
    |   |   run1.py
    |   |   
    |   \---templates
    |           go.html
    |           master.html
    |           
    +---data
    |       categories.csv
    |       DisasterResponse.db
    |       messages.csv
    |       process_data.py
    |       Twitter-sentiment-self-drive-DFE.csv
    |       
    +---models
    |       train_classifier1.py
    |       
    +---Notebooks
    |       ETL Pipeline Preparation.ipynb
    |       ML Pipeline Preparation.ipynb
    |       
    \---screenshots
            Main Page.PNG
            Percentages of categories.PNG
            Search Results.PNG
Note: Did not include the classifier.pkl because of its large size.    
 
## Steb by step instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier1.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run1.py`

3. Click on the generated hyperlink in the terminal or go to                 http://127.0.0.1:3001/
   
     
     
![screenshots](screenshots/Main%20Page.png)

![screenshots](screenshots/Percentages%20of%20categories.png)

## Examples

  - Message 1 : Stranded at home. Need food and water for a family of 6.

![screenshots](screenshots/Search%20esults.png)

  -

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
