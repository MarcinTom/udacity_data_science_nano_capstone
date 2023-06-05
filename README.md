# Udacity Data Science Nanodegree Capstone Project

This repository has all the code and report for my Udacity Data Scientist Nanodegree Capstone project.

## Starbucks Capstone Challenge: Using Starbucks app user data to predict effective offers

### 1. Installations
This project was written in Python, using Jupyter Notebook on Anaconda. The relevant Python packages for this project are as follows:

- pandas
- numpy
- math
- json
- matplotlib
- seabron
- time
- datetime
- sklearn.model_selection (train_test_split, GridSearchCV)
- sklearn.preprocessing (StandardScaler)
- sklearn.tree (DecisionTreeClassifier)
- sklearn.ensemble (RandomForestClassifier)
- sklearn.metrics (classification_report, f1_score)

### 2. Project Motivation
This project is the Capstone project of my Data Scientist nanodegree with Udacity. As students in the nanodegree, we have the option to take part in the Starbucks Capstone Challenge.

For the challenge, Udacity provided simulated data that mimics customer behavior on the Starbucks rewards mobile app.

In this project, I use the data to answer 2 business questions:

  - What are the main drivers of an effective offer on the Starbucks app?
  - Out of the compared models which one is better in predicting the 

To answer the above 2 questions, I created 2 models for the data on the 3 offer types provided. The three offers are: Buy One Get One Free (BOGO), Discount (discount with purchase), and Informational (provides information about products).

As a brief summary of my findings:
- For Question 1, the feature importance given by all 2 models were that the membership time is the biggest predictor of the effectiveness of an offer.

- For Question 2, the comparison of baseline Decision Tree models to Random Forests ones showed that in all 3 types of offers the Random Forest has better results.

### 3. File Descriptions
This repository contains data files plus the solution notebook. 
Additional files were added to the repo with the description of the project.

There is also a blog article file available [Blog Article](https://github.com/MarcinTom/udacity_data_science_nano_capstone/blob/96c6458a8a4406aa3fab63d18a29d2aa6fc3f124/Blog%20article.md)

### 4. Licensing, Authors, Acknowledgements, etc.

Data for coding project was provided by Udacity.
