## Project Overview

This is a blog  summary to present my finding for the Starbuck's Capstone Project of the Udacity Data Science Nanodegree.  

All the details of the project are available here [GitHub repository](https://github.com/MarcinTom/udacity_data_science_nano_capstone.git) 
The code and the analysis is available here [Jupyter Notebook](https://github.com/MarcinTom/udacity_data_science_nano_capstone/blob/8a0cdd28e4f1b6d1468952b7f40b1af9e618b81c/Starbucks_Capstone_notebook.ipynb)

## Problem Statement
During the project I focused on answering the below two questions:
1. What are the main drivers of an effective offer on the Starbucks app?
2. Out of the compared models (Decision Tree and Random Forest) which one is better in predicting the correct classifications

## Data
The dataset is contained in three files:
1. portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
- id (string) - offer id
- offer_type (string) - type of offer ie BOGO, discount, informational
- difficulty (int) - minimum required spend to complete an offer
- reward (int) - reward given for completing an offer
- duration (int) - time for offer to be open, in days
- channels (list of strings)

2. profile.json - demographic data for each customer
- age (int) - age of the customer
- became_member_on (int) - date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

3. transcript.json - records for transactions, offers received, offers viewed, and offers completed.
- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since start of test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record

## Data exploration and analysis
I start my analysis with the portfolio dataframe. It has no missing values. Based on id column we can confirm that there are 10 unique offers in dataframe. 

The 3 types of offers are:
BOGO - buy one get one free
Discount - discount with purchase
Informational - provides information about products

![Portfolio table](./project_images/portfolio_head.JPG)

I decide to convert channels column list to a separate columns with bool encoding


## Data preprocessing


## Data modelling



### Project conclusions

The goal of the project was to predict customer positive response on the Starbucks offer based on all available information. I created a simple classification model using Decision Tree classifier. Then compared it to the results of the Random Forest performance. The latter was better in all all 3 types of offers. I also verified the feature importance to check which variables do influence the target variable in most significant way.

The most relevant factors for offer success based on the model are:
1. Membership time
2. Income
3. Age


### Potential future improvement:
- Explore other classification models types
- Apply hyperparameter tuning
- Add additional variables with feature engineering