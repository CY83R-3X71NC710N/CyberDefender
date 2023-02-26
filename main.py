
#!/usr/bin/env python
# CY83R-3X71NC710N Copyright 2023

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Defining the CyberDefender class
class CyberDefender:
    def __init__(self):
        # Initializing the class
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.accuracy = None
    
    # Function to load the data from a URL
    def load_data(self, url):
        self.data = pd.read_csv(url)
    
    # Function to preprocess the data
    def preprocess_data(self):
        # Dropping unnecessary columns
        self.data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
        # Dropping rows with missing values
        self.data.dropna(inplace=True)
        # Splitting the data into features and labels
        X = self.data.drop('label', axis=1)
        y = self.data['label']
        # Splitting the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Function to train the model
    def train_model(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
    
    # Function to make predictions
    def make_predictions(self):
        self.predictions = self.model.predict(self.X_test)
    
    # Function to calculate the accuracy of the model
    def calculate_accuracy(self):
        self.accuracy = self.model.score(self.X_test, self.y_test)
    
    # Function to plot the results
    def plot_results(self):
        plt.plot(self.predictions, label='Predicted')
        plt.plot(self.y_test.values, label='Actual')
        plt.title('Predicted vs Actual')
        plt.xlabel('Test Instances')
        plt.ylabel('Label')
        plt.legend()
        plt.show()

# Creating an instance of the CyberDefender class
cyber_defender = CyberDefender()

# Loading the data
cyber_defender.load_data('INSERT_URL_HERE')

# Preprocessing the data
cyber_defender.preprocess_data()

# Training the model
cyber_defender.train_model()

# Making predictions
cyber_defender.make_predictions()

# Calculating the accuracy
cyber_defender.calculate_accuracy()

# Plotting the results
cyber_defender.plot_results()
