# Import dependencies
#_______________________________________________________________________________
# General
import numpy as np
import pandas as pd
#_______________________________________________________________________________
# Calculating likelihoods
from scipy.stats import norm
from math import log, isnan

#_______________________________________________________________________________
# Preprocessing

def preprocess(filepath):
# Open the data, assign 'n.a' to values of 9999
    with open(filepath, "r") as f:
        header = ["head_x", "shoulders_x", "elbowR_x", "wristR_x", "elbowL_x",\
                    "wristL_x", "hips_x", "kneeR_x", "footR_x", "kneeL_x", \
                    "footL_x", "head_y", "shoulders_y", "elbowR_y", "wristR_y",\
                    "elbowL_y", "wristL_y", "hips_y", "kneeR_y", "footR_y", \
                    "kneeL_y", "footL_y"]
        data = pd.read_csv(f,na_values = "9999", names = header)
    return data

#_______________________________________________________________________________
# Predicting data

# Takes a model and a filepath to a test set and returns a list of predictions
# made by the model
def predict_set(model, filepath):
    # Load the test data set and initialise our predictions space
    test = preprocess(filepath)
    predictions = []
    
    i = 0
    # For each test instance
    for instance in [test.iloc[i] for i in range(len(test))]:
        
#         print(f"\r : {i} / {len(test)}")
#         i += 1
        # Predict the result and add it to predictions
        predictions.append(model.predict(instance))
    
    return predictions

#_______________________________________________________________________________
# Evaluating Performance

# Simple accuracy measure
def evaluate(model, filepath):
    true_results = preprocess(filepath).index
    predictions = predict_set(model, filepath)
    
    num_correct = 0
    # For each prediction
    for i in range(len(predictions)):
        # If it was correct, add to tally
        if (predictions[i] == true_results[i]):
            num_correct += 1
    
    return num_correct/ len(predictions)
#_______________________________________________________________________________
# Gaussian Bayes
class Gaussian_Naive_Bayes:

    def train(self,training_filepath):
        # First we will process our data
        data = preprocess(training_filepath)
        
        # A Naive Bayes Learner will predict the class that maximises the posterior 
        # given the attributes seen, the equation we will use is
        #
        #    P(class|data) = (P(data|class) * P(class))/P(data))
        #
        # We will be ignoring P(data) since it does not assist in selecting a class
        #
        # We will construct dictionaries to fetch values for each term:
        
        
        #  ==> P(class)
        # The priors for each class are P(Class) = class_dict[class]/number of 
        # instances - ie. The proportion of each class in the data set:
        n_instances = len(data)
        priors = {}

        for _class in data.index.unique():
            priors[_class] = len(data[data.index == _class])/n_instances
        
        
        #  ==> P(data|class)
        #
        # We will also need the mean and standard deviation of
        # each attribute within each class  
        # To do so we will create a dictionary that maps each class-attribute 
        # combination to it's mean and standard deviation
        # eg. params_dict[('bridge', 'head_x')] 
        #   = (2.17716120689655, 147.45625730166142)
        params_dict = {}
        # For each class
        for _class in data.index.unique():
            params = data[data.index == _class].describe(include='all').loc[\
                                                                ['mean','std']]
            # For each attribute
            for attribute in data.columns:
                # Map (class, attribute) to (mean, sd)
                params_dict[(_class, attribute)] = (params[attribute][0], \
                                                        params[attribute][1])

        # Assign our dictionaries
        self.priors = priors
        self.params_dict = params_dict

        # Finally keep track of the column names
        self.columns = data.columns

    # Takes a single instance and returns a predicted class for it
    def predict(self, instance):
        # Initially our max probability and class values have not been assigned
        max_prob = None
        max_class = None
        
        # For each possible class
        for _class in list(self.priors):
            
            # Initialise the probability for this class with the first term,
            # log(P(class))
            prior = self.priors[_class]
            prob = log(prior)
            
            # For each attribute
            attr_index = 0
            for attr in instance:
                
                # Extract the mean and standard deviation for that attribute given the current class
                (mean, stddev) = self.params_dict[(_class, self.columns[attr_index])]
                
                # Calculate the likelihood of the data given this class
                likelihood = norm.pdf(attr, mean, stddev)
                
                # If the likelihood is valid
                if (not isnan(likelihood)) and likelihood != 0:
                    
                    # Add the log-likelihood to our ongoing sum
                    prob = prob + log(likelihood)
                
                # Move on to the next attribute
                attr_index = attr_index + 1
            
            # Once all attributes have been examined,
            
            # If this is the first class we've looked at or if this is a new maximum
            if (not max_prob) or (max_prob and (prob > max_prob)):
                # Reassign the new maximum value and class
                max_prob = prob
                max_class = _class
                
        # After looking at all possible classes, return the most likely
        return max_class
    

#_______________________________________________________________________________
# KDE
class KDE_Naive_Bayes:
    
    # In KDE, we take the list of data points for each attribute in each class
    # and store them for use when classifying new instances
    def train(self, filepath):

        # Fetch the training data
        data = preprocess(filepath)
        attr_class_dict = {}
        self.data = data
        # For each class
        for _class in self.data.index.unique():
            
            # Subset the data to that class
            subset = data[data.index == _class]

            # For each attribute
            for attr in data.columns:
                # Designate the data points for that attribute
                attr_class_dict[(_class, attr)] = subset[attr].tolist()
            
        # Add the dictionary as an attribute
        self.attr_class_dict = attr_class_dict
        
        # As with Gaussian Naive Bayes, We will also need the prior 
        # probabilities for each class
        n_instances = len(data)
        priors = {}

        for _class in data.index.unique():
            priors[_class] = len(data[data.index == _class])/n_instances
        
        self.priors = priors

        # Finally we will store the attribute names
        self.columns = data.columns
     
    # Predicts a class for a single instance
    def predict(self, instance):
        # Initially our max probability and class values have not been assigned
        max_prob = None
        max_class = None
        data = self.data
        
        # Cycle through and evaluate each possible class
        for _class in data.index.unique():
            
            # Initialise the probability for this class with the first term,
            # log(P(class))
            prior = self.priors[_class]
            prob = log(prior)
            # For each attribute
            attr_index = 0
            for attr in instance:
                
                # Fetch the data points for this attribute/class combination
                data_points = data[data.index == _class][self.columns[attr_index]]
            
                # Calculate the KDE for that attribute value and add it to our prob
                score = self._KDE(attr, data_points)
                
                # Guard against log(0)
                if score != 0:
                    prob += log(score)
                # If the prob was 0, just take an arbitrary high negative value to simulate the log(0)
                else:
                    prob -= 10000
                
                
                # Move on to the next attribute
                attr_index += 1
            
            # Once all attributes have been examined,
            # If this is the first class we've looked at or if this is a new maximum
            if (not max_prob) or (max_prob and (prob > max_prob)):
                # Reassign the new maximum value and class
                max_prob = prob
                max_class = _class
                
        print(max_class)
        # Return the predicted class
        return max_class
    
    # Takes a set of data points, an x value and optionally a bandwidth value, and
    # returns the KDE likelihood of that x value
    def _KDE(self, x, data_points, bandwidth = 5):
        N = len(data_points)
        _sum = 0
        for i in range(len(data_points)):
            # Ignore NaN values
            if (not np.isnan(data_points[i])) and (not np.isnan(x)):
                diff = x - data_points[i]
                _sum += norm.pdf(diff, 0, bandwidth)
        
        return _sum/N

#_______________________________________________________________________________
# Zero R Classifier
class Zero_R_Classifier:
    def train(self,filepath):
        data = preprocess(filepath)
        # Work out the most common class and set it as our rule
        priors = {}
        for _class in data.index.unique():
            priors[_class] = len(data[data.index == _class])/len(data)
        
        max_prior = 0
        max_prior_class = None
        for _class in list(priors):
            if priors[_class] >= max_prior:
                max_prior = priors[_class]
                max_prior_class = _class
        self.rule = max_prior_class

    def predict(self, instance):
        return self.rule
#_______________________________________________________________________________
# Random Baseline

# Will make random guesses based on the class distribution in the training set
class Random_Classifier:
    def train(self,filepath):
        data = preprocess(filepath)
        # Work out the most common class and set it as our rule
        self.class_labels = data.index.unique
        ordered_priors = []
        for _class in class_labels:
            ordered_priors.append(len(data[data.index == _class])/len(data))
        

    def predict(self, instance):
        return np.random.choice(self.class_labels, 1, self.ordered_priors)
