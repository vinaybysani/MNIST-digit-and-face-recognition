# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    prob = util.Counter()
    for l in trainingLabels:
        prob[l] += 1 #Counting
    prob.normalize() #Normalizing


    self.P = prob

    # Initialization
    counts = {}
    totals = {}
    for feature in self.features:
        counts[feature] = {0: util.Counter(), 1: util.Counter()}
        totals[feature] = util.Counter()

    # Calculating totals and counts for each feature-label pair
    for i, datum in enumerate(trainingData):
        label = trainingLabels[i]
        for feature, value in datum.items():
            counts[feature][value][label] += 1.0
            totals[feature][label] += 1.0

    maxConditionals = {}
    maxAccuracy = None
    # Evaluate each k, and use the one that yields the best accuracy

    for k in kgrid or [0.0]:
        correct = 0
        conditionals = {}
        for feature in self.features:
            conditionals[feature] = {0: util.Counter(), 1: util.Counter()}

        for feature in self.features:
            for value in [0, 1]:
                for label in self.legalLabels:
                    conditionals[feature][value][label] = (counts[feature][value][label] + k) / (totals[feature][label] + k * 2) #Laplace smoothing

        # Calculating accuracy for k
        self.conditionals = conditionals
        guesses = self.classify(validationData)

        for i, guess in enumerate(guesses):
            if validationLabels[i]==guess:
                correct += 1.0
        accuracy = correct / len(guesses)

        # Best K tracking
        if accuracy > maxAccuracy or maxAccuracy is None:
            maxAccuracy = accuracy
            maxConditionals = conditionals
            self.k = k

    self.conditionals = maxConditionals
    print "Best K -- ", self.k

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"

    for y in self.legalLabels:
        logJoint[y] = math.log(self.P[y]) #log(P(C))
        for f in self.conditionals:
            prob = self.conditionals[f][datum[f]][y]
            logJoint[y] += math.log(prob) #log(P(D|C))
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    #Looping through the self.features and finding the conditional probability for the feature and the mentioned labels and calculating ration.
    #Picking top 100 of these ratios

    for feature in self.features:
        num = self.conditionals[feature][1][label1]
        denom = self.conditionals[feature][1][label2]
        ratio = num / denom
        featuresOdds.append((feature, ratio))

    for feature, top in sorted(featuresOdds, key=lambda t: -t[1])[:100]:
        featuresOdds = feature
    return featuresOdds
