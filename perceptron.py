import time
# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights == weights;
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    start_time = time.time()
    self.features = trainingData[0].keys() # could be useful later

    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      for i in range(len(trainingData)):
          "*** YOUR CODE HERE ***"

          # This would take each datum and iterate over all legal labels and find the best match/guess
          bestScore = None
          bestGuess = None
          datum = trainingData[i]
          for y in self.legalLabels:
              score = datum * self.weights[y]
              if score > bestScore or bestScore is None:
                  bestScore = score
                  bestGuess = y

          # if the guess is wrong,we would then modify the weight matrix
          actualLabel = trainingLabels[i]
          if bestGuess != actualLabel:
              self.weights[actualLabel] = self.weights[actualLabel] + datum
              self.weights[bestGuess] = self.weights[bestGuess] - datum
    print("--- %s seconds ---" % (time.time() - start_time))
  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    
    "*** YOUR CODE HERE ***"
    weights = self.weights[label]
    featuresWeights = weights.sortedKeys()[0:100]
        
    return featuresWeights

