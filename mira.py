# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.
    
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False 
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors" 
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."    
            
        self.features = trainingData[0].keys() # this could be useful for your code later...
        
        if (self.automaticTuning):
                Cgrid = [0.002, 0.004, 0.008]
        else:
                Cgrid = [self.C]
                
        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.    Train the classifier for each value of C in Cgrid, 
        then store the weights that give the best currentAccuracy on the validationData.
        
        Use the provided self.weights[label] data structure so that 
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        bestWeightsSoFar = {}
        bestAccuracySoFar = None
        # For each C value, the following steps are iterated.
        for c in Cgrid:
            weights = self.weights.copy()
            for n in range(self.max_iterations):
                for i, datum in enumerate(trainingData):

                    # This would take each datum and iterate over all legal labels and find the best match/guess
                    bestScoreSoFar = None
                    bestGuessForLabel = None
                    for y in self.legalLabels:
                        score = datum * weights[y]
                        if score > bestScoreSoFar or bestScoreSoFar is None:
                            bestScoreSoFar = score
                            bestGuessForLabel = y
                    
                    actualLabel = trainingLabels[i]
                    if bestGuessForLabel != actualLabel:
                        # if the guess is wrong,we would then modify the weight matrix
                        # This is where MIRA varies wrt perceptron
                        # This update rule makes use of C value
                        f = datum.copy()
                        tau = min(c, ((weights[bestGuessForLabel] - weights[actualLabel]) * f + 1.0) / (2.0 * (f * f)))
                        f.divideAll(1.0 / tau)

                        # f and tau value explained
                        # if tau 0.00309985121886
                        # if f[0] value is 0, then f.divideAll(tau) would be 0
                        # if f[0] value is 1, then f.divideAll(tau) would be 0.00309985121886
                        # basically its a multiplication of datum with this tau value

                        weights[actualLabel] = weights[actualLabel] + f
                        weights[bestGuessForLabel] = weights[bestGuessForLabel] - f
            
            # We would then choose the weight matrix for that C, for which we get max accuracy
            correctGuesses = 0
            guesses = self.classify(validationData)
            for i, guess in enumerate(guesses):
                correctGuesses += (validationLabels[i] == guess and 1.0 or 0.0)
            currentAccuracy = correctGuesses / len(guesses)
            
            if currentAccuracy > bestAccuracySoFar or bestAccuracySoFar is None:
                bestAccuracySoFar = currentAccuracy
                bestWeightsSoFar = weights
        
        self.weights = bestWeightsSoFar

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.    See the project description for details.
        
        Recall that a datum is a util.counter... 
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    
    def findHighOddsFeatures(self, label1, label2):
        """
        Returns a list of the 100 features with the greatest difference in feature values
                                         w_label1 - w_label2

        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"

        return featuresOdds

