import numpy as np
import sys

class NaiveBayes:
    def __init__(self):
        if not sys.warnoptions:
            import warnings
        warnings.simplefilter("ignore")
        


    def predict(self, X):
        self.maps_detail, leading_probs = [] , []
        for row in X:
            probs = {}
            for class_name, features in self.class_summary.items():
                likelihood = 0.0 #probality of feature given the class
                for i in range(len(features['summary'])):
                    feature = row[i]
                    normal_proba = self._probsGD(
                        feature, features['summary'][i]['mean'], features['summary'][i]['std']
                        )
                    likelihood += np.log(normal_proba) # + is there because of log of product of probs * => + (irs just faster and more accurate)
                prior = features['prior'] #prob of a class being there on avarege
                probs[class_name] = prior + likelihood 
                #probs[class_name] = prior * likelihood #interesting mean acuuracy drops from by roughly 20%
            self.maps_detail.append(probs)
            leading_probs.append(max(probs, key = probs.get)) #choosing the best probs
        return leading_probs
    


    #returns comparasion %(prediction/data)
    def accuracy(self, Y_test, Y_pred):
        correct = 0
        for y_test, y_pred in zip(Y_test, Y_pred):
            if y_test == y_pred:
                correct += 1 
        return correct / len(Y_test)



    #fitting the bayes model
    def fit(self, X, Y):
        sep =  self._separate_by_class(X, Y)
        self.class_summary = {}
        for class_id, feature in sep.items():
            self.class_summary[class_id] = {
                'prior'  : len(feature)/len(X),
                'summary': [i for i in self.data_summary(feature)], #here yield (generator) is used, a loop has generating object(method) and each time invokes .next() method
            }
        return self.class_summary



    #propabilities of getting x from gaussian distribution
    def _probsGD(self, x, mean, stddev):
        coef = (stddev*(np.sqrt(2*np.pi)))
        exponent = np.exp(-((x-mean)**2 / (2*stddev**2)))
        return exponent / coef

    

    #yields stats for a feature columnn 
    def data_summary(self, X): 
            for feature in zip(*X): 
                yield {
                'std' : self._std_dev(feature),
                'mean' : self._mean(feature)
                } 



    #X - features, Y - labels, returns  dictionary with classes and each feature matrix of that class
    def _separate_by_class(self, X, Y): 
        sep ={}
        for i in range(len(X)):
            class_id = Y[i]
            features = X[i]
            if class_id not in sep:
                sep[class_id] = []
            sep[class_id].append(features)
            #print(sep) #tool to view some raw data being processed
        return sep



    #Standard deviation
    @staticmethod
    def _std_dev(nums):
        N = len(nums)
        mean = sum(nums)/float(N)
        varience = sum([(x - mean)**2 for x in nums])/float(N)
        return np.sqrt(varience)

    @staticmethod
    def _mean(nums):
        return sum(nums)/float(len(nums))