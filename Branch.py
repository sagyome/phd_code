import numpy as np
from operator import mul
from operator import mul
from functools import reduce
class Branch:
    def __init__(self,feature_names,label_names,label_probas=None,number_of_samples=None):
        """Branch inatance can be initialized in 2 ways. One option is to initialize an empty branch
        (only with a global number of features and number of class labels) and gradually add
        conditions - this option is relevant for the merge implementation.
        Second option is to get the number of samples in branch and the labels
        probability vector - relevant for creating a branch out of an existing tree leaf.
        """
        self.label_names=label_names
        self.number_of_features=len(feature_names)
        self.feature_names=feature_names
        self.features_upper=[np.inf]*self.number_of_features #upper bound of the feature for the given rule
        self.features_lower=[-np.inf]*self.number_of_features #lower bound of the feature for the given rule
        self.label_probas=label_probas
        self.number_of_samples=number_of_samples #save number of samples in leaf (not relevant for the current model)
    def addCondition(self, feature, threshold, bound):
        """
        This function gets feature index, its threshold for the condition and whether
        it is upper or lower bound. It updates the features thresholds for the given rule.
        """
        if bound == 'lower':
            if self.features_lower[feature] < threshold:
                self.features_lower[feature] = np.round(threshold, 2)
        else:
            if self.features_upper[feature] > threshold:
                self.features_upper[feature] = np.round(threshold, 2)
    def contradictBranch(self, other_branch):
        """
        check wether Branch b can be merged with the "self" Branch. Returns Boolean answer.
        """
        if np.sum(np.array(other_branch.features_upper) > np.array(self.features_lower)) != self.number_of_features or \
                                np.sum(np.array(other_branch.features_lower) < np.array(self.features_upper)) != self.number_of_features:
            return True
        return False
    def mergeBranch(self, other_branch):
        """
        This method gets Branch b and create a new branch which is a merge of the "self" object
        with b. As describe in the algorithm.
        """
        new_label_probas=[k+v for k,v in zip(self.label_probas,other_branch.label_probas)]
        new_number_of_samples=np.sqrt(self.number_of_samples * other_branch.number_of_samples)
        new_b = Branch(self.feature_names,self.label_names,new_label_probas,new_number_of_samples)
        new_b.features_upper, new_b.features_lower = list(self.features_upper), list(self.features_lower)
        for feature in range(self.number_of_features):
            new_b.addCondition(feature, other_branch.features_upper[feature], 'upper')
            new_b.addCondition(feature, other_branch.features_lower[feature], 'lower')
        return new_b
    def toString(self):
        """
        This function creates a string representation of the branch (only for demonstration purposes)
        """
        s = ""
        for feature, threshold in enumerate(self.features_lower):
            if threshold != (-np.inf):
                s +=  self.feature_names[feature] + ' > ' + str(np.round(threshold,3)) + ", "
        for feature, threshold in enumerate(self.features_upper):
            if threshold != np.inf:
                s +=  self.feature_names[feature] + ' <= ' + str(np.round(threshold,3)) + ", "
        s += 'labels: ['
        for k in range(len(self.label_probas)):
            s+=self.label_names[k]+' : '+str(self.label_probas[k])+' '
        s+=']'
        s+=' Number of samples: '+str(self.number_of_samples)
        return s
    def printBranch(self):
        # print the branch by using tostring()
        print(self.toString())
    def containsInstance(self, instance):
        """This function gets an ibservation as an input. It returns True if the set of rules
        that represented by the branch matches the instance and false otherwise.
        """
        if np.sum(self.features_upper >= instance)==len(instance) and np.sum(self.features_lower < instance)==len(instance):
            return True
        return False
    def getLabel(self):
        # Return the predicted label accordint to the branch
        return np.argmax(self.label_probas)
    def containsInstance(self, v):
        if np.sum(self.features_upper>=v)==len(v) and np.sum(self.features_lower<v)==len(v):
            return True
        return False
    def get_branch_records(self):
        returned_records=[]
        features={}
        for feature,value in enumerate(self.features_upper):
            if value==np.inf:
                continue
            features[str(feature)+'<='+str(value)]=1
        for feature,value in enumerate(self.features_lower):
            if value==-np.inf:
                continue
            features[str(feature)+'>'+str(value)]=1
        for proba,label_name in zip(self.label_probas,self.label_names):
            d={'label':label_name,'weight':proba,'num_of_samples':self.number_of_samples}
            d.update(features)
            returned_records.append(d)
        return returned_records
    def calculate_branch_probability_by_ecdf(self, ecdf):
        features_probabilities=[]
        for i,lower,upper in zip(range(len(ecdf.keys())),self.features_lower,self.features_upper):
            probs=ecdf[i]([lower,upper])
            features_probabilities.append(probs[1]-probs[0])
        return reduce(mul, features_probabilities, 1)

