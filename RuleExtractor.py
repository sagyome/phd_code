from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from Branch import *
class RuleExtractor():
    def __init__(self,rf_model,feature_names,label_names):
        self.rf_model=rf_model
        self.feature_names=feature_names
        self.label_names=label_names

    def predict(self):
        branch=Branch(self.feature_names,self.label_names)

