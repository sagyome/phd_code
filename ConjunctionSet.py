from Branch import Branch
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
class ConjunctionSet():
    def __init__(self,feature_names,model):
        self.model=model
        self.feature_names=feature_names
        self.label_names=model.classes_
        self.generateBranches()
        self.buildConjunctionSet()
    def generateBranches(self):
        trees=[estimator.tree_ for estimator in self.model.estimators_]
        self.branches_lists=[self.get_tree_branches(tree_) for tree_ in trees]
    def get_tree_branches(self,tree_):
        leaf_indexes = [i for i in range(tree_.node_count) if tree_.children_left[i] == -1 and tree_.children_right[i] == -1]
        branches=[self.get_branch_from_leaf_index(tree_,leaf_index) for leaf_index in leaf_indexes]
        return branches
    def get_branch_from_leaf_index(self,tree_,leaf_index):
        sum_of_probas=np.sum(tree_.value[leaf_index][0])
        label_probas=[i/sum_of_probas for i in tree_.value[leaf_index][0]]
        new_branch=Branch(self.feature_names,self.label_names,label_probas=label_probas,
                          number_of_samples=tree_.n_node_samples[leaf_index])#initialize branch
        node_id=leaf_index
        while node_id: #iterate over all nodes in branch
            ancesor_index=np.where(tree_.children_left==node_id)[0] #assuming left is the default for efficiency purposes
            bound='upper'
            if len(ancesor_index)==0:
                bound='lower'
                ancesor_index = np.where(tree_.children_right == node_id)[0]
            new_branch.addCondition(tree_.feature[ancesor_index[0]], tree_.threshold[ancesor_index[0]], bound)
            node_id=ancesor_index[0]
        return new_branch
    def buildConjunctionSet(self):
        conjunctionSet=self.branches_lists[0]
        for i,branch_list in enumerate(self.branches_lists[1:]):
            print('Iteration '+str(i+1)+": "+str(len(conjunctionSet))+" conjunctions")
            conjunctionSet=self.merge_branch_with_conjunctionSet(branch_list,conjunctionSet)
        self.conjunctionSet=conjunctionSet
    def merge_branch_with_conjunctionSet(self,branch_list,conjunctionSet):
        new_conjunction_set=[]
        for b1 in conjunctionSet:
            new_conjunction_set.extend([b1.mergeBranch(b2) for b2 in branch_list if b1.contradictBranch(b2)==False])
        new_conjunction_set=[b for b in new_conjunction_set if b.calculate_branch_probability_by_ecdf(self.ecdf_dict)>0]
        return new_conjunction_set
    def get_conjunction_set_df(self):
        records = []
        for b in self.conjunctionSet:
            records.extend(b.get_branch_records())
        return pd.DataFrame(records).fillna(0)
    def predict(self,X):
        predictions=[]
        for inst in X:
            for conjunction in self.conjunctionSet:
                if conjunction.containsInstance(inst):
                   predictions.append(self.label_names[conjunction.getLabel()])
        return predictions
    def get_instance_branch(self,inst):
        for conjunction in self.conjunctionSet:
            if conjunction.containsInstance(inst):
                return conjunction
    def set_ecdf(self,data):
        self.ecdf_dict={indx:ECDF(data[col])for indx,col in enumerate(self.feature_names)}



