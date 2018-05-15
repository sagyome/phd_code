from Branch import Branch
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

class ConjunctionSet():
    def __init__(self,feature_names,original_data,model,amount_of_branches_threshold,filter_approach):
        self.amount_of_branches_threshold=amount_of_branches_threshold
        self.original_data=original_data
        self.model=model
        self.feature_names=feature_names
        self.label_names=model.classes_
        self.filter_approach=filter_approach
        self.set_ecdf(original_data)
        self.generateBranches()
        self.number_of_branches_per_iteration = []
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
        excluded_branches=[]
        for i,branch_list in enumerate(self.branches_lists[1:]):
            #print('Iteration '+str(i+1)+": "+str(len(conjunctionSet))+" conjunctions")
            conjunctionSet=self.merge_branch_with_conjunctionSet(branch_list,conjunctionSet)
            print('i='+str(i))
            if i>3:
                conjunctionSet,this_iteration_exclusions=self.exclude_branches_from_cs(conjunctionSet)
                excluded_branches.extend(this_iteration_exclusions)
                print('Number of exclusions: '+str(len(excluded_branches)))
                print('Number of remained: '+str(len(conjunctionSet)))
        self.conjunctionSet=excluded_branches+conjunctionSet
    def exclude_branches_from_cs(self,cs):
        filtered_cs=[]
        excludable_brancehs=[]
        for branch in cs:
            if branch.is_excludable_branch():
                excludable_brancehs.append(branch)
            else:
                filtered_cs.append(branch)
        return filtered_cs,excludable_brancehs
    def filter_conjunction_set(self,cs):
        if len(cs)<=self.amount_of_branches_threshold:
            return cs
        if self.filter_approach=='probability':
            branches_metrics=[b.calculate_branch_probability_by_ecdf(self.ecdf_dict) for b in cs]
        elif self.filter_approach=='number_of_samples':
            branches_metrics = [b.number_of_samples for b in cs]
        elif self.filter_approach=='combination':
            branches_metrics = [b.calculate_branch_probability_by_ecdf(self.ecdf_dict)*b.number_of_samples for b in cs]
        threshold=sorted(branches_metrics,reverse=True)[self.amount_of_branches_threshold-1]
        return [b for b,metric in zip(cs,branches_metrics) if metric >= threshold]

    def merge_branch_with_conjunctionSet(self,branch_list,conjunctionSet):
        new_conjunction_set=[]
        for b1 in conjunctionSet:
            new_conjunction_set.extend([b1.mergeBranch(b2) for b2 in branch_list if b1.contradictBranch(b2)==False])
        #print('number of branches before filterring: '+str(len(new_conjunction_set)))
        new_conjunction_set=self.filter_conjunction_set(new_conjunction_set)
        #print('number of branches after filterring: ' + str(len(new_conjunction_set)))
        self.number_of_branches_per_iteration.append(len(new_conjunction_set))
        return new_conjunction_set
    def get_conjunction_set_df(self):
        return pd.DataFrame([b.get_branch_dict(self.ecdf_dict) for b in self.conjunctionSet])
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


