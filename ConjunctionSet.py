from Branch import Branch
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

class ConjunctionSet():
    def __init__(self,feature_names,original_data,model,amount_of_branches_threshold,filter_approach, exclusion_starting_point=10):
        self.amount_of_branches_threshold = amount_of_branches_threshold
        self.original_data = original_data
        self.model = model
        self.feature_names = feature_names
        self.label_names = model.classes_
        self.filter_approach = filter_approach
        self.exclusion_starting_point = exclusion_starting_point
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
            if i >= self.exclusion_starting_point:
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
        if len(cs) <= self.amount_of_branches_threshold:
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
        print('number of branches before filterring: '+str(len(new_conjunction_set)))
        new_conjunction_set=self.filter_conjunction_set(new_conjunction_set)
        print('number of branches after filterring: ' + str(len(new_conjunction_set)))
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
    def group_by_label_probas(self,conjunctionSet):
        probas_hashes={}
        for i,b in enumerate(conjunctionSet):
            probas_hash = hash(tuple(b.label_probas))
            if probas_hash not in probas_hashes:
                probas_hashes[probas_hash]=[]
            probas_hashes[probas_hash].append(i)
        return probas_hashes
    def inner_merge(self, conjuncrionSet):
        probas_groups = self.group_by_label_probas(conjuncrionSet)
        new_branches=[]
        exclude_indexes = []
        print(probas_groups)
        for k in probas_groups:
            for index1 in range(len(probas_groups[k])):
                for index2 in range(index1+1,len(probas_groups[k])):
                    if probas_groups[k][index1] in exclude_indexes or probas_groups[k][index2] in exclude_indexes:
                        continue
                    print('k: '+str(k))
                    print("index1: "+str(index1))
                    print("index2: " + str(index2))
                    b1 = conjuncrionSet[probas_groups[k][index1]]
                    b2 = conjuncrionSet[probas_groups[k][index2]]
                    if b1.is_addable(b2):
                        new_branches.append(b1.add_branch(b2))
                        exclude_indexes.append(probas_groups[k][index1])
                        exclude_indexes.append(probas_groups[k][index2])
                    print("len excluded: " + str(len(exclude_indexes)))
                    print("len new branches: " + str(len(new_branches)))
                    print('----------------------')
            new_branches.extend([conjuncrionSet[indx] for indx in probas_groups[k] if indx not in exclude_indexes])
        return new_branches