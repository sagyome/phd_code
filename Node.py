import numpy as np
import pandas as pd
from scipy.stats import entropy
EPSILON=0.000001
class Node():
    def __init__(self,mask):
        self.mask=mask
    def split(self,df):
        if np.sum(self.mask)==1:
            self.left=None
            self.right=None
            return
        self.features = [int(i.split('_')[0]) for i in df.columns if 'upper' in i]
        self.split_feature,self.split_value=self.select_split_feature(df)
        self.create_mask(df)
        if np.sum(self.right_mask)==0 or np.sum(self.left_mask)==0:
            self.left = None
            self.right = None
            return
        self.left=Node(list(np.logical_and(self.mask,np.logical_or(self.left_mask,self.both_mask))))
        self.right = Node(list(np.logical_and(self.mask,np.logical_or(self.right_mask,self.both_mask))))
        self.left.split(df)
        self.right.split(df)
    def create_mask(self,df):
        self.left_mask = [True if upper <= self.split_value  else False for upper in df[str(self.split_feature) + "_upper"]]
        self.right_mask = [True if lower >= self.split_value else False for lower in df[str(self.split_feature) + '_lower']]
        self.both_mask = [True if self.split_value < upper and self.split_value > lower else False for lower, upper in
                     zip(df[str(self.split_feature) + '_lower'], df[str(self.split_feature) + "_upper"])]

    def select_split_feature(self,df):
        feature_to_value={}
        feature_to_metric={}
        for feature in self.features:
           value,metric=self.check_feature_split_value(df,feature)
           feature_to_value[feature]=value
           feature_to_metric[feature] = metric
        feature = min(feature_to_metric, key=feature_to_metric.get)
        return feature,feature_to_value[feature]

    def check_feature_split_value(self,df,feature):
        value_to_metric={}
        values=set(list(df[self.mask][str(feature)+'_upper'])+list(df[self.mask][str(feature)+'_lower']))
        for value in values:
            left_mask=[True if upper <= value  else False for upper in df[self.mask][str(feature)+"_upper"]]
            right_mask=[True if lower>= value else False for lower in df[self.mask][str(feature)+'_lower']]
            both_mask=[True if value < upper and value> lower else False for lower,upper in zip(df[self.mask][str(feature)+'_lower'],df[self.mask][str(feature)+"_upper"])]
            if np.sum(left_mask)==0 or np.sum(right_mask)==0:
                value_to_metric[value]=np.inf
                continue
            """print(feature)
            print(value)
            print('Left: ')
            print(df[self.mask][left_mask])
            print('Right: ')
            print(df[self.mask][right_mask])
            print('Both: ')
            print(df[self.mask][both_mask])"""
            value_to_metric[value]=self.get_value_metric(df,left_mask,right_mask,both_mask)
        val=min(value_to_metric,key=value_to_metric.get)
        return val,value_to_metric[val]

    def get_value_metric(self,df,left_mask,right_mask,both_mask):
        l_df=df[self.mask][np.logical_or(left_mask,both_mask)]
        r_df=df[self.mask][np.logical_or(right_mask,both_mask)]
        l_entropy,r_entropy=self.calculate_entropy(l_df),self.calculate_entropy(r_df)
        l_prop=len(l_df)/len(df)
        r_prop=len(r_df)/len(df)
        return l_entropy*l_prop+r_entropy*r_prop

    def predict(self,inst,training_df):
        if self.left is None and self.right is None:
            return self.get_node_prediction(training_df),1
        #print(self.split_feature)
        #print(self.split_value)
        if inst[self.split_feature] <= self.split_value:
            prediction,depth = self.left.predict(inst,training_df)
            return prediction,depth+1
        else:
            prediction, depth =self.right.predict(inst,training_df)
            return prediction, depth + 1
    def node_probas(self,inst,df):
        class_probas = np.array([np.array(l) for l in df[self.mask]['probas']])
        branches_probas = np.array(df[self.mask]['branch_probability'].values).reshape(len(df[self.mask]), 1)
        class_probas= class_probas * branches_probas
        class_probas = class_probas.mean(axis=0)
        probas_sum=np.sum(class_probas)
        class_probas = [i / probas_sum for i in class_probas]
        return class_probas
    def get_node_prediction(self,training_df):
        v=training_df[self.mask]['probas'].values[0]
        v=[i/np.sum(v) for i in v]
        return np.array(v)
    def opposite_col(self,s):
        if 'upper' in s:
            return s.replace('upper','lower')
        else:
            return s.replace('lower', 'upper')
    def calculate_entropy(self,test_df):
        class_probas = np.array([np.array(l)/np.sum(l) for l in test_df['probas']])
        class_probas = class_probas.mean(axis=0)
        probas_sum = np.sum(class_probas)
        class_probas = [i / probas_sum for i in class_probas]
        return entropy(class_probas)
    def count_depth(self):
        if self.right==None:
            return 1
        return max(self.left.count_depth(),self.right.count_depth())+1
    def number_of_children(self):
        if self.right==None:
            return 1
        return 1+self.right.number_of_children()+self.left.number_of_children()