
# author: Cheng Peng

# brief : build general purpose classification framework based on Adaptive Boost method, which is able to construct classifiers using Naive Bayes model, and use classifiers to assign labels to unseen test data instances.

# -----------------------------------------------------------------------------
# import modules

from __future__ import division
from collections import defaultdict
import numpy as np
import argparse
import os
import math
import random
import copy
from NaiveBayes import *


# -----------------------------------------------------------------------------
# define a global dictionary that contains the weight for classes with format {index: [record, [w+, w-]]}

class_weight = defaultdict(list)


# -----------------------------------------------------------------------------
# define a class that implement AdaBoost methods

class AdaBoost(Bayes_Model):
    
    def __init__(self, File):
        
        super().__init__(File)
        self.weight_table = defaultdict(float)  # a dictionary that contains weights of tuples
    
    
    def GetValues(self):
        num_t = super().GetValues()
        return num_t
    
    
    def Classify(self, training_model):
        true_prediction, false_prediction, prediction_rec, tp_prediction, tn_prediction, fp_prediction, fn_prediction = super().Classify(training_model)
        
        return true_prediction, false_prediction, prediction_rec, tp_prediction, tn_prediction, fp_prediction, fn_prediction
    
    
    def WeightGenerator(self):
        # initialize a weight table so that each record has weight 1/D
        
        num_t = self.GetValues()
        self.init_weight = 1 / num_t
        
        with open(self.data_file, 'r') as Data:
            for line in Data:
                self.weight_table[line.strip()] = self.init_weight
    
        return num_t


    def ClassifierGenerator(self, file_name, directory, num, data, training_model, class_weight):
        # generate a good classifier and update the weights table
    
        # ----- sample D with replacement according to the tuple weights to obtain Di -----
        
        t_file = "{0}.txt".format(file_name)
        dst_path = os.path.join(directory, t_file)
        n_rec = 0
        candidates = list()
        
        with open(dst_path, "w") as fo:
            for tup, weight in self.weight_table.items():
                candidates.append(tup)
                if weight >= training_model.weight_table[tup]:
                    fo.write(tup + "\n")
                    n_rec += 1    # count the number of misclassified tuples that are written into the file
            while n_rec < num:
                fo.write(random.choice(candidates) + "\n")  # randomly write a tuple until the file has the same size as orginal
                n_rec += 1
    
    
        # ----- use the training set Di to derive a model, Mi; -----
        
        training_model.data_file = dst_path
        training_model.GetValues()
        true_prediction, false_prediction, classification_rec, tp_prediction, tn_prediction, fp_prediction, fn_prediction = self.Classify(training_model)
        
        
        # ----- calculate error rate of Mi -----
        
        error = 0  # initialize the error of Mi
        for rec in classification_rec:
            if rec.endswith("false"):
                error += self.weight_table[rec.rsplit('->', 1)[0].strip()]
        print(error)

        if error > 0.5:   # try to re-generate a training set and derive a new model
            self.ClassifierGenerator(file_name, directory, num, data, training_model)

        elif error != 0:   # otherwise, update the weight of correctly classified tuple in Di
            training_model.weight_table = dict()
            training_model.weight_table = copy.deepcopy(self.weight_table)   # deep copy an old table
            old_Z = sum(self.weight_table.values())   # compute the sum of old weights
            alpha = 0.5 * math.log((1 - error) / error)  # compute the weight of classifier's vote
            for rec in classification_rec:
                if rec.endswith("true"):
                    self.weight_table[rec.rsplit('->', 1)[0].strip()] *=  (error / (1 - error))
            ClassPrediction(data, training_model, alpha, class_weight)


            # ----- calculate normalized weights -----

            new_Z = sum(self.weight_table.values())   # compute the sum of old weights

            for tup, weight in self.weight_table.items():
                factor = weight * old_Z / new_Z
                self.weight_table[tup] *= factor


# ----- define a function that uses the ensemble method to classify each tuple, X in the test file -----

def ClassPrediction(test_file, model, alpha, classweight):

    lkhood_positive = defaultdict(float)  # contain the probability(attr|cls=+1)
    lkhood_negative = defaultdict(float)  # contain the probability(attr|cls=-1)
 
    with open(test_file, 'r') as testData:
        for index, tup in enumerate(testData):
            classweight[index] = [tup.strip(), [0, 0]]   # initialize a record
            instance = tup.strip().split(' ')
            lkhood_positive[str(instance[1:])] = 1.0
            lkhood_negative[str(instance[1:])] = 1.0
            for attribute_value in instance[1:]:
                attr, value = attribute_value.split(':')
                positive_match = ('+1', attr, value)
                negative_match = ('-1', attr, value)
                if positive_match in model.lkhood_positive.keys():
                    lkhood_positive[str(instance[1:])] *= model.lkhood_positive[positive_match]
                if negative_match in model.lkhood_negative.keys():
                    lkhood_negative[str(instance[1:])] *= model.lkhood_negative[negative_match]
            
            prob_p = lkhood_positive[str(instance[1:])] * model.prior_probability['+1']   # P(C='+1'|X)
            prob_n = lkhood_negative[str(instance[1:])] * model.prior_probability['-1']   # P(C='-1'|X)
            
            if prob_p > prob_n:
                classweight[index][1][0] += alpha
            elif prob_p < prob_n:
                classweight[index][1][1] += alpha


# ----- define function that make final classifications based on the largest weight -----

def classification_summary(test_weight):

    true_prediction = 0  # count the number of true predictions
    false_prediction = 0
    tp_prediction = 0  # count the number of true positive predictions
    tn_prediction = 0  # count the number of true negative predictions
    fp_prediction = 0  # count the number of false positive predictions
    fn_prediction = 0  # count the number of false negative predictions
    prediction_rec = list() # a list that records the predictions
    
    for index, tup_votes in test_weight.items():
        tup, votes = tup_votes[0], tup_votes[1]
        cls_label = tup.split(' ')[0]
        p_vote, n_vote = votes
        if p_vote > n_vote and cls_label == "+1":
            record = cls_label + " " + str(tup[2:].strip()) + " -> true"   # data in LIBSVM format
            prediction_rec.append(record)
            tp_prediction += 1
            true_prediction += 1
        
        elif p_vote < n_vote and cls_label == "-1":
            record = cls_label + " " + str(tup[2:].strip()) + " -> true"
            prediction_rec.append(record)
            tn_prediction += 1
            true_prediction += 1
        
        elif p_vote < n_vote and cls_label == "+1":
            record = cls_label + " " + str(tup[2:].strip()) + " -> false"
            prediction_rec.append(record)
            fp_prediction += 1
            false_prediction += 1
        
        else:
            record = cls_label + " " + str(tup[2:].strip()) + " -> false"
            prediction_rec.append(record)
            fn_prediction += 1
            false_prediction += 1

    return true_prediction, false_prediction, prediction_rec, tp_prediction, tn_prediction, fp_prediction, fn_prediction


def main():
    
    # ----- parse arguments -----
    
    parser = argparse.ArgumentParser(description='Implement AdaBoost Algorithm')
    parser.add_argument('training_dataset', help="Require a trainning dataset location")
    parser.add_argument('test_dataset', help="Require a test dataset location")
    parser.add_argument('dstDirectory', help="Require an output directory")
    parser.add_argument('num_attribute', help="Require the max number of attributes")
    parser.add_argument('iteration', help="Require the number of iterations")
    
    args = parser.parse_args()
    training_file = args.training_dataset
    testing_file = args.test_dataset
    num_attr = int(args.num_attribute)
    n = int(args.iteration)
    dst_directory = str(args.dstDirectory)
    
    
    # ----- make a directory -----
    
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)
    
    
    # ----- generate names of training files -----
    
    filename_list = list()
    file_prefix = "TrainingSet_"
    for num in range(n):
        filename_list.append(file_prefix + str(num))


    # ----- replace missing attributes -----

    new_training = os.path.join(dst_directory, "new_" + os.path.basename(training_file))
    new_test =  os.path.join(dst_directory, "new_" + os.path.basename(testing_file))
    new_training_file = data_calibration(training_file, new_training, num_attr)
    new_testing_file = data_calibration(testing_file, new_test, num_attr)


    # ----- implementation ensemble methods -----
    
    M = AdaBoost(new_training_file)
    mt = AdaBoost(new_training_file)
    n_tuples = M.WeightGenerator()
    mt.weight_table = copy.deepcopy(M.weight_table)

    for file_name in filename_list:
        M.ClassifierGenerator(file_name, dst_directory, n_tuples, new_testing_file, mt, class_weight)


    # ----- generate classified results of the test file -----

    t_prediction, f_prediction, predictions, tp_prediction, tn_prediction, fp_prediction, fn_prediction = classification_summary(class_weight)

    print("tp", tp_prediction, "tn", tn_prediction, "fp", fp_prediction, "fn", fn_prediction)


    # ----- write the output -----

    name = os.path.basename(testing_file)
    X = os.path.splitext(name)[0].split(',')[0]
    f_name = "Ensemble - " + X
    write_output(dst_directory, f_name, t_prediction, f_prediction, predictions)


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()




