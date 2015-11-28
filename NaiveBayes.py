
# author: Cheng Peng
# brief : build general purpose classification framework based on Naive Bayes method, which is able to construct classifiers, and use the classifier to assign labels to unseen test data instances.

# -----------------------------------------------------------------------------
# import modules

from __future__ import division
from collections import defaultdict
import argparse
import os


# ----- define a function that writes the output into a text file -----

def write_output(directory, filename, true_pred, false_pred, preds):

    file_o = "{0}.txt".format(filename)
    dst_path = os.path.join(directory, file_o)
    with open(dst_path, "w") as fo:
        fo.write("Total instance:" + str(len(preds)) + "\n" + "True predictions:" + str(true_pred) + "\n" + "False predictions:" + str(false_pred) + "\n" + "Accuracy:" + str(true_pred/len(preds) * 100) + "%" + "\n\n************************************************\n")
        for p in preds:
            fo.write(p + "\n")


# ----- define a function that handles the replacement of missing attributes -----

def data_calibration(input_file, output_file, num_attr):

    with open(input_file, 'r') as fi, open(output_file, 'w') as fo:
        for line in fi:
            tup = line.strip().split(' ')
            length = len(tup)
            new_line = list()
            shift = 1
            if length - 1 < num_attr:
                new_line.append(tup[0])
                for index, attribute_value in enumerate(tup[1:]):
                    attribute, value = attribute_value.split(':')
                    if index+shift < int(attribute):
                        num = int(attribute) - (index+shift)
                        for k in range(num):
                            new_line.append("{0}:{1}".format(index+shift+k,0))
                        new_line.append(attribute_value)
                        shift += num
                    else:
                        new_line.append(attribute_value)
                new_length = len(new_line) - 1
                if new_length < num_attr:
                    num = num_attr - new_length
                    for i in range(num):
                        new_line.append("{0}:{1}".format(new_length+i+1,0))
                for item in new_line:
                    fo.write("%s " % item)
                fo.write('\n')
            else:
                fo.write(line)

    return output_file


# ----- define a class that implements Naive Bayes method -----

class Bayes_Model(object):
    
    def __init__(self, File):
        # class constructor

        self.data_file = File


    def GetValues(self):
        # calculate prior probablility P(Cls) and likelyhood P(attr|Cls)
        
        self.lkhood_positive = defaultdict(int)  # a dictionary that counts the number of each attribute-value
        self.lkhood_negative = defaultdict(int)
        self.prior_frequency = defaultdict(int)  # a dictionary that counts the number of each class label
        self.prior_probability = defaultdict(float)  # contain the prior probability of each class label

        with open(self.data_file, 'r') as trainData:
            for line in trainData:
                if line.startswith('+1'):
                    instance = line.strip().split(' ')
                    self.prior_frequency[instance[0]] += 1
                    for attribute_value in instance[1:]:
                        index, value = attribute_value.split(':')
                        self.lkhood_positive[(instance[0], index, value)] += 1
                elif line.startswith('-1'):
                    instance = line.strip().split(' ')
                    self.prior_frequency[instance[0]] += 1
                    for attribute_value in instance[1:]:
                        index, value = attribute_value.split(':')
                        self.lkhood_negative[(instance[0], index, value)] += 1
    
        instance_counter = sum([instance for instance in self.prior_frequency.values()])
         
        for key, value in self.prior_frequency.items():   # calculate P(Cls)
            self.prior_probability[key] = value / instance_counter
                 
        for key, value in self.lkhood_positive.items():   # calculate P(X|C='+1')
            self.lkhood_positive[key] = value / self.prior_frequency['+1']
        
        for key, value in self.lkhood_negative.items():   # calculate P(X|C='-1')
            self.lkhood_negative[key] = value / self.prior_frequency['-1']

        return instance_counter


    def Classify(self, training_model):
    
        true_prediction = 0  # count the number of true predictions
        false_prediction = 0
        tp_prediction = 0  # count the number of true positive predictions
        tn_prediction = 0  # count the number of true negative predictions
        fp_prediction = 0  # count the number of false positive predictions
        fn_prediction = 0  # count the number of false negative predictions
        prediction_rec = list() # a list that records the predictions
        lkhood_positive = defaultdict(float)  # contain the probability(attr|cls=+1)
        lkhood_negative = defaultdict(float)  # contain the probability(attr|cls=-1)

        with open(self.data_file, 'r') as testData:
            for line in testData:
                instance = line.strip().split(' ')
                cls_label = instance[0]
                lkhood_positive[str(instance[1:])] = 1.0
                lkhood_negative[str(instance[1:])] = 1.0
                for attribute_value in instance[1:]:
                    attr, value = attribute_value.split(':')
                    positive_match = ('+1', attr, value)
                    negative_match = ('-1', attr, value)
                    if positive_match in training_model.lkhood_positive.keys():
                        lkhood_positive[str(instance[1:])] *= training_model.lkhood_positive[positive_match]
                    if negative_match in training_model.lkhood_negative.keys():
                        lkhood_negative[str(instance[1:])] *= training_model.lkhood_negative[negative_match]

                prob_p = lkhood_positive[str(instance[1:])] * training_model.prior_probability['+1']   # P(C='+1'|X)
                prob_n = lkhood_negative[str(instance[1:])] * training_model.prior_probability['-1']   # P(C='-1'|X)
                if prob_p > prob_n and cls_label == '+1':   # true positive prediction
                    record = cls_label + " " + str(line[2:].strip()) + " -> true"   # data in LIBSVM format
                    prediction_rec.append(record)
                    tp_prediction += 1
                    true_prediction += 1
                
                elif prob_p < prob_n and cls == '-1':   # true negative prediction
                    record = cls_label + " " + str(line[2:].strip()) + " -> true"   # data in LIBSVM format
                    prediction_rec.append(record)
                    tn_prediction += 1
                    true_prediction += 1
                      
                elif prob_p < prob_n and cls_label == '+1':   # false positive prediction
                    record = cls_label + " " + str(line[2:].strip()) + " -> false"
                    prediction_rec.append(record)
                    fp_prediction += 1
                    false_prediction += 1
    
                elif prob_p > prob_n and cls_label == '-1':   # false negative prediction
                    record = cls_label + " " + str(line[2:].strip()) + " -> false"
                    prediction_rec.append(record)
                    fn_prediction += 1
                    false_prediction += 1

        return true_prediction, false_prediction, prediction_rec, tp_prediction, tn_prediction, fp_prediction, fn_prediction


def main():

# ----- parse arguments -----

    parser = argparse.ArgumentParser(description='Implement Naive Bayes Algorithm')
    parser.add_argument('training_dataset', help="Require a trainning dataset location")
    parser.add_argument('test_dataset', help="Require a test dataset location")
    parser.add_argument('dstDirectory', help="Require an output directory")
    parser.add_argument('num_attribute', help="Require the max number of attributes")
    
    args = parser.parse_args()
    training_file = args.training_dataset
    test_file = args.test_dataset
    dst_directory = str(args.dstDirectory)
    num_attr = int(args.num_attribute)

# ----- make a directory -----

    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)


# ----- replace missing attributes -----


    new_training = os.path.join(dst_directory, "new_" + os.path.basename(training_file))
    new_test =  os.path.join(dst_directory, "new_" + os.path.basename(test_file))
    new_training_file = data_calibration(training_file, new_training, num_attr)
    new_test_file = data_calibration(test_file, new_test, num_attr)


# ----- read training dataset and test dataset -----

    nbmodel = Bayes_Model(new_training_file)
    num_train = nbmodel.GetValues()
    nbtest = Bayes_Model(new_test_file)
    t_prediction, f_prediction, predictions, tp_prediction, tn_prediction, fp_prediction, fn_prediction = nbtest.Classify(nbmodel)
    print("tp", tp_prediction, "tn", tn_prediction, "fp", fp_prediction, "fn", fn_prediction)


# ----- write the output -----

    name = os.path.basename(test_file)
    X = os.path.splitext(name)[0].split(',')[0]
    file_name = "Basic - " + X
    write_output(dst_directory, file_name, t_prediction, f_prediction, predictions)

# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()
