# Classification
Implement Naive Bayes &amp; Naive Bayes based AdaBoost Classifications on Dataset with LIBSVM format.

- Implementation of Code

$python ~/Code/NaiveBayes.py ~/Dataset/breast_cancer\,\ training.txt ~/Dataset/breast_cancer\,\ test.txt ~/Output/ 9

where ~/Code/NaiveBayes.py is the script, ~/Dataset/breast_cancer\,\ training.txt is the training dataset, ~/Dataset/breast_cancer\,\ test.txt is the testing dataset, and ~/Output is the location for outputs, “9” is the number of attributes for the corresponding datasets. 

$python ~/Code/AdaBoost.py ~/Dataset/breast_cancer\,\ training.txt ~/Dataset/breast_cancer\,\ test.txt ~/Output 9 6

where ~/Code/AdaBoost.py is the script, ~/Dataset/breast_cancer\,\ training.txt is the training dataset, ~/Dataset/breast_cancer\,\ test.txt is the testing dataset, and ~/Output is the output directory, “9” is the number of attributes, and “6” is the number of iterations.


- What the scrips do

Running NaiveBayes.py generates a Basic - X.txt file in your Output directory, and prints out the number of predictions for true positive, true negative, false positive, and false negative.

Running AdaBoost.py generates a Ensemble - X.txt file in your Output directory, and prints out the number of predictions for true positive, true negative, false positive, and false negative. Besides, it would generate some immediate “TrainingSet_N.txt” files, which are the training files for creating new Naive Bayes Classifiers, and whose number is based on the iterations. 

For the case of incomplete attributes in tuples, such as led dataset files, running scripts above would also generate two files named new_X, training.txt and new_X, test.txt in the Output directory, which are regenerated files with attributes replacement.
