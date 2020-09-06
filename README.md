# asu-project
For your own code logic, you have 4 tasks to do:

Task 1. You need to first extract features from the your original trainset in order to convert the original data arrays to 2-Dimentional data points.

You are required to extract the following two features for each image:

Feature1:The average brightness of each image (average all pixel brightness values within a whole image array)

Feature2:The standard deviation of the brightness of each image (standard deviation of all pixel brightness values within a whole image array)

We assume that these two features are independent, and that each image is drawn from a normal distribution. 

Task 2. You need to calculate all the parameters for the two-class naive bayes classifiers respectively, based upon the 2-D data points you generated in Task1. (Totally you should have 8 parameters)

(No.1) Mean of feature1 for digit0

(No.2) Variance of feature1 for digit0

(No.3) Mean of feature2 for digit0

(No.4) Variance of feature2 for digit0

(No.5) Mean of feature1 for digit1

(No.6) Variance of feature1 for digit1

(No.7) Mean of feature2 for digit1

(No.8) Variance of feature2 for digit1

Task 3. Since you get the NB classifiers' parameters from Task2, you need to implement their calculation formula according to their Mathematical Expressions. Then you use your implemented classifiers to classify/predict all the unknown labels of newly coming data points (your test data points converted from your original testset for both digit0 and digit1). Thus, in this task, you need to work with the testset for digit0 and digit1 (2 Numpy Arrays: test0 and test1 mentioned above) and you need to predict all the labels of them.

PS: Remember to first convert your original 2 test data arrays (test0 and test1) into 2-D data points as exactly the same way you did in task1.

Task 4. In task3 you successfully predicted the labels for all the test data, now you need to calculate the accuracy of your predictions for testset for both digit0 and digit1 respectively.
