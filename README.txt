Source code are in nblr/src folder:
  fileParse.py, used for parse the training and testing set
  mcapLR.py, implementation of MCAP Logistic Regression Classifier
  multiNB.py, implementation of multinomial Naive Bayes Classifier
  miattrsele.py, this class is designed for extra credit part,
    which uses mutual information to select top ranked related attributes
  testLR.py, designed for testing MCAP Logistic Regression Classifier
  testNB.py, designed for testing multinomial Naive Bayes Classifier
 
How to compile & run (all in command line):
  method 1:
    use cd command to the nblr/src folder
    type python testNB.py arg1 arg2 arg3 arg4 arg5 arg6 to run the test for Naive Bayes Classifier
    where,
      arg1, path of training ham data set
      arg2, path of training spam data set
      arg3, path of testing ham data set
      agr4, path of testing spam data set
      arg5, argument of determining which operation on the original text set to be used,
            1 for original sequence(including punctuation),
            2 for word sequence,
            3 for word sequence without stop words (stop words are in nblr/stopwords file)
            4 for word sequence after filtering by mutual information
      arg6, percent of vocabulary size after using mutual information as a filter, range is [0, 1]
    type python testLR.py arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8 arg9
    where,
      arg1, path of training ham data set
      arg2, path of training spam data set
      arg3, path of testing ham data set
      arg4, path of testing spam data set
      arg5, argument of determining which operation on the original text set to be used,
            2 for word sequence,
            3 for word sequence without stop words (stop words are in nblr/stopwords file)
            4 for word sequence after filtering by mutual information
            (note that 1 is not available for avoiding overflow)
      arg6, learning rate
      arg7, lambda (for prior)
      arg8, iteration number
      arg9, percent of vocabulary size after using mutual information as a filter, range is [0, 1]
  method 2:
    use cd command to the nblr folder
    specify the argument mentioned above
      in nb.sh, which is for testing the naive bayes classifier,
      in lr.sh, which is for testing the logistic regression classifier
    type sh nb.sh, or sh lr.sh to run the corresponding testing
    
    
## Detailed report about the accuracies can be find in the nblr/HW2Report.pdf file. ##
    
      
      