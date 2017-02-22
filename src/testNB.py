'''
Created on Feb 20, 2017

@author: mdy
'''
from __future__ import division
from multiNB import multiNB
import sys
import time

def testNB():
    
    print "Test Multinomial Naive Bayes Classifier"
    
    #training sets path
    tr_ham_path = str(sys.argv[1])
    tr_spam_path = str(sys.argv[2])
    
    #testing sets path
    te_ham_path = str(sys.argv[3])
    te_spam_path = str(sys.argv[4])
    
    #word selection method
    use_which = int(sys.argv[5])
    #mutual information selection rate
    prct = float(sys.argv[6])
    
    if use_which==2:
        print "-- use original word sequence --"
    if use_which==3:
        print "-- use word sequence after filter the stop words --"
    if use_which==4:
        print "-- using word sequence after filter the words selected by mutual information --"
        print "selection rate: " + str(prct)
    #train
    t1 = time.time()
    mnb = multiNB(tr_ham_path, tr_spam_path, use_which, prct)
    t2 = time.time()
    print "training time: " + str(t2-t1) +'s'
    
    mnb.apply_multinomial_NB(te_ham_path)
    accr1 = (len(mnb.p_rst) - sum(mnb.p_rst)) / len(mnb.p_rst)
    print "prediction accuracy on ham set: " + str(accr1*100) + '%'
    
    pst1 = mnb.p_rst
    
    mnb.apply_multinomial_NB(te_spam_path)
    accr2 = sum(mnb.p_rst) / len(mnb.p_rst)
    print "prediction accuracy on spam set: " + str(accr2*100) + '%'
    
    pst2 = mnb.p_rst
    
    accr = (len(pst1) - sum(pst1) + sum(pst2)) / (len(pst1) + len(pst2))
    print "total prediction accuracy: " + str(accr*100) + '%'
    
    
if __name__=='__main__':
    testNB()
    
    
    
    
    