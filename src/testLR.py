'''
Created on Feb 20, 2017

@author: mdy
'''
from __future__ import division
from mcapLR import mcapLR
import sys
import time

def testLR():
    
    print "test MCAP Logistic Regression Classifier"
    
    tr_ham_path = str(sys.argv[1])
    tr_spam_path = str(sys.argv[2])
    
    te_ham_path = str(sys.argv[3])
    te_spam_path = str(sys.argv[4])
    
    use_which = int(sys.argv[5])
    
    p_ita = float(sys.argv[6])
    p_lam = float(sys.argv[7])
    p_itr = int(sys.argv[8])
    
    prct = float(sys.argv[9])
    
    #loop until convergence is not implemented in my design
    p_mer = 0.0001
    
    if use_which==2:
        print "-- use original word sequence --"
    if use_which==3:
        print "-- use word sequence after filter the stop words --"
    if use_which==4:
        print "-- using word sequence after filter the words selected by mutual information --"
        print "selection rate: " + str(prct)
        
    print "learning rate setting: " + str(p_ita)
    print "prior parameter lambda: " + str(p_lam)
    print "iteration number: " + str(p_itr)
    
    #train
    t1 = time.time()
    lr = mcapLR(tr_ham_path, tr_spam_path, use_which, p_ita, p_lam, p_mer, p_itr, prct)
    t2 = time.time()
    print "training time: " + str(t2-t1) + 's'
    
    lr.apply_mcapLR(te_ham_path)
    pst1 = lr.p_rst
    accr1 = (len(pst1)-sum(pst1))/len(pst1)
    print "prediction accuracy in ham set: " + str( accr1*100 ) +'%'
    
    lr.apply_mcapLR(te_spam_path)
    pst2 = lr.p_rst
    accr2 = sum(pst2)/len(pst2)
    print "prediction accuracy in spam set: " + str( accr2*100 ) + '%'
    
    accr = (len(pst1)-sum(pst1)+sum(pst2))/(len(pst1)+len(pst2))
    print "total prediction accuracy: " + str( accr*100 ) +'%'
    
    
if __name__=='__main__':
    testLR()
    
    
    
    