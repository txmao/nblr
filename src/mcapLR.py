'''
Created on Feb 11, 2017

@author: mdy
'''

from __future__ import division
from fileParse import fileParse
#import math
import numpy as np
import time
from miattrsele import miattrsele

class mcapLR:
    def __init__(self, ham_path, spam_path, use_which, p_ita, p_lam, min_er, max_iter, prct):
        # p_ita is learning rate, p_lam is the parameter of prior
        # max_iter is the max iteration it can take to train
        # min_er determine convergence
        
        # parse for training
        pHam = fileParse(ham_path)
        pSpam = fileParse(spam_path)
        
        self.D_ham = []
        self.D_spam = []
        
        if use_which == 1:
            self.D_ham = pHam.D_row
            self.D_spam = pSpam.D_row
            
        if use_which == 2:
            self.D_ham = pHam.D_wrd
            self.D_spam = pSpam.D_wrd
            
        if use_which == 3:
            self.D_ham = pHam.D_flt
            self.D_spam = pSpam.D_flt
            
        if use_which == 4:
            ham_for_mi = pHam.D_wrd
            spa_for_mi = pSpam.D_wrd
            misel = miattrsele(ham_for_mi, spa_for_mi, prct)
            self.D_ham = misel.mi_ham
            self.D_spam = misel.mi_spa
            
        self.use_which = use_which
            
        # 0->ham, 1->spam, list of data
        self.D = {}
        self.D.setdefault(0, self.D_ham)
        self.D.setdefault(1, self.D_spam)
        
        # get statistic information
        # this is set->list, in order to get the index
        self.Vocabulary_Set = []
        
        self.get_statistic_info()
        
        #matrix used for train
        self.XY = self.getXYmatrix()
        
        # set initial weight, will be updated during training
        self.W = [0]*(len(self.Vocabulary_Set)+1)
        
        self.train_mcapLR(p_ita, p_lam, min_er, max_iter)
        
        # predict
        self.p_rst = []
        
        
    def train_mcapLR(self, p_ita, p_lam, min_er, max_iter):
        cur_err = min_er+1
        cur_itr = 0
        while (cur_err>min_er and cur_itr<max_iter):
            for i in range(len(self.XY)):
                loss1 = self.getLoss(i, self.XY, self.W, p_ita, p_lam)
                for j in range(len(self.W)):
                    self.W[j] = self.W[j] + loss1*self.XY[i][j] - p_ita*p_lam*self.W[j]
                    #print self.W[j]
                    
            cur_itr += 1
        
    
    def getLoss(self, i, xy, w, ita, lam):
        WX = 0
        for l in range(len(w)):
            WX += w[l]*xy[i][l]
            
        py1x = np.exp(WX) /( 1+np.exp(WX) )
        
        return ita*( xy[i][-1] - py1x )
        
        
        
    def get_statistic_info(self):
        list1 = []
        for k in self.D.iterkeys():
            list1.append(self.D[k])
            
        list2 = []
        list2.extend(list1[0])
        list2.extend(list1[1])
        
        list3 = []
        for i in range(len(list2)):
            list3.extend(list2[i])
        
        self.Vocabulary_Set = list(set(list3))
        #print len(self.Vocabulary_Set)
        #print len(list3)
        
        
    def getXYmatrix(self):
        XY = []
        for key in self.D.iterkeys():
            for doc in self.D[key]:
                x_vec = self.get_xy_vector(doc)
                x_vec[-1] = key
                XY.append(x_vec)
            
        return XY
    
    def get_xy_vector(self, doc):
        x = [0]*(len(self.Vocabulary_Set)+2)
        x[0] = 1 # for intersection
        for wd in doc:
            if wd in self.Vocabulary_Set:
                ind = self.Vocabulary_Set.index(wd)
                x[ind + 1] += 1
                
        #print sum(x)
        return x
    
    
    def apply_mcapLR(self, p_path):
        pdoc = fileParse(p_path)
        if self.use_which == 1:
            doc = pdoc.D_row
            
        if self.use_which == 2:
            doc = pdoc.D_wrd
            
        if self.use_which == 3:
            doc = pdoc.D_flt
            
        if self.use_which == 4:
            doc = pdoc.D_wrd
            
        p_rst = []
        for i in range(len(doc)):
            p = 0
            X = self.get_x_vector(doc[i])
            for i in range(len(self.W)):
                p += self.W[i]*X[i]
            
            if p>0:
                p_rst.append(1)
            else:
                p_rst.append(0)
                
        self.p_rst = p_rst    
            
            
            
    def get_x_vector(self, d):
        X = [0] * ( len(self.Vocabulary_Set) + 1 )
        X[0] = 1
        for wd in d:
            if wd in self.Vocabulary_Set:
                ind = self.Vocabulary_Set.index(wd)
                X[ind+1] += 1
                
        return X
            
        
            
        
if __name__=='__main__':
    hampath = '/home/mdy/Desktop/hw2/train/ham'
    spampath = '/home/mdy/Desktop/hw2/train/spam'
    predictpath = '/home/mdy/Desktop/hw2/testNB/ham'
    t1 = time.time()
    mlr = mcapLR(hampath, spampath, 1, 0.1, 0.001, 0.00001, 30, 0.9)
    mlr.apply_mcapLR(predictpath)
    t2 = time.time()
    #print mlr.W
    print sum(mlr.p_rst)
    print (t2-t1)
    print len(mlr.Vocabulary_Set)
    
    
    
