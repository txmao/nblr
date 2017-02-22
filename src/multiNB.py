'''
Created on Feb 11, 2017

@author: mdy
'''

from __future__ import division
from fileParse import fileParse
import math
from miattrsele import miattrsele

class multiNB:
    
    # spam & ham path
    # use_which D_row, D_wrd, D_flt, D_mut
    def __init__(self, ham_Path, spam_Path, use_which, prct):
        # parse used files
        pHam = fileParse(ham_Path)
        pSpa = fileParse(spam_Path)
        # train set D_ham & D_spa
        self.D_ham = []
        self.D_spa = []
        if use_which == 1 :
            self.D_ham = pHam.D_row
            self.D_spa = pSpa.D_row
            
        if use_which == 2 :
            self.D_ham = pHam.D_wrd
            self.D_spa = pSpa.D_wrd
            
        if use_which == 3 :
            self.D_ham = pHam.D_flt
            self.D_spa = pSpa.D_flt
            
        if use_which == 4 :
            ham_for_mi = pHam.D_row
            spa_for_mi = pSpa.D_row
            misel = miattrsele(ham_for_mi, spa_for_mi, prct)
            self.D_ham = misel.mi_ham
            self.D_spa = misel.mi_spa
            
            
        self.use_which = use_which
            
        # 0->ham, 1->spam, a dictionary
        self.D = {}
        self.D.setdefault(0, self.D_ham)
        self.D.setdefault(1, self.D_spa)
        
        # train statistics
        self.Vocabulary_Set = set()
        self.Words_Sequence = []
        self.Doc_Count = 0
        self.ConcatenateTextOfAllDocInClass = [[], []]
        
        self.statistic_extraction(self.D)
        
        #train
        self.prior = [0, 0]
        self.condprob = [{}, {}]
        self.train_multinomial_NB(self.D)
        
        
        # predict_part
        self.p_rst = []
        
        
    
    def train_multinomial_NB(self, D):
        V = self.Vocabulary_Set
        N = self.Doc_Count
        
        N_c = [0, 0]
        prior_c = [0, 0]
        text_c = self.ConcatenateTextOfAllDocInClass
        Tct = [{}, {}]
        condprob = [{}, {}]
        for k in D.iterkeys():
            N_c[k] = len(D[k])
            prior_c[k] = N_c[k]/N
            for t in V:
                Tct[k].setdefault(t, self.__getWordNumber__(text_c[k], t))
                
            for t in V:
                condprob[k].setdefault(t, ( Tct[k][t]+1 )/( len(text_c[k]) + len(V) ) )
                
        self.prior = prior_c
        self.condprob = condprob
        return
    
    
    
    def __getWordNumber__(self, text_c, t):
        cnt = 0
        for i in range(len(text_c)) :
            if text_c[i] == t :
                cnt += 1
                
        return cnt
        
            
    
    
    
    def statistic_extraction(self, D):
        total_list = []
        #[[],[]]
        for k in D.iterkeys():
            total_list.append(D[k])
            
        total_list1 = []
        total_list1.extend(total_list[0])
        total_list1.extend(total_list[1])
        self.Doc_Count = len(total_list1)
        
        merge_list = []
        for i in range(len(total_list1)):
            merge_list.extend(total_list1[i])
            
        self.Words_Sequence = merge_list
        merge_set = set(merge_list)
        self.Vocabulary_Set = merge_set
        
        #print len(total_list)
        lst = [[], []]
        for j in range(len(total_list)):
            for l in range(len(total_list[j])):
                lst[j].extend(total_list[j][l])
                
        self.ConcatenateTextOfAllDocInClass = lst
        
        return
    
    
    def apply_multinomial_NB(self, predictpath):
        pDoc = fileParse(predictpath)
        if self.use_which == 1:
            doc = pDoc.D_row
            
        if self.use_which == 2:
            doc = pDoc.D_wrd
            
        if self.use_which == 3:
            doc = pDoc.D_flt
            
        if self.use_which == 4:
            doc = pDoc.D_row
        
        p_rst = []
        #score = [0, 0]
        for i in range(len(doc)):
            W = self.extract_tokens_from_doc(self.Vocabulary_Set, doc[i])
            score = [0, 0]
            #print len(W)
            for k in self.D.iterkeys() :
                score[k] = math.log(self.prior[k])
                for t in W :
                    score[k] += math.log(self.condprob[k][t])
                    
            if score[0] > score[1]:
                p_rst.append(0)
                
            else:
                p_rst.append(1)
                    
        
        self.p_rst = p_rst        
            
        return
    
    
    def extract_tokens_from_doc(self, V, d):
        W = []
        for i in range(len(d)):
            if d[i] in V :
                W.append(d[i])
                
        return set(W)
    
    
    
    

if __name__=='__main__':
    hampath = '/home/mdy/Desktop/hw2/train/ham'
    spampath = '/home/mdy/Desktop/hw2/train/spam'
    predictpath = '/home/mdy/Desktop/hw2/testNB/spam'
    mnb = multiNB(hampath, spampath, 2, 0.8)
    mnb.apply_multinomial_NB(predictpath)
    print sum(mnb.p_rst)
    #print len(mnb.Vocabulary_Set)
    #print len(mnb.Words_Sequence)
    #print mnb.D_ham[0]
            
            
            