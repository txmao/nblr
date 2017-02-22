'''
Created on Feb 17, 2017

@author: mdy
'''
from __future__ import division
from fileParse import fileParse
import numpy as np

class miattrsele:
    # row ham and spam, and selection percent
    def __init__(self, ham_for_mi, spa_for_mi, s_perc):
        self.mi_ham = []
        self.mi_spa = []
        self.vocaset = []
        self.w_seq = []
        #
        self.vocaset0 = []
        self.w_seq0 = []
        #
        self.vocaset1 = []
        self.w_seq1 = []
        self.__hsinfo__(ham_for_mi, spa_for_mi)
        self.mi_selection(ham_for_mi, spa_for_mi, s_perc)
        
    def __hsinfo__(self, ham, spa):
        ls_ham = []
        ls_spa = []
        for i in range(len(ham)):
            ls_ham.extend(ham[i])
            
        for i in range(len(spa)):
            ls_spa.extend(spa[i])
            
        self.w_seq0 = ls_ham
        self.w_seq1 = ls_spa
        vs_ham = set(ls_ham)
        vs_spa = set(ls_spa)
        vl_ham = list(vs_ham)
        vl_spa = list(vs_spa)
        self.vocaset0 = vl_ham
        self.vocaset1 = vl_spa
        
        
    def mi_selection(self, ham_for_mi, spa_for_mi, s_perc):
        # original data and label stored in D_dict, 0 for ham and 1 for spam
        D_dict = self.__extractDict__(ham_for_mi, spa_for_mi)
        Voca = self.__getVocaset__(D_dict)
        self.vocaset = Voca
        Ij = self.__Ivector__(ham_for_mi, spa_for_mi, Voca)
        
        ind_rank = self.__getRankedIndex__(Ij)
        
        l_Voca = len(Voca)
        sel_num = int(l_Voca*s_perc)
        
        sel_ind = [-1] * sel_num
        for i in range(sel_num):
            sel_ind[i] = ind_rank[l_Voca-1 - i]
            
        sel_voca = []
        for i in range(len(sel_ind)):
            sel_voca.append(Voca[sel_ind[i]])
        
        self.__miData__(sel_voca, ham_for_mi, spa_for_mi)
        
        
    
    def __miData__(self, selvoca, ham, spa):
        for i in range(len(ham)):
            for j in range(len(ham[i])):
                if ham[i][j] not in selvoca:
                    ham[i][j] = None
                    
            ham[i] = [x for x in ham[i] if x!=None]
            
        self.mi_ham = ham
        
        for i in range(len(spa)):
            for j in range(len(spa[i])):
                if spa[i][j] not in selvoca:
                    spa[i][j] = None
                    
            spa[i] = [x for x in spa[i] if x!=None]
            
        self.mi_spa = spa
        
            
    def __extractDict__(self, ham, spa):
        D = {}
        D.setdefault(0, ham)
        D.setdefault(1, spa)
        return D
    
    def __getVocaset__(self, D):
        ls1 = []
        for k in D.iterkeys():
            ls1.append(D[k])
            
        ls2 = []
        ls2.extend(ls1[0])
        ls2.extend(ls1[1])
        
        ls3 = []
        for i in range(len(ls2)):
            ls3.extend(ls2[i])
            
        self.w_seq = ls3
        #print len(ls3)
        mgst = set(ls3)
        mgst1 = list(mgst)
        return mgst1
    
    
    def __Ivector__(self, ham, spa, voca):
        I_j = [0]*len(voca)
        pi0 = len(ham) / (len(ham) + len(spa))
        pi1 = 1-pi0
        
        len_seq = len(self.w_seq)
        # count the frequency of every word in vocabulary
        thetaj = [0]*len(voca)
        for j in range(len(voca)):
            cntj = 0
            for l in range(len(self.w_seq)):
                if self.w_seq[l]==voca[j]:
                    cntj += 1
                    
            thetaj[j] = (cntj + 1) / (len_seq + len(self.vocaset))
            
        thetaj0 = [0]*len(voca)
        len_seq0 = len(self.w_seq0)
        for j in range(len(voca)):
            cntj0 = 0
            for l in range(len(self.w_seq0)):
                if self.w_seq0[l]==voca[j]:
                    cntj0 += 1
                    
            thetaj0[j] = (cntj0 + 1) / (len_seq0 + len(self.vocaset0))
            
        
        thetaj1 = [0]*len(voca)
        len_seq1 = len(self.w_seq1)
        for j in range(len(voca)):
            cntj1 = 0
            for l in range(len(self.w_seq1)):
                if self.w_seq1[l]==voca[j]:
                    cntj1 += 1
                    
            thetaj1[j] = (cntj1 + 1) / (len_seq1 + len(self.vocaset1))
            
        
        #print sum(thetaj)
        # compute Ij
        
        I_j0 = [0]*len(voca)
        for j in range(len(voca)):
            p1 = thetaj0[j] * pi0 * np.log( thetaj0[j] / thetaj[j] )
            p2 = (1 - thetaj0[j]) * pi0 * ( np.log( (1-thetaj0[j]) / (1-thetaj[j]) ) )
            I_j0[j] = p1+p2
            
        I_j1 = [0]*len(voca)
        for j in range(len(voca)):
            p1 = thetaj1[j] * pi1 * np.log( thetaj1[j] / thetaj[j] )
            p2 = (1 - thetaj1[j]) * pi1 * ( np.log( (1-thetaj1[j]) / (1-thetaj[j]) ) )
            I_j1[j] = p1+p2
            
        for j in range(len(voca)):
            I_j[j] = (I_j0[j] + I_j1[j]) * 1000
            
        
        return I_j
    
    
    def __getRankedIndex__(self, Ij):
        return np.argsort(Ij)
    
    
        
        
    
if __name__=='__main__':
    hampath = '/home/mdy/Desktop/hw2/train/ham'
    spampath = '/home/mdy/Desktop/hw2/train/spam'
    ph = fileParse(hampath)
    ps = fileParse(spampath)
    hfm = ph.D_row
    sfm = ps.D_row
    #print hfm[0]
    #print sfm[0]
    mias = miattrsele(hfm, sfm, 0.5)
    print mias.mi_ham[0]
    