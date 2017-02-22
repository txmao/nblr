'''
Created on Feb 11, 2017

@author: mdy
'''

import os
import string

class fileParse:
    
    def __init__(self, rd_path):
        self.rd_path = rd_path
        # Drow is original, Dwrd is only word
        # Dflt is after filter stop words, D_mut is mutual information
        # use index 1,2,3,4
        self.D_row = []
        self.D_wrd = []
        self.D_flt = []
        self.D_mut = [] # left it after nl and lr
        self.stop_set = self.stop_set_construct()
        self.f_parse()
        
    
    
    def stop_set_construct(self):
        # stop words
        stp_p = '../stopwords'
        with open(stp_p) as f:
            ln = f.read().split()
            ln = [word.strip(string.punctuation).lower() for word in ln]
            stop_set = set (ln)
            
        return stop_set
    
    
    
    def f_parse(self):
        f_p = self.rd_path
        # D_row, original, include
        # D_wrd, without punctuation
        for f_name in os.listdir(f_p) :
            with open(f_p + '/' + f_name) as fl :
                # D_row
                ln = fl.read().split()
                #self.D_row.append(ln)
                wrdr = [word for word in ln]
                wrds = [x.lower() for x in wrdr if x]
                self.D_row.append(wrds)
                # D_wrd
                wrd1 = [word.strip(string.punctuation) for word in ln]
                wrd = [x.lower() for x in wrd1 if x]
                self.D_wrd.append(wrd)
                # D_flt
                self.D_flt.append( self.__stopFilter__(wrd, self.stop_set) )
                fl.close()
                
                
    def __stopFilter__(self, wrd, stp_set):
        #
        wfl = []
        for w in wrd:
            if w not in stp_set:
                wfl.append(w)
                
        return wfl
        
    
    
    

if __name__ == '__main__' :
    f_p = '/home/mdy/Desktop/hw2/train/spam'
    D = fileParse(f_p)
    #print D.D_row[0]
    print D.D_row[0]
    print D.D_wrd[0]
    print D.D_flt[0]
    ls = [1,2,3,5]
    ls[2] = None
    ls[3] = None
    ls.remove(None)
    print ls
    if 9 not in ls:
        print 'okay'
