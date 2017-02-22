#!/bin/sh

cdir=$(pwd)

#specify the arguments
tr_ham_path=$cdir/dataset/train/ham
tr_spam_path=$cdir/dataset/train/spam
te_ham_path=$cdir/dataset/test/ham
te_spam_path=$cdir/dataset/test/spam
#use_which can be set as 2, 3, 4
use_which=2
p_ita=0.01
p_lam=0
p_itr=20
prct=0.7

cd $cdir/src
python testLR.py $tr_ham_path $tr_spam_path $te_ham_path $te_spam_path $use_which $p_ita $p_lam $p_itr $prct
