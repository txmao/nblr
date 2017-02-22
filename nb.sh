#!/bin/sh

#get current directory
cdir=$(pwd)

#specify the arguments
tr_ham_path=$cdir/dataset/train/ham
tr_spam_path=$cdir/dataset/train/spam
te_ham_path=$cdir/dataset/test/ham
te_spam_path=$cdir/dataset/test/spam
#use_which can be set as 1, 2, 3, 4
use_which=4
prct=0.6

cd $cdir/src
python testNB.py $tr_ham_path $tr_spam_path $te_ham_path $te_spam_path $use_which $prct
