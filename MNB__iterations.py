"""
This script trains, tests, and outputs performance metrics for a Multinomial Naive Bayes (MNB)
machine learning classification of protein sequences. This algorithm supports multiple classes
but only one label per sample. That is, if you want to train a classifier on taxonomy, you should
use only one taxonomic level but you can use many different groups at that level. For example,
in Lavington, Zhao, and Duffy we used Genus as the taxonmonic level and trained the classifier on
14-19 Genera at once.

This script takes in protein coding sequence and a single label for each sequence then:
-transforms the sequence data into a sparse array of vectors
-randomly splits the data in half by class, such that each class is represented equally in the training and test sets
-trains a MNB classifier on the test data
-tests the trained classifier with the test set and outputs performance metrics
-repeats the split, train, test, metric output 'n' times

Input File should be in CSV format with the following column names:
Genus
Sequence

All other columns will be ignored and some performance output will be added to the data input file then
output as a new file.
Most of the performance metric will be output to separate files.

"""

#import required packages
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support as prfs

#--------Set variables here for your own use------------------------------------------------------------------------------------------

#Set ngram range, number of iterations, filenames
shortest=1 #shortest ngram, must be >=1
longest=12 #longest ngram, must be >= shortest
n=10000 #iterations
prefix="" #prefix to add to output filenames, underscore will be added for you
suffix="" #suffix to add to output filenames, underscore will be added for you

#read in data; header=0 uses the first row as column names; index_col=0 uses the first column as the index of the pandas array
#To set the working directory, if other than current, uncomment the next two lines and fill in: os.chdir("<your_working_directory")
#import os
#os.chdir("")
data=pd.read_csv("<your_data_file>",header=0,index_col=0)

#alternatively, generate your own pandas array with the labels column "Genus" and the sequences as "Sequence"
#please do not use lists or separate pandas iterators, the sort function on the pandas array is necessary (line 57)

#--------------------------------------------------------------------------------------------------------------------------------------

#_____________________Main Script______________________________________________________________________________________________________

#tidy up the prefix and suffix, if defined
if prefix:
	prefix+="_"
if suffix:
	suffix="_"+suffix


#sort the data by 
data.sort(columns='Genus',inplace=True)

if (shortest == 1) & (longest == 1):
	AAs=['a','c','d','e','f','g','h','i','k','l','m','n','p','q','r','s','t','v','w','y']
	cv=CountVectorizer(analyzer='char',ngram_range=(shortest,longest),vocabulary=AAs)
else:
	cv=CountVectorizer(analyzer='char',ngram_range=(shortest,longest))
   
#vectorize the text data  
X=cv.fit_transform(data['Sequence'])

#get the feature names
features=cv.get_feature_names()

#zero out the prediction counts
for i in data.Genus.unique():
    data[i]=0

#initialize feature use arrays
scores=pd.DataFrame(columns=("iteration","score"))
aa_top_10=pd.DataFrame(index=data.Genus.unique())
aa_next_10=pd.DataFrame(index=data.Genus.unique())
precisions=pd.DataFrame(columns=data.Genus.unique())
recalls=pd.DataFrame(columns=data.Genus.unique())
fbetas=pd.DataFrame(columns=data.Genus.unique())
micros=pd.DataFrame(columns=['Precision','Recall','Fbeta'])
macros=pd.DataFrame(columns=['Precision','Recall','Fbeta'])

#repeat n times
for j in range(n):
	
	randomstate=j+5342
	
    #initialize classifier and temporary DataFrame
	mnb=MultinomialNB()
	
	#build training and test data sets
	X_train,X_test,y_train,y_test=train_test_split(X,data.Genus,test_size=0.5,stratify=data.Genus,random_state=randomstate)
	
    #train the classifier
	mnb.fit(X_train,y_train)
	
    #make predictions for the original dataset
	y_pred=mnb.predict(X)
	score=mnb.score(X_test,y_test)
	scores.loc[j]=[j,score]
	data['prediction']=y_pred
	
	for c in mnb.classes_:
		data.loc[data.prediction==c,c]+=1
	
	for m in range(len(mnb.class_count_)):
		temp=np.argsort(mnb.coef_[m])[-20:]
		for e in temp[-10:]:
			if features[e] in aa_top_10.columns:
				aa_top_10[features[e]][mnb.classes_[m]]+=1
			else:
				aa_top_10[features[e]]=0
				aa_top_10[features[e]][mnb.classes_[m]]=1 
		for k in temp[:10]:
			if features[k] in aa_next_10.columns:
				aa_next_10[features[k]][mnb.classes_[m]]+=1
			else:
				aa_next_10[features[k]]=0
				aa_next_10[features[k]][mnb.classes_[m]]=1 
			
	metrics=prfs(data.Genus,y_pred)		
	#record simulation data
	precisions.loc[j]=metrics[0]
	recalls.loc[j]=metrics[1]
	fbetas.loc[j]=metrics[2]
	
	metrics=prfs(data.Genus,y_pred,average='micro')
	micros.loc[j,'Precision']=metrics[0]
	micros.loc[j,'Recall']=metrics[1]
	micros.loc[j,'Fbeta']=metrics[2]
	
	metrics=prfs(data.Genus,y_pred,average='macro')
	macros.loc[j,'Precision']=metrics[0]
	macros.loc[j,'Recall']=metrics[1]
	macros.loc[j,'Fbeta']=metrics[2]
	
#save the data	
data.to_csv("{0}MNB{1}.csv".format(prefix,suffix))
scores.to_csv("{0}MNB{1}.scores".format(prefix,suffix))
aa_top_10.to_csv("{0}MNB{1}_top10.features".format(prefix,suffix))
aa_next_10.to_csv("{0}MNB{1}_11-20.features".format(prefix,suffix))
precisions.to_csv("{0}MNB{1}.precision".format(prefix,suffix))
recalls.to_csv("{0}MNB{1}.recall".format(prefix,suffix))
fbetas.to_csv("{0}MNB{1}.fbeta".format(prefix,suffix))
micros.to_csv("{0}MNB{1}.micro".format(prefix,suffix))
macros.to_csv("{0}MNB{1}.macro".format(prefix,suffix))

quit()
			
