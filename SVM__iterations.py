"""
This script trains, tests, and outputs performance metrics for a C-Support Vector Machine (SVC)
classification of protein sequences. This algorithm supports multiple classes
but only one label per sample. That is, if you want to train a classifier on taxonomy, you should
use only one taxonomic level but you can use many different groups at that level. For example,
in Lavington, Zhang, and Duffy we used Genus as the taxonmonic level and trained the classifier on
14-19 Genera at once.

This script takes in protein coding sequence and a single label for each sequence then:
-transforms the sequence data into a sparse array of vectors
-randomly splits the data in half by class, such that each class is represented equally in the training and test sets
-fits and transforms the data so that training and test set feature vectors are on the same scale
-trains a SVC classifier on the test data
-tests the trained classifier with the test set and outputs performance metrics
-repeats the split, train, test, metric output 'n' times

Input File should be in CSV format with the following column names:
Genus
Sequence

All other columns will be ignored and some performance output will be added to the data input file then
output as a new file.
Most of the performance metric will be output to separate files.

"""
#for local use, known classifications are in the last column, first line is a header
#each row is a sample, each column (except the last) is a feature
#character features have been transformed to numeric values BEFORE using this code
#[i.e. all data must be numeric]

#import required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as prfs

#--------Set variables here for your own use------------------------------------------------------------------------------------------

#Set ngram range, number of iterations, filenames
shortest=1 #shortest ngram, must be >=1
longest=12 #longest ngram, must be >= shortest
n=10000 #iterations
prefix="" #prefix to add to output filenames, underscore will be added for you
suffix="" #suffix to add to output filenames, underscore will be added for you

kernel=['linear','rbf']
 
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
	
#initialize text data vectorizer
if (shortest == 1) & (longest == 1):
	AAs=['a','c','d','e','f','g','h','i','k','l','m','n','p','q','r','s','t','v','w','y']
	cv=CountVectorizer(analyzer='char',ngram_range=(shortest,longest),vocabulary=AAs)
else:
	cv=CountVectorizer(analyzer='char',ngram_range=(shortest,longest))
	
dataVect=cv.transform(data.sequence)

	
#repeat 12 times for each explicit ngram length of 1-12
for k in kernel:

	#initialize feature use arrays
	precisions=pd.DataFrame(columns=data.Genus.unique())
	recalls=pd.DataFrame(columns=data.Genus.unique())
	fbetas=pd.DataFrame(columns=data.Genus.unique())
	micros=pd.DataFrame(columns=['Precision','Recall','Fbeta'])
	macros=pd.DataFrame(columns=['Precision','Recall','Fbeta'])
	scores=pd.DataFrame(columns=("iteration","score"))
	
	#initialize the genus classification for this ngram
	for i in data.Genus.unique():
		data[i]=0

		#repeat n times for each ngram
	for j in range(n):
		
		randomstate=j+5342
		
		#initialize classifier
		#uses default settings for the classifier, see scikit-learn documentation sklearn.svm.SVC for details 
		clf=SVC(kernel=k)
  
		#build training and test data sets
		X_train,X_test,y_train,y_test=train_test_split(dataVect,data.Genus,test_size=0.5,stratify=data.Genus,random_state=randomstate)
		
		#Scale the data to the training set
		StSc=StandardScaler(copy=True,with_mean=False)
		StSc.fit(X_train)
		X_sc_train=StSc.transform(X_train)
		X_sc_test=StSc.transform(X_test)
		X_scaled=StSc.transform(dataVect)
		
		#train the classifier
		clf.fit(X_sc_train,y_train)
   
		#make predictions for the original dataset
		y_pred=clf.predict(X_scaled)
		score=clf.score(X_sc_test,y_test)
		scores.loc[j]=[j,score]
		data['prediction']=y_pred
		
		for c in clf.classes_:
			data.loc[data.prediction==c,c]+=1
			
		#record simulation data
		metrics=prfs(data.Genus,y_pred)
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
	
	data.to_csv("{0}SVM_{1}_{2}.csv".format(prefix,k,suffix))
	scores.to_csv("{0}SVM_{1}_{2}.scores".format(prefix,k,suffix))	
	precisions.to_csv("{0}SVM_{1}_{2}.precision".format(prefix,k,suffix))
	recalls.to_csv("{0}SVM_{1}_{2}.recall".format(prefix,k,suffix))
	fbetas.to_csv("{0}SVM_{1}_{2}.fbeta".format(prefix,k,suffix))
	micros.to_csv("{0}SVM_{1}_{2}.micro".format(prefix,k,suffix))
	macros.to_csv("{0}SVM_{1}_{2}.macro".format(prefix,k,suffix))

quit()