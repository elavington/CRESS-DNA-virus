"""
This script trains, tests, and outputs performance metrics for a C-Support Vector Machine (SVC)
classification of protein sequences. This algorithm supports multiple classes
but only one label per sample. That is, if you want to train a classifier on taxonomy, you should
use only one taxonomic level but you can use many different groups at that level. For example,
in Lavington, Zhao, and Duffy we used Genus as the taxonmonic level and trained the classifier on
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

#import required packages
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as prfs


def permutations(data, shortest, longest, iterations, kernel, prefix, suffix):

	#initialize text data vectorizer
	if (shortest == 1) & (longest == 1):
		AAs=['a','c','d','e','f','g','h','i','k','l','m','n','p','q','r','s','t','v','w','y']
		cv=CountVectorizer(analyzer='char',ngram_range=(shortest,longest),vocabulary=AAs)
	else:
		cv=CountVectorizer(analyzer='char',ngram_range=(shortest,longest))
	
	dataVect=cv.transform(data.Sequence)
	
	# repeat for each kernel type:
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

		#repeat n times
		for j in range(iterations):
		
			randomstate=j+5342 #make sure the randomstate input is at least 4 digits and repeatable
		
			#initialize classifier
			#uses default settings for the classifier, see scikit-learn documentation sklearn.svm.SVC for details 
			clf=SVC(kernel=k)
  
			#build training and test data sets
			X_train,X_test,y_train,y_test=train_test_split(
				dataVect,data.Genus,test_size=0.5,stratify=data.Genus,random_state=randomstate)
		
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
			
			#increment the Genus column counter for each Genus 
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
	
		data.to_csv("{0}_SVM_{1}_{2}.csv".format(prefix,k,suffix))
		scores.to_csv("{0}_SVM_{1}_{2}.scores".format(prefix,k,suffix))	
		precisions.to_csv("{0}_SVM_{1}{2}.precision".format(prefix,k,suffix))
		recalls.to_csv("{0}_SVM_{1}_{2}.recall".format(prefix,k,suffix))
		fbetas.to_csv("{0}_SVM_{1}_{2}.fbeta".format(prefix,k,suffix))
		micros.to_csv("{0}_SVM_{1}_{2}.micro".format(prefix,k,suffix))
		macros.to_csv("{0}_SVM_{1}_{2}.macro".format(prefix,k,suffix))


def Get_ngram_length(s=0, l=0):
	while True:#check to make sure the user input is an integer
		try:
			s = int(input("Shortest n-gram: "))
			break #first input is an integer, exit the loop
		except ValueError:
			print("Please enter n-gram range as integers. To choose only one n-gram length,\nplease enter that length as both shortest and longest.")
			return s,l #returns 0,0
			
	while True:#check to make sure the user input is an integer
		try:
			l = int(input("Longest n-gram: "))
			break #first input is an integer, exit the loop
		except ValueError:
			print("Please enter n-gram range as integers. To choose only one n-gram length,\nplease enter that length as both shortest and longest.")
			return s,l #returns 0,0
			
	if (s > l) & (s != l): #check if the user input is as expected: shorter is smaller than or equal to longer 
		print("Please make sure that the shorter n-gram length is smaller than the longer length")
		
	return s,l #return the user input integers

def Get_OutputName():
	happy='n'
	while 'Y' not in happy.upper():
		prefix = input("Please enter an output name prefix (required): ")
		suffix = input("Please enter an output name suffix (optional): ")
		happy = input("Are you happy with this output filename template (Y/N)? {0}_SVM_{1}_{2}.csv".format(prefix,'kernel',suffix))
	return prefix, suffix

def Get_iterations(i=0):
	while True:
		try:
			i = int(input("Number of permutations to run: "))
			break
		except ValueError:
			print("Please enter iterations value as an integer.")
			return i
	return i
	
def Get_kernel():
		
	def confused(input):
		print("\nSorry, I don't quite understand {0}. Please double check that you match one of the inputs in the parentheses.".format(input))
	
	both = input("Do you want to run both Linear and RBF SVM kernels? (Y/N): ")
	
	if 'Y' in both.upper():
		kernel = ['linear', 'rbf']
		return kernel
		
	elif 'N' in both.upper():
		happy = 'n'
		while 'Y' not in happy.upper():
			kernel = input("Which kernel do you want to run, Linear or RBF SVM (L/R)? ")
			if 'L' in kernel.upper():
				happy = input("Running Linear SVM only, okay? (Y/N): ")
			elif 'R' in kernel.upper():
				happy = input("Running RBF SVM only, okay? (Y/N): ")	
			else:
				happy = 'n'
				confused(kernel)
		if 'L' in kernel.upper():
				kernel = ['linear']
				return kernel
				
		elif 'R' in kernel.upper():
				kernel = ['rbf']
				return kernel
				
	else:
		confused(both)
		kernel=[]
		return kernel

def main():	
	
	#Ask for the data file path, open the file, and sort on the Genus Column
	file_path = input("Enter the path of your file: ")
	assert os.path.exists(file_path), "I did not find the file at, "+file_path+"\nPlease check spelling, that you are using any necessary slashes ( / ), and that you are including the file extension."
	data=pd.read_csv(file_path,header=0)
	
	#sort the data by Genus
	if ('Genus' in data.columns) and ('Sequence' in data.columns):
		print("Found required columns in data file "+str(file_path))
		data.sort_values(by='Genus',inplace=True)
	else:
		print("Could not find one or both required columns (Genus, Sequence) in the data file.")
		quit()
	
	#Ask for the n-gram range
	shortest, longest = 0,0
	while ((shortest < 1) | (longest < 1)) | (shortest > longest):
		shortest, longest = Get_ngram_length()
	
	#Ask for the SVC kernel(s)
	kernel=[]
	while len(kernel) == 0:
		kernel=Get_kernel()
	
	#Ask for the number of iterations
	n = 0
	while n == 0:
		n = Get_iterations()
	
	#Ask for the output file prefix and suffix
	prefix, suffix = Get_OutputName()
	
	#Run the cross validation
	permutations(data=data, shortest=shortest, longest=longest, iterations=n, kernel=kernel, prefix=prefix, suffix=suffix)
			
		
if __name__ == "__main__":
	main()


	

