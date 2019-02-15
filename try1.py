# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:03:00 2018

@author: NEERAJ
"""
import pandas as pd
import json
import re

file = open("C:/Users/NEERAJ/.spyder-py3/booksummaries.txt","r", encoding = "utf8")

print(file.readline())



genre = []
genre_count = [0] * 227
for line in file:
    components = line.split('\t')
    if len(components[5].strip()) != 0:
        dict = json.loads(components[5].strip())
    entry_genre = dict.values()
    for i in entry_genre:
        if i not in genre:
            genre.append(i)
            genre_count[genre.index(i)] = genre_count[genre.index(i)] +1
            #print(i)
        else:
            genre_count[genre.index(i)] = genre_count[genre.index(i)] + 1
            
df = pd.DataFrame( {"Genre" : genre, "Genre_count" : genre_count})

print(df)            


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    cleaned = cleaned.lower()
    return cleaned
   
filter_genre = df[df["Genre_count"] > 10]   
#print(filter_genre)
genre_transposed = filter_genre.transpose()
#print(genre_transposed)

genre_list = list(filter_genre["Genre"])
print(len(genre_list))

clean_genre_list = []

for genre in genre_list:
    clean_genre_list.append(cleanPunc(genre))
    
print(len(clean_genre_list))    
print(clean_genre_list)   


column_list = ["Wiki ID", "Name", "Summary"]
#print(column_list)

column_list1 =column_list + clean_genre_list
df1 = pd.DataFrame(columns = column_list1)
#print(column_list1)



file = open("C:/Users/NEERAJ/.spyder-py3/booksummaries.txt","r", encoding = "utf8")
for line in file:
    components = line.split('\t')
    list_append = []
    list_append.append(components[0].strip())
    list_append.append(components[2].strip())
    list_append.append(components[6].strip())
    genre_flag_list = [0] * len(genre_list)
    if len(components[5].strip()) != 0:
        dict = json.loads(components[5].strip())
        book_genre = dict.values()
        for j in book_genre:
            if j in genre_list: 
                genre_flag_list[genre_list.index(j)] = 1 
    list_append1 = list_append + genre_flag_list
    #print(len(genre_flag_list))
    #print(len(list_append1))
    #print(len(column_list1))
    #print(len(list_append))
    #print(list_append)
    row = pd.Series(list_append1,column_list1)
    #print(row)
    #print(row)
    df1 = df1.append([row],ignore_index=True)
    #print(df1)


print(df1.head())
print(df1.shape)


###################################################################################################################################
###############################################################  MODEL DEVELOPMENT ##################################################
############################################################################################################################

import re
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import ComplementNB
#from sklearn.naive_bayes import ComplementNB


from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns




#df1['Summary'] = df1['Summary'].map(lambda summ : clean_text(summ))


##########################Train Test split transformation Output Input separation####################################################

df1['Summary'] = df1['Summary'].apply(cleanPunc)


print(df1.head())

df1 = df1.infer_objects()
 
train, test = train_test_split(df1, random_state=42, test_size=0.33, shuffle=True)
X_train = train.Summary
X_test = test.Summary
print(X_train.shape)
print(X_test.shape)

Y_train = train.drop(labels = ['Wiki ID','Name','Summary'], axis =1)
Y_test = test.drop(labels = ['Wiki ID','Name','Summary'], axis =1)



######################################################################################################################################


####################################### Label Powerset Transformation #################################################
from skmultilearn.problem_transform import LabelPowerset

Y_LP_train = LabelPowerset().transform(Y_train)
#print(Y_LP_train)
Y_LP_test = LabelPowerset().transform(Y_test)

######################################################################################################################




###################### TFIDF Vectorizer##################################################################
vectorizer = TfidfVectorizer(stop_words=stop_words)
vectorizer.fit(X_train)
vectorizer.fit(X_test)
x_train_vect = vectorizer.transform(X_train)
x_test_vect = vectorizer.transform(X_test)
##########################################################################################################


multiclassifier = MultinomialNB()
multiclassifier.fit(x_train_vect,Y_LP_train)
multiclass_predict = multiclassifier.predict(x_test_vect)
print(Y_LP_test)
print(multiclass_predict)
print('Test accuracy {}'.format(accuracy_score(Y_LP_test,multiclass_predict)))
##############################Label Powerset classifier####################### 
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

LP_classifier = LabelPowerset(GaussianNB())

LP_classifier.fit(x_train_vect, Y_train)

LP_predictions = LP_classifier.predict(x_test_vect)

print('Test accuracy {}'.format(accuracy_score(Y_test,LP_prediction)))



#########################################################################################################




############################ using Naive Bayes pipeline###################################################
NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
prediction = pd.DataFrame().reindex_like(Y_test)                
for genre_category in clean_genre_list:
    print('... Processing {}'.format(genre_category))
    # train the model using X_dtm & y
    NB_pipeline.fit(X_train, train[genre_category])
    # compute the testing accuracy
    prediction[genre_category]  = NB_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(accuracy_score(test[genre_category], prediction[genre_category])))
 

print('Test accuracy {}'.format(accuracy_score(Y_test,prediction)))

########################################################################################################

########################## using Label Powerset pipeline ########################################
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
LP_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('lpclf',LabelPowerset(GaussianNB())),])


# train
LP_pipeline.fit(X_train, Y_train)

# predict
predictions = LP_pipeline.predict(X_test)

    
print('Test accuracy {}'.format(accuracy_score(Y_test,predictions)))
######################################################################################################
print(Y_test)

print(prediction)


#print( list(prediction["science fiction"]))

##################Evaluation Metrics####################################################################

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

return(TP, FP, TN, FN)

TPC = []
FPC = []
TNC = []
FNC = []
    
for genre_category in clean_genre_list:
     
#######################################################################################################    
"""    
    


    
from skmultilearn.problem_transform import BinaryRelevance

BR_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', BinaryRelevance(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
BR_pipeline.fit(X_train, Y_train)
predictions = BR_pipeline.predict(X_test)    
print("Accuracy = ",accura cy_score(Y_test,predictions))

" " " 
