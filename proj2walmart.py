import bs4 
import requests
import lxml

# from selenium import webdriver
# driver = webdriver.Chrome(executable_path='P:\progies000\chromedriver.exe')
reviews=[]
ratings=[]

# # Testing for missing values 
# for i in range (5,70,5):        
#     hairreviews=[]
#     hairratings=[]
#     print(i)      
#     for k in range(i,i+5):
#         url= 'https://www.walmart.com/reviews/product/314022535?page='+str(k)
#         res= requests.get(url)
#         soup = bs4.BeautifulSoup(res.text,'lxml')
    
#         entries = soup.find_all('div', class_='review-text')
#         for i in range(len(entries)):
#             hairreviews.append(entries[i].text)
#         entry2 = soup.find_all(itemprop='ratingValue')
#         for j in range(len(entry2)):
#             hairratings.append(entry2[j].get('content'))    
#     print(len(hairreviews))
#     print(hairreviews[40])
#     print(len(hairratings))
#     print(hairratings[40])
# # 2-45 46-92 air pod https://www.walmart.com/reviews/product/604342441?page=
# # 2-62 hair https://www.walmart.com/reviews/product/734692345?page=
# # 2-27 vacuum https://www.walmart.com/reviews/product/506507232?page=
# # 2-4 6-41 42-71 tv https://www.walmart.com/reviews/product/314022535?page= 
    
    
    
# airpods
for k in range(2,45):
    url= 'https://www.walmart.com/reviews/product/604342441?page='+str(k)
    res= requests.get(url)
    soup = bs4.BeautifulSoup(res.text,'lxml')
    entries = soup.find_all('div', class_='review-text')
    for i in range(len(entries)):
        reviews.append(entries[i].text)
    entry2 = soup.find_all(itemprop='ratingValue')
    for j in range(len(entry2)):
        ratings.append(entry2[j].get('content'))
for k in range(46,92):
    url= 'https://www.walmart.com/reviews/product/604342441?page='+str(k)
    res= requests.get(url)
    soup = bs4.BeautifulSoup(res.text,'lxml')
    entries = soup.find_all('div', class_='review-text')
    for i in range(len(entries)):
        reviews.append(entries[i].text)
    entry2 = soup.find_all(itemprop='ratingValue')
    for j in range(len(entry2)):
        ratings.append(entry2[j].get('content'))
# revlon hiar
for k in range(2,62):
    url= 'https://www.walmart.com/reviews/product/734692345?page='+str(k)
    res= requests.get(url)
    soup = bs4.BeautifulSoup(res.text,'lxml')
    entries = soup.find_all('div', class_='review-text')
    for i in range(len(entries)):
        reviews.append(entries[i].text)
    entry2 = soup.find_all(itemprop='ratingValue')
    for j in range(len(entry2)):
        ratings.append(entry2[j].get('content'))
#vacuum 
for k in range(2,27):
    url= 'https://www.walmart.com/reviews/product/506507232?page='+str(k)
    res= requests.get(url)
    soup = bs4.BeautifulSoup(res.text,'lxml')
    entries = soup.find_all('div', class_='review-text')
    for i in range(len(entries)):
        reviews.append(entries[i].text)
    entry2 = soup.find_all(itemprop='ratingValue')
    for j in range(len(entry2)):
        ratings.append(entry2[j].get('content'))
# tv
for k in range(2,4):
    url= 'https://www.walmart.com/reviews/product/314022535?page='+str(k)
    res= requests.get(url)
    soup = bs4.BeautifulSoup(res.text,'lxml')
    entries = soup.find_all('div', class_='review-text')
    for i in range(len(entries)):
        reviews.append(entries[i].text)
    entry2 = soup.find_all(itemprop='ratingValue')
    for j in range(len(entry2)):
        ratings.append(entry2[j].get('content'))
        
for k in range(6,41):
    url= 'https://www.walmart.com/reviews/product/314022535?page='+str(k)
    res= requests.get(url)
    soup = bs4.BeautifulSoup(res.text,'lxml')
    entries = soup.find_all('div', class_='review-text')
    for i in range(len(entries)):
        reviews.append(entries[i].text)
    entry2 = soup.find_all(itemprop='ratingValue')
    for j in range(len(entry2)):
        ratings.append(entry2[j].get('content'))
for k in range(42,71):
    url= 'https://www.walmart.com/reviews/product/314022535?page='+str(k)
    res= requests.get(url)
    soup = bs4.BeautifulSoup(res.text,'lxml')
    entries = soup.find_all('div', class_='review-text')
    for i in range(len(entries)):
        reviews.append(entries[i].text)
    entry2 = soup.find_all(itemprop='ratingValue')
    for j in range(len(entry2)):
        ratings.append(entry2[j].get('content'))


print(len(reviews))
print(len(ratings))



import numpy as np
reviews=np.array(reviews)
ratings=np.array(ratings)

import pandas as pd



combo = pd.DataFrame([reviews.T,ratings.T])
combo = combo.T
combo.columns=['reviews','ratings']
creviewsDF=combo.loc[:,'reviews']
cratingsDF=combo.loc[:,'ratings']
fiftybins=[0,0,0,0,0,0,0]

creviewlengths=[]
for review in reviews:
    creviewlengths.append(len(review))
star1=combo.loc[combo['ratings']==str(1)]
star2=combo.loc[combo['ratings']==str(2)]
star3=combo.loc[combo['ratings']==str(3)]
star4=combo.loc[combo['ratings']==str(4)]
star5=combo.loc[combo['ratings']==str(5)]
cratingdist=[len(star1),len(star2),len(star3),len(star4),len(star5)]

target_names= ['1-Star','2-Star','3-Star','4-Star','5-Star']
import matplotlib.pyplot as plt
import seaborn as sb
plt.figure(figsize=(10,7))
sb.set_style(style='dark')
sb.boxplot(x=creviewlengths)
plt.xlabel('Number of characters In Review')

plt.title('Number of characters In Review')
plt.show()
    
plt.figure(figsize=(10,7))
sb.set_style(style='dark')
sb.barplot(x=target_names,y=cratingdist)
plt.xlabel('Number of Stars')
plt.ylabel('Number of Reviews')
plt.title('Ratings Distribution')
plt.show()



maxreviewlen=450
minreviewlen=40
overmin=combo[combo.reviews.str.len()>minreviewlen]
undermax=overmin[overmin.reviews.str.len()<maxreviewlen]
fixed=undermax.reset_index(drop=True)
notfive=fixed[fixed.ratings != '5']
five = fixed[fixed.ratings == '5']

reviewsDF=fixed.loc[:,'reviews']
ratingsDF=fixed.loc[:,'ratings']

reviewlengths=[]
for review in reviewsDF:
    reviewlengths.append(len(review))
star1=fixed.loc[fixed['ratings']==str(1)]
star2=fixed.loc[fixed['ratings']==str(2)]
star3=fixed.loc[fixed['ratings']==str(3)]
star4=fixed.loc[fixed['ratings']==str(4)]
star5=fixed.loc[fixed['ratings']==str(5)]
ratingdist=[len(star1),len(star2),len(star3),len(star4),len(star5)]

target_names= ['1-Star','2-Star','3-Star','4-Star','5-Star']
import matplotlib.pyplot as plt
import seaborn as sb
plt.figure(figsize=(10,7))
sb.set_style(style='dark')
sb.boxplot(x=reviewlengths)
plt.xlabel('Number of characters In Review')

plt.title('Number of characters In Review')
plt.show()
    
plt.figure(figsize=(10,7))
sb.set_style(style='dark')
sb.barplot(x=target_names,y=ratingdist)
plt.xlabel('Number of Stars')
plt.ylabel('Number of Reviews')
plt.title('Ratings Distribution')
plt.show()
evenMOREdata=notfive.append(five.head(500),ignore_index=True)
evenMOREdata=evenMOREdata.sample(frac=1).reset_index(drop=True)



ereviewsDF=evenMOREdata.loc[:,'reviews']
eratingsDF=evenMOREdata.loc[:,'ratings']
ereviewlengths=[]
for review in ereviewsDF:
    ereviewlengths.append(len(review))
estar1=evenMOREdata.loc[evenMOREdata['ratings']==str(1)]
estar2=evenMOREdata.loc[evenMOREdata['ratings']==str(2)]
estar3=evenMOREdata.loc[evenMOREdata['ratings']==str(3)]
estar4=evenMOREdata.loc[evenMOREdata['ratings']==str(4)]
estar5=evenMOREdata.loc[evenMOREdata['ratings']==str(5)]
eratingdist=[len(estar1),len(estar2),len(estar3),len(estar4),len(estar5)]

target_names= ['1-Star','2-Star','3-Star','4-Star','5-Star']
import matplotlib.pyplot as plt
import seaborn as sb
plt.figure(figsize=(10,7))
sb.set_style(style='dark')
sb.boxplot(x=ereviewlengths)
plt.xlabel('Number of characters In Review')

plt.title('Number of characters In Review')
plt.show()
    
plt.figure(figsize=(10,7))
sb.set_style(style='dark')
sb.barplot(x=target_names,y=eratingdist)
plt.xlabel('Number of Stars')
plt.ylabel('Number of Reviews')
plt.title('Ratings Distribution')
plt.show()







from sklearn.model_selection import train_test_split

xtrain, xtest,  ytrain, ytest = train_test_split(reviewsDF,ratingsDF,test_size=.3,random_state=15)
extrain, extest,  eytrain, eytest = train_test_split(ereviewsDF,eratingsDF,test_size=.3,random_state=15)
train=pd.DataFrame([xtrain,ytrain]).T
test=pd.DataFrame([xtest,ytest]).T
train=pd.get_dummies(train,columns=['ratings'])
test=pd.get_dummies(test,columns=['ratings'])
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(stop_words='english')
xtraindtm=cv.fit_transform(xtrain)
xdtm=cv.transform(ereviewsDF)
xtestdtm=cv.transform(xtest)
poo=cv.transform(extest)

cv2=CountVectorizer(stop_words='english')
extraindtm=cv2.fit_transform(extrain)
extestdtm=cv2.transform(extest)
trail=cv2.transform(xtrain)
traill=cv2.transform(xtest)
words=np.array(cv.get_feature_names())



# from sklearn.feature_extraction.text import TfidfTransformer
# tfid=TfidfTransformer()
# trainTFID=tfid.fit_transform(reviewvectors)
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


import seaborn as sb 

from sklearn.metrics import classification_report

nbs=[]
nbspred=[]
nbsacc=[]
yreals=[]
for i in range (1,6):
    name='ratings_'+str(i)
    nbs.append(MultinomialNB())
    yreals.append(test.loc[:,name])
    nbs[i-1].fit(xtraindtm,train.loc[:,name])
    nbspred.append(nbs[i-1].predict(xtestdtm))
    nbsacc.append(accuracy_score(test.loc[:,name],nbspred[i-1]))
    print('nb',i,': ',nbsacc[i-1])                         
probs=[]
for i in range (5):
    report = classification_report(yreals[i], nbspred[i])
    print('nb',i)
    print(report)
    print(confusion_matrix(yreals[i], nbspred[i]))
for n in nbs:
    probs.append(n.predict_proba(xtestdtm))

ensemblepred=[]
for j in range(len(probs[0])):
    maxx=0
    rating=0
    for i in range(5):
        if probs[i][j,1]>maxx:
            rating=str(i+1)
    ensemblepred.append(rating)
print('enesemblecoreeleated')
print(accuracy_score(ytest,ensemblepred))
print(confusion_matrix(ytest,ensemblepred))
print(classification_report(ytest,ensemblepred))

from sklearn.model_selection import GridSearchCV as gscv

nb=MultinomialNB()
nb.fit(xtraindtm,ytrain)
nbypred=nb.predict(xtestdtm)
print(accuracy_score(ytest,nbypred))
report2=classification_report(ytest,nbypred)
print(report2)
cmMNB=confusion_matrix(ytest,nbypred)

plt.figure(figsize=(10,7))
sb.heatmap(cmMNB,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for MultinominalNB')
plt.show()

nb2=MultinomialNB()
nb2.fit(extraindtm,eytrain)

mnbG=gscv(MultinomialNB(),
{},
cv=5,return_train_score=False)
mnbG.fit(xtraindtm,ytrain)
bestMNBgscv= mnbG.best_estimator_
ressmnbG = mnbG.cv_results_
resmnbG=pd.DataFrame(ressmnbG)
print('Best parameters for MNB: ',bestMNBgscv)
savegscv = resmnbG.to_csv('gscvMNBLESS.csv', index=True)



gb = GaussianNB()
gb.fit(xtraindtm.toarray(),ytrain)
gbypred=gb.predict(xtestdtm.toarray())
print(accuracy_score(ytest,gbypred))
cmGNB=confusion_matrix(ytest,gbypred)
print(classification_report(ytest,gbypred))
plt.figure(figsize=(10,7))
sb.heatmap(cmGNB,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for GaussianNB')
plt.show()
gnbG=gscv(GaussianNB(),
{},
cv=5,return_train_score=False)
gnbG.fit(xtraindtm.toarray(),ytrain)
bestGNBgscv= gnbG.best_estimator_
ressgnbG = gnbG.cv_results_
resgnbG=pd.DataFrame(ressgnbG)
print('Best parameters for GNB: ',bestGNBgscv)
savegscv = resgnbG.to_csv('gscvGNBLESS.csv', index=True)



from sklearn.linear_model import LogisticRegression
lg=LogisticRegression(multi_class='ovr',class_weight='none',solver='sag')
lg.fit(xtraindtm,ytrain)
lgpred=lg.predict(xtestdtm)
# print(lg)
# print(accuracy_score(ytest,lgpred))
cmLG=confusion_matrix(ytest,lgpred)
# print(classification_report(ytest,lgpred))
plt.figure(figsize=(10,7))
sb.heatmap(cmLG,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

lg2=LogisticRegression(multi_class='ovr',class_weight='none',solver='lbfgs')
lg2.fit(extraindtm,eytrain)

lgG=gscv( LogisticRegression(),
{'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'class_weight':['balanced','none'],'multi_class':['ovr','auto']},
cv=5,return_train_score=False)
lgG.fit(xdtm,eratingsDF)
bestLGgscv= lgG.best_estimator_
resslgG = lgG.cv_results_
reslgG=pd.DataFrame(resslgG)
print('Best parameters for LG: ',bestLGgscv)
savegscv = reslgG.to_csv('gscvLGLESS.csv', index=True)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
ld=lda(solver='svd')
ld.fit(xtraindtm.toarray(),ytrain)
ldypred=ld.predict(xtestdtm.toarray())
# print(accuracy_score(ytest,ldypred))
cmLDA=confusion_matrix(ytest,ldypred)
# print(classification_report(ytest,ldypred))
plt.figure(figsize=(10,7))
sb.heatmap(cmLDA,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for LDA')
plt.show()
knpred=[]
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,37,2):
    kn=KNeighborsClassifier(n_neighbors=i)
    kn.fit(xtraindtm,ytrain)
    knpred.append(accuracy_score(ytest,kn.predict(xtestdtm)))

plt.figure()
plt.plot(range(1,37,2),knpred)
plt.ylabel('number of neighbors')
plt.xlabel('accuarcy')
plt.title('Knearestneighbor accuracy')
plt.show()

kn=KNeighborsClassifier(n_neighbors=7,)
kn.fit(xtraindtm,ytrain)
knpred11=kn.predict(xtestdtm)
cmKN=confusion_matrix(ytest,knpred11)
# print(classification_report(ytest,knpred11))
plt.figure(figsize=(10,7))
sb.heatmap(cmKN,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for KNN')
plt.show()


from sklearn.ensemble import VotingClassifier

ens = VotingClassifier(estimators=[ ('nb' ,nb),('lg2',lg2),('ld',ld)]
                       ,voting='hard')
ens.fit(extraindtm.toarray(),eytrain)
ensypred= ens.predict(extestdtm)
print('voting')
print(accuracy_score(eytest,ensypred))
cmVEN=confusion_matrix(eytest,ensypred)
print(classification_report(eytest,ensypred))
plt.figure(figsize=(10,7))
sb.heatmap(cmVEN,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for ENSEMBLE NB LG LDA LESS')
plt.show()

from sklearn.ensemble import AdaBoostClassifier

ad=AdaBoostClassifier(base_estimator=nb)
ad.fit(extraindtm,eytrain)
adypred=ad.predict(extestdtm)
cmAD=confusion_matrix(eytest,adypred)
plt.figure(figsize=(10,7))
sb.heatmap(cmAD,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for AdaBoost LESS NB')
plt.show()
print(classification_report(eytest,adypred))

from sklearn.ensemble import BaggingClassifier
bc=BaggingClassifier(base_estimator=nb)
bc.fit(extraindtm.toarray(),eytrain)
bcypred=bc.predict(extestdtm.toarray())
cmBC=confusion_matrix(eytest,bcypred)
plt.figure(figsize=(10,7))
sb.heatmap(cmBC,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for  Bag LESS NB')
plt.show()
print(classification_report(eytest,bcypred))