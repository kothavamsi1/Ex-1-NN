<H3>ENTER YOUR NAME: KOTHA VAMSI
<H3>ENTER YOUR REGISTER NO: 212222040081
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
Developed by:Preetha.S
Register no :212222230110

import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))
```


## OUTPUT:

## DATASET:

![image](https://github.com/Preetha-Senthamilan/Ex-1-NN/assets/119390282/2dfc1641-b24e-4156-a8d1-df18d8460e43)

X VALUES:

![image](https://github.com/Preetha-Senthamilan/Ex-1-NN/assets/119390282/edb69d82-0e6a-4d0f-ab38-6ed5f25d388d)

Y VALUES:

![image](https://github.com/Preetha-Senthamilan/Ex-1-NN/assets/119390282/f95f53c9-b209-40de-9587-0fb3018dea5c)

NULL VALUES:

![image](https://github.com/Preetha-Senthamilan/Ex-1-NN/assets/119390282/946da728-2c72-470b-bfe6-06b94c42e77e)

DUPLICATED VALUES:

![image](https://github.com/Preetha-Senthamilan/Ex-1-NN/assets/119390282/f89fe104-7abc-46a0-979c-9d1d2c24e817)


DESCRIPTION:

![image](https://github.com/Preetha-Senthamilan/Ex-1-NN/assets/119390282/58a0f531-3c53-4319-a8d6-ab7b1f866846)


NORMALIZED DATASET:

![image](https://github.com/Preetha-Senthamilan/Ex-1-NN/assets/119390282/2c7b171d-c312-4404-8cab-9af1e1721856)


TRAINING DATA

![image](https://github.com/Preetha-Senthamilan/Ex-1-NN/assets/119390282/b0b85b94-dad2-411e-ad0a-f7d5ba55555a)


TESTING DATA:

![image](https://github.com/Preetha-Senthamilan/Ex-1-NN/assets/119390282/8455f181-eac8-47a3-b934-4b1ee4f0197a)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


