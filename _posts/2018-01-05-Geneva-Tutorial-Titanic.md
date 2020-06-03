---
layout: post
title: "Welcome to Lagrange!"
author: "Yacine Haddad"
categories: journal
tags: [ML, tutorial]
image: 01-titanic.jpg 
---

This notebook is a gentle introduction to data analysis in python step-by-step. Starting from raw data to making prediction model on the Titanic Survivors dataset.

On this example, I will cover some basics on pandas, `numpy` array and data-visualisation.
I have based this example on few resources that are listed below:
* https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish
* https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic
* https://www.kaggle.com/jasonm/large-families-not-good-for-survival

We start first by setting the environment and loading the dataset from Kaggle website aka [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic#tutorials). Then explore the data through various visualisation examples that will help us understand the dataset that we are analysing and shape new variables to be used in the predictive model. 

1. [Dive into data](#Dave into datal)
    1. [What story data tell us ?](## What story data tell us ?)
    2. [Building new features](## Building new features )
    3. [Dealing with missing data](## Dealing with missing data)
2. [Making predictive model](#Making predictive model)
    1. [What model to choose ?](## What model to choose ?)
    2. [Fine tune my model ](## Fine tune my model )
3. [Make submission](# Make submission)

## Dive into data

![.](http://s2.quickmeme.com/img/56/566939e4d16f26f06f1b648b74270c264240bb66a4d778e53c3f85f84ab3976c.jpg)

First things first! If you want to be a data scientist, then you need to get some data and dive in it. Kaggle is probably the place to find some data to play with an get started as an analyst. As I am using python, you need to import your exploration kit that will allow you to load, visualise and, most importantly, understand your data. For this tutorial, we ill import numpy and pandas for data manipulation, matplotlib and seaborn for data visualisation: 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-dark-palette')
%matplotlib inline
```

The Titanic challenge comes with a training and validation datasets. The first can be used to understand the content of the data and train a model to make a prediction, and then the second can be used to test the model that we will build. 

``Pandas`` has a built-in function to read and load CSV files and load the data directly into a ``Pandas`` dataframe.


```python
# loading the dataset into pandas dataframes
data_train = pd.read_csv('../input/train.csv')
data_valid = pd.read_csv('../input/test.csv')
```

> Sometimes, if you can, you can open the data files diretly and look into the data. This will allow you to spot any requirement to read the data, for example in case where the columns are separated by `;` instead of `,`. You can also add columns names in there are not present in the datafile... etc


## What story data tell us ?
When get hands to data for the first time, it good to know the origin and the context of the data, any information that might come with data is can be useful to build your model. Kaggle page for this dataset (that you can find  [here](http://https://www.kaggle.com/c/titanic/data)), is quite illuminating:

| Variable        | Definition           | 
| ------------- |:-------------:| 
| survival    | Survival	0 = No, 1 = Yes | 
| pclass	  | Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd|
| sex	        | Sex	|
| Age 	      | Age in years	|
| sibsp	     | # of siblings / spouses aboard the Titanic	|
| parch	    | # of parents / children aboard the Titanic	|
| ticket 	| Ticket number	|
| fare	 |Passenger fare	|
| cabin	|Cabin number	|
| embarked|	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton|

Ah ha! Now, our data make more sens, isn't it ? It a good practice to see what the data contains, and `describe()` and `dead` are the best `pandas` functions to do that:


```python
# return first 5 rows in the dataframe
data_train.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# print a dcription of the dataset
data_train.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



From the previous outputs,  we can tell the nature of the data. each row represent a passenger and the columns represent some characteristic that describe the passenger, such as the ticket fare, Age, Name... etc.  We also can tell that the dataset has some missing values, for example the column `Age` has only 714 rows instead of 891, which means that 177 values are missing.  In the column Cabin, we can see that may passenger, with no or unknown cabin. We will see later how to deal with missing data later, and for now let visualise our data. 

We wanna see, if the parity is respected during the disaster, and therefore can look to the fraction of females and males survived during the disaster.
> *Such investigation can be done in two ways, numerical or graphical. But you know probably that saying: A picture is worth a thousand words, so option 2 will be ;) *


```python
data_train.Sex[data_train.Survived==1].hist(bins=2,range=[-0.5,1.5], alpha=0.5, normed=1, label='survived')
data_train.Sex[data_train.Survived==0].hist(bins=2,range=[-0.5,1.5], alpha=0.5, normed=1, label='sinked')
plt.legend()
plt.show()
```


![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_10_0.png)


Wow! It is clear, in 2012, that rule of "children and **women** first" is well respected and this histogram is the proof. more than 65% of the females survived against roughly 30% of males. This is just an example of what a histogram can tell us. 
> *You can notice here that the historgrams are normalised. The reason is we want to have instead of a simple count, the survival rate for each sex.*

Now, let use Seaborn, a wonderful tool that allows us to make similar histograms, by combining 2 or 3 features, in only one line of code. For example, we want to see the survival rate by ticket class for male and females. This can be achieved by calling `barplots` as such:


```python
# how is the fraction of survivors in each class and each class
sns.barplot(y="Pclass", x="Survived", hue="Sex",orient='h', data=data_train)
plt.show()
```


![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_12_0.png)


The Titanic has collected passengers in 3 different ports. It departed from **Southampton (England)** and made two stops in **Cherbourg (France)** and **Queenstown (Irland)**. We want to see if the port of embarkment has something todo with the survival rate. We did then the same exercise as before, here we go:


```python
sns.barplot(y="Embarked", x="Survived", hue="Sex",orient='h', data=data_train)
plt.show()
```


![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_14_0.png)


Obviously, the survival rate seems unbalanced! it seems like the French passengers had more chances of survival. How is this possible? We will try to find out later what secret hides behind this observation.

We want to simplify the variable age to include ages tranches. We could then define a function that does that for us:


```python
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df['Age_category'] = categories
    return df

data_train = simplify_ages(data_train)
data_valid = simplify_ages(data_valid)
```


```python
plt.figure(figsize=(5,8))
sns.barplot(x="Survived", y="Age_category", hue="Sex", orient='h', data=data_train)
plt.show()
```


![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_17_0.png)


Following the same logic, we can also extract information about the cabin, and fill missing data with a "None" variable. We could also extract the information about the title of the passenger, this could be helpful when we are going to building a prediction model.


```python
import re
data_train['Title'] = data_train.Name.apply(lambda x: re.sub("(.*, )|(\\..*)", "", x))
data_train.groupby(['Title','Sex']).size()
```




    Title         Sex   
    Capt          male        1
    Col           male        2
    Don           male        1
    Dr            female      1
                  male        6
    Jonkheer      male        1
    Lady          female      1
    Major         male        2
    Master        male       40
    Miss          female    182
    Mlle          female      2
    Mme           female      1
    Mr            male      517
    Mrs           female    125
    Ms            female      1
    Rev           male        6
    Sir           male        1
    the Countess  female      1
    dtype: int64



For our curiosity, Let check some of the personalities on board, and learn some information about them. 
We can see that on this boat there was a woman with the title "the Countess", she is Noël Leslie, Countess of Rothes. Following Wikipedia [https://en.wikipedia.org/wiki/Noël_Leslie,_Countess_of_Rothes](https://en.wikipedia.org/wiki/Noël_Leslie,_Countess_of_Rothes) she was a heroine of the Titanic disaster:
> ... famous for taking the tiller of her lifeboat and later helping row the craft to the safety of the rescue ship Carpathia.



```python
plt.figure(figsize=(5,8))
sns.barplot(y="Title", x="Survived", hue="Sex", orient='h', data=data_train)
plt.show()
```


![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_21_0.png)



```python
def simplify_titles(df):
    df['Title'] = df.Name.apply(lambda x: re.sub("(.*, )|(\\..*)", "", x))
    df['Title'].value_counts()
    rare_title = np.array(['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'])

    # Also reassign mlle, ms, and mme accordingly
    df.iloc[df.Title.values == 'Mlle' ,df.columns.get_loc('Title')] = 'Miss' 
    df.iloc[df.Title.values == 'Ms'   ,df.columns.get_loc('Title')] = 'Miss'
    df.iloc[df.Title.values == 'Mme'  ,df.columns.get_loc('Title')] = 'Mrs' 
    df.iloc[df.Title.isin(rare_title).values ,df.columns.get_loc('Title')] = 'Rare Title'
    return df
data_train = simplify_titles(data_train)
data_valid = simplify_titles(data_valid)
```


```python
sns.barplot(y="Title", x="Survived", orient='h', data=data_train)
plt.show()
```


![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_23_0.png)



```python
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df['Fare_category'] = categories
    return df
    
def drop_features(df):
     return df.drop(['Ticket', 'Name'], axis=1)
    
    
def simplify_embarked(df):
    df.Embarked = df.Embarked.fillna('N')
    return df

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = simplify_titles(df)
    df = simplify_embarked(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_valid = transform_features(data_valid)
data_train.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_category</th>
      <th>Title</th>
      <th>Fare_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>N</td>
      <td>S</td>
      <td>Student</td>
      <td>Mr</td>
      <td>1_quartile</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>C</td>
      <td>Adult</td>
      <td>Mrs</td>
      <td>4_quartile</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>N</td>
      <td>S</td>
      <td>Young Adult</td>
      <td>Miss</td>
      <td>1_quartile</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>C</td>
      <td>S</td>
      <td>Young Adult</td>
      <td>Mrs</td>
      <td>4_quartile</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>N</td>
      <td>S</td>
      <td>Young Adult</td>
      <td>Mr</td>
      <td>2_quartile</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(5,8))
sns.barplot(y="Cabin", x="Survived", hue="Sex", orient='h', data=data_train)
plt.show()
```


![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_25_0.png)



```python
plt.figure(figsize=(5,6))
sns.barplot(y="Fare_category", x="Survived", hue="Sex",orient='h', data=data_train)
plt.show()
```


![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_26_0.png)


### Does a family size matters ?


```python
data_train['Fsize'] = data_train.SibSp + data_train.Parch + 1
data_valid['Fsize'] = data_valid.SibSp + data_valid.Parch + 1
```


```python
plt.figure(figsize=(5,6))
sns.barplot(y="Fsize", x="Survived",orient='h', data=data_train)
plt.axvline(0.5, ls='--')
```




    <matplotlib.lines.Line2D at 0x111cb5d50>




![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_29_1.png)



```python
plt.figure(figsize=(5,5))
sns.countplot(x="Fsize", hue="Survived", data=data_train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x111a61e10>




![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_30_1.png)



```python
data_train.Age[data_train.Survived==1].hist(bins=80,range=[0,80], alpha=0.5, normed=1, label='survived')
data_train.Age[data_train.Survived==0].hist(bins=80,range=[0,80], alpha=0.5, normed=1, label='sinked')
plt.legend()
plt.show()
```


![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_31_0.png)



```python
data_train.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_category</th>
      <th>Title</th>
      <th>Fare_category</th>
      <th>Fsize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>N</td>
      <td>S</td>
      <td>Student</td>
      <td>Mr</td>
      <td>1_quartile</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>C</td>
      <td>Adult</td>
      <td>Mrs</td>
      <td>4_quartile</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>N</td>
      <td>S</td>
      <td>Young Adult</td>
      <td>Miss</td>
      <td>1_quartile</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_valid.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_category</th>
      <th>Title</th>
      <th>Fare_category</th>
      <th>Fsize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>N</td>
      <td>Q</td>
      <td>Young Adult</td>
      <td>Mr</td>
      <td>1_quartile</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>N</td>
      <td>S</td>
      <td>Adult</td>
      <td>Mrs</td>
      <td>1_quartile</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>N</td>
      <td>Q</td>
      <td>Senior</td>
      <td>Mr</td>
      <td>2_quartile</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Making predictive model


```python
from sklearn import preprocessing
def encode_features(df_train, df_test):
    str_features = [
        'Sex', 'Cabin', 'Embarked', 'Age_category', 'Title', 'Fare_category'
    ]
    df_combined = pd.concat([df_train[str_features], df_test[str_features]])
    
    for feature in str_features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_valid = encode_features(data_train, data_valid)
data_train.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_category</th>
      <th>Title</th>
      <th>Fare_category</th>
      <th>Fsize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>7</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>7</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>7</td>
      <td>3</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 13 , 13 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

plot_correlation_map( data_train )
```


![png](/assets/ipynb/Geneva-Tutorial-Titanic_files/Geneva-Tutorial-Titanic_36_0.png)



```python
from sklearn.model_selection import train_test_split, cross_val_score

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

num_test = 0.30
X_train, X_valid, y_train, y_valid = train_test_split(
    X_all, y_all, 
    test_size=num_test, 
    random_state=23
)
```


```python
X_train.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_category</th>
      <th>Title</th>
      <th>Fare_category</th>
      <th>Fsize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>387</th>
      <td>2</td>
      <td>0</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>603</th>
      <td>3</td>
      <td>1</td>
      <td>44.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>691</th>
      <td>3</td>
      <td>0</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>13.4167</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
```


```python
clfs = {
    'LogisticRegression'         : LogisticRegression(),
    "RandomForestClassifier"     : RandomForestClassifier(),
    "GradientBoostingClassifier" : GradientBoostingClassifier()
}
```


```python
for i, clf in clfs.items():
    clf.fit(X_train, y_train)
```


```python
from sklearn.metrics import make_scorer, accuracy_score
```


```python
from sklearn import cross_validation
for i, clf in clfs.items():
    cv_score = cross_validation.cross_val_score(clf,X_all,y_all, cv=5, scoring='accuracy')
    print("%30s CV Score : Mean - %.3g +\- %.4g (Min - %.3g, Max - %.3g)" % (
        i, np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)
    ))
```

    /usr/local/lib/python2.7/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)


                LogisticRegression CV Score : Mean - 0.786 +\- 0.01672 (Min - 0.764, Max - 0.808)
        GradientBoostingClassifier CV Score : Mean - 0.841 +\- 0.01686 (Min - 0.815, Max - 0.865)
            RandomForestClassifier CV Score : Mean - 0.81 +\- 0.01962 (Min - 0.781, Max - 0.837)


### Fine tune classifier parameters


```python
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth'         : range(2,15,2), 
              'min_samples_split' : np.linspace(0.001,0.01,6),
              'min_samples_leaf'  : range(1,10,2),
              'subsample'         : [0.7,0.8,0.9],
             }

grid_obj = GridSearchCV(clfs['GradientBoostingClassifier'], 
                        parameters, 
                        cv=5, 
                        scoring='accuracy',
                        n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

cv_score = cross_validation.cross_val_score(grid_obj.best_estimator_, 
                                            X_all, y_all, cv=5, 
                                            scoring='accuracy')

print (grid_obj.best_params_)
print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (
    np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)
))
```

    {'min_samples_split': 0.001, 'subsample': 0.7, 'max_depth': 2, 'min_samples_leaf': 5}
    CV Score : Mean - 0.8328 | Std - 0.01057 | Min - 0.8156 | Max - 0.8436


Run everything and go make yourself  a tea or a coffe, this might take a while. 
When the grid search finiches, it returns the parameters that gives better accuracy and for instance, for our case, the best parameters are :
```
{'max_depth': 2, 'min_samples_leaf': 7, 'min_samples_split': 0.01, 'subsample': 0.8}
```


## Make predictions and submission


```python
submit = grid_obj.best_estimator_.predict(data_valid.drop(['PassengerId'], axis=1))
```


```python
df = pd.DataFrame()
df['PassengerId'] = data_valid['PassengerId']
df['Survived'] = submit
```


```python
df.to_csv('submission.csv',index=False)
```


```python
df.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


