---
layout: post
title: "titanic-survivors-analysis"
tags:
    - python
    - notebook
--- 
# Titanic Survivors Analysis
**Yacine Haddad, December 2017**

Back when I was living in London, I was attending a meetup called the Kaggle-
Dojo. A meetup about data science, machine learning. It was an occation to just
hangout with a bunch of pizza eateres while trying to solve some challanges
together. It was a fun and enjoyable encouter. When I moved to Geneva, I started
looking for similar meetups, and sadely anough, beside wine testing or (paid)
jugging meetups, there was not much meetup about data-science. With my friend
Vince, who also moved almost at the same time from London we decided to team
with a local pationate David and create our own mettup. For the first meeting we
decided to give a gentle introduction to data-science using python and what you
can do with a dataset. The following post is the tuturial I prepared.


On this example, I will cover some basics on pandas, numpy array and data-
visualisation.
I have been inspired by few resources listed below:
* https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish
* https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic
* https://www.kaggle.com/jasonm/large-families-not-good-for-survival

We start first by setting the environment and loading the dataset from
[Kaggle](https://www.kaggle.com/c/titanic#tutorials). Then explore the data
through various visualisations examples that will help us understand the dataset
that we are analysing and shape new variables to be used in the predictive
model.

1. [Dive into data](#Dive into data)
2. [Making predictive model](#Making a predictive model) 
 
## Dive into data 



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.style.use('physics')
%matplotlib inline
```
 
First, we want to know what is the fraction of females and males survived during
the disaster. For that we have two options, using data visualisation or inline
calculation: 



```python
# how is the fraction of survivors in each class and each class
sns.barplot(y="Pclass", x="Survived", hue="Sex",orient='h', data=data_train)
plt.show()
```

 
![png]({{ site.url }}/assets/notebooks/titanic-survivors-analysis_4_0.png) 

 
The Titanic has collected passengers in 3 different ports. It departed from
**Southampton (England)** and made two stops in **Cherbourg (France)** and
**Queenstown (Irland)**. We want to see if the port of embarkment has something
todo with the survival rate. We di then the same exercise as before, here we go
: 



```python
sns.barplot(y="Embarked", x="Survived", hue="Sex",orient='h', data=data_train)
plt.show()
```

 
![png]({{ site.url }}/assets/notebooks/titanic-survivors-analysis_6_0.png) 

