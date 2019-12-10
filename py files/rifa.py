# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.


print('Helo hlo')

#This is Dhim Dham Tana Na
n = 12
if (n<10):
         print('yaaaay')
         print('Boom Boom')
         
elif(n>10):
    print('Baam Baam Bole')
   """ 
'''This 
is 
multiple line 
comment

#User input
x = input()
if (x<10):
         print('yaaaay')
         print('Boom Boom')
         
elif(x>10):
    print('Baam Baam Bole')
    
else:
    print('Get your Bum Bum out of here.')

    
import numpy as np
a = 1
b = 2
c = np.add(a,b)
    
    '''
    
import numpy as np    
import pandas as pd
    
df = pd.read_csv('C:\\Users\\rifaz\\Desktop\\py files\\af.csv')
    
#print some data
print(df.head(5))    
print(df.tail(5))     

#types
print(df['class'].value_counts()) 

#rename
df['class'] = [1 if x == '<=50K' else 0 for x in df['class']]

print(df['class'].value_counts())

#droping column 0-> rows 1->column
X=df.drop('class',1)
print(y.head(5))
Y=df['class']

#column names
print(df.columns)

#how many data in each category of education column
print(df.education.value_counts)

#oe hot encoding
df['education']=pd.get_dummies(X['education'])

print(pd.get_dummies(X['education']).head(5))

#How many uique categories
len(X['education'].unique())
print(X.colums)
for col_name in X.columns:
    if(X[col_name].dtypes == 'object'):
        unique_cat = len(X[col_name].unique())
        #print()
        print(col_name,unique_cat)
        
X['native-country'].value_counts().sort_values(ascending=False).head(10)
X['native-country']=['United-States']=['United States' if x=='United-States' else 'others' for x in X['native-country']]
todummy_list=['workclass','education','marial-status','occupation','relationship','race','sex','native-country']

def dummy_df(df,todummy_list):
    for x in todummy_list:
        dummies =pd.get_dummies=pd.get_dummies(df[x],prefix=x,dummy_na =False)
        df = df.drop(x,1)
        df = pd.concat([df,dummies],axis=1)
        return  df

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        