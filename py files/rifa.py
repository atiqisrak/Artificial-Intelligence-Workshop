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
y=df.drop('class',1)
print(y.head(5))
z=df['class']

#column names
print(df.columns)

print(df.education.value_counts)s