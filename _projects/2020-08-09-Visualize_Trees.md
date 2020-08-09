---
layout: project
title: Visualize Tree Based Mdels
description: How to Visualize Tree Based Models in Python
category: Interpretability
---

Going to show how to interpret the results of a decision tree.  

First I am going to load in the default dataset.  I will be using `student`, `balance`, and `income` to predict `default`.  


```python
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/sik-flow/datasets/master/Default.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>student</th>
      <th>balance</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>729.526495</td>
      <td>44361.625074</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>817.180407</td>
      <td>12106.134700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>529.250605</td>
      <td>35704.493935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>785.655883</td>
      <td>38463.495879</td>
    </tr>
  </tbody>
</table>
</div>



Fit a decision tree model to the data.  I am not going to tune the hyperparameters (but would advise this as decision trees have a tendency to overfit).  


```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(df.drop('default', axis = 1), df['default']);
```

Now lets visualize what our decision tree looks like. 


```python
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,8), dpi=300)
tree.plot_tree(dt);
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Visualize_Trees_files/Visualize_Trees_5_0.png)


We see that the decision tree has so many nodes that it is not readable.  I am going to set the max depth, so I can only see the top 2 layers of the tree.  The top layers are typically the most important splits in the data.  


```python
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,8), dpi=300)
tree.plot_tree(dt, max_depth = 2);
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Visualize_Trees_files/Visualize_Trees_7_0.png)


It is readable now, but it would be nice if we had the column labels instead of `X[1]`, `X[2]`, and `X[3]`.  I am also going to be saving the image as a png. 


```python
fn=['student', 'balance', 'income']
cn=['not-default', 'default']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dt,
               feature_names = fn, 
               class_names=cn,
               filled = True, max_depth = 2);
fig.savefig('tree.png')
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Visualize_Trees_files/Visualize_Trees_9_0.png)


Now I am going to show how to see the feature importances.  The feature importances tells us how each feature improved the purity of each node.  These are normalized to a 100 scale. 


```python
plt.bar(df.drop('default', axis = 1).columns, dt.feature_importances_)
plt.title('Feature Importance');
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Visualize_Trees_files/Visualize_Trees_11_0.png)


This means that `balance` accounted for about 70% of the improvement to purity, `income` accounted for about 30% of the improvement to purity and `student` accounted for less than 1%. 

Now I am going to fit a random forest classifier and plot out some trees and look at the feature importance. 


```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(df.drop('default', axis = 1), df['default']);
```

A random forest, by default, makes 100 decision trees.  I can view the first decision tree by using `rf.estimators_[0]` and the second decision tree by using `rf.estimators_[1]`


```python
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(rf.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True, max_depth = 2);
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Visualize_Trees_files/Visualize_Trees_16_0.png)



```python
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(rf.estimators_[1],
               feature_names = fn, 
               class_names=cn,
               filled = True, max_depth = 2);
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Visualize_Trees_files/Visualize_Trees_17_0.png)


Now to plot the feature importance 


```python
plt.bar(df.drop('default', axis = 1).columns, rf.feature_importances_)
plt.title('Feature Importance');
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Visualize_Trees_files/Visualize_Trees_19_0.png)


We see the values are similar to the decision tree
