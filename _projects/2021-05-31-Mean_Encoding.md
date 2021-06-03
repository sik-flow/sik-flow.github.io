---
layout: project
title: A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems 
description: Mean Encoding Paper Review
category: Papers
---

Link to [Paper](https://dl.acm.org/doi/10.1145/507533.507538)
> Note: paper behind paywall 

### What did the authors try to accomplish?
The authors showed a technique for dealing with high-cardinality categorical features by applying mean encoding and a smoothing technique. This is very different than previous techniques that typically call for one-hot encoding, which makes the dataset very wide when there are a large number of categories. 

### What were the key elements of the approach?
Traditionally with categorical features you would apply a technique such as one-hot encoding.  For example if I had the following dataset: 


```python
df
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
      <th>animal</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dog</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dog</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dog</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cat</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>horse</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



If I wanted to process the animal variable, I would use one-hot encoding (also known as dummy encoding) 


```python
df_dummies = pd.get_dummies(df)
df_dummies
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
      <th>class</th>
      <th>animal_cat</th>
      <th>animal_dog</th>
      <th>animal_horse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



This makes the dataset wide, as we went from having 2 columns to 4 columns.  Another approach is using mean-encoding.  This takes the conditional probability for each category $ \Pr( class | category )  $ and replaces the category with that probability.  An example is below 


```python
val = df.groupby('animal')['class'].mean()
val
```




    animal
    cat      0.500000
    dog      0.333333
    horse    1.000000
    Name: class, dtype: float64



This is the the conditional probabilities for each of the 3 animals.  Then I will replace the categorical column with these conditional probabilities. 


```python
df_mean_encoding = df.copy()
df_mean_encoding['animal'] = df['animal'].map(val)
df_mean_encoding
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
      <th>animal</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.333333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.500000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.500000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.333333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.333333</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.500000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.500000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We now have only 2 columns, however since there was only one instance of horse the conditional probability for horse is 100%.  This can lead to overfitting when dealing with categories with smaller counts and this is where the smoothing can help out.  For this part I'm showing a technique based on ([this](https://maxhalford.github.io/blog/target-encoding/) and [this](https://www.kaggle.com/dustinthewind/making-sense-of-mean-encoding)).  I am going to be applying a smoothing function to handle for when the category only has a few instances.  I will appply the following formula: 

$$ \frac{n(mu) + smooth(prior)}{n + smooth}  $$

where:
- n = count for each class 
- mu = conditional probability of class given category 
- smooth = number to weigh either the conditional probability or the prior probability.  Larger number means more emphasis on prior 
- prior = probability of positive class 


```python
n = df.groupby('animal').size()
mu = df.groupby('animal')['class'].mean()
smooth = 10
prior = df['class'].mean()
mu_smoothed = (n * mu + smooth * prior) / (n + smooth)
mu_smoothed
```




    animal
    cat      0.500000
    dog      0.461538
    horse    0.545455
    dtype: float64




```python
# conditional probabilities 
val
```




    animal
    cat      0.500000
    dog      0.333333
    horse    1.000000
    Name: class, dtype: float64



After applying the smoothing, we see that the value for dog increased from 33% to 46% and the value for horse decreases from 100% to 55% - this is because the prior probability is 50%.  If I decrease the value of the `smooth` variable, it will decrease how much I weigh the probabilities towards the prior.  Now I will use a `smooth` value of 5.


```python
smooth = 5
mu_smoothed = (n * mu + smooth * prior) / (n + smooth)
mu_smoothed
```




    animal
    cat      0.500000
    dog      0.437500
    horse    0.583333
    dtype: float64



### What can you use yourself?

High-cardinality is relatively common with categorical features.  Applying this technique can allow for faster models (due to not increasing the number of features) and potentially more accurate models.  Additionally, the author mentioned some techniques for hierarchical data.  The author mentions that one way to reduce cardinality with zipcodes is to convert them from having 5 numbers to just the first 3 numbers.  The first 3 numbers represent zip codes in the same metropolitan area.  Thinking about ways to transform your data before blindly applying an encoding strategy is something that I will be considering going forward.    

### What other references do you want to follow?

The author of the paper recently wrote a [blog post](https://towardsdatascience.com/extending-target-encoding-443aa9414cae) talking about this paper (due to the popularity in recent years).  [CatBoost](https://catboost.ai/) uses a similar process for handling categorical features with a smoothing component.  
