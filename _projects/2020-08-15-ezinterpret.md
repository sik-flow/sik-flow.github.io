---
layout: project
title: Interpret Linear Regression Model in Python
description: Interpret Linear Regression Model in Python
category: Interpretability
---

Demonstration of [EZInterpret](https://github.com/sik-flow/ezinterpret/blob/master/ezinterpret.py).

Start with loading in necessary packages and dataset. 


```python
import pandas as pd
from statsmodels.formula.api import ols
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ezinterpret
```


```python
df = pd.read_csv('https://raw.githubusercontent.com/sik-flow/datasets/master/bike.csv')
```

Do some small data cleaning and feature engineering. 


```python
df['summer'] = df['season'].map(lambda x: 1 if x == 3 else 0)
df['fall'] = df['season'].map(lambda x: 1 if x == 4 else 0)
df['winter'] = df['season'].map(lambda x: 1 if x == 1 else 0)

df['misty'] = df['weathersit'].map(lambda x: 1 if x == 2 else 0)
df['rain_snow_storm'] = df['weathersit'].map(lambda x: 1 if x > 2 else 0)

df['dteday'] = pd.to_datetime(df['dteday'])

df['days_since_2011'] = df['dteday'].map(lambda x: (x - df.loc[0, 'dteday']).days)

df['temp'] = df['temp'] * (39 - (-8)) + (-8)
df['windspeed'] = 67 * df['windspeed']
df['hum'] = df['hum'] * 100
```

Build my linear regression model 


```python
formula = "cnt ~ summer+fall+winter+holiday+workingday+misty+rain_snow_storm+temp+hum+windspeed+days_since_2011"
model = ols(formula= formula, data=df).fit()

model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>cnt</td>       <th>  R-squared:         </th> <td>   0.794</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.790</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   251.2</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 10 Aug 2020</td> <th>  Prob (F-statistic):</th> <td>1.05e-237</td>
</tr>
<tr>
  <th>Time:</th>                 <td>21:01:57</td>     <th>  Log-Likelihood:    </th> <td> -5993.0</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   731</td>      <th>  AIC:               </th> <td>1.201e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   719</td>      <th>  BIC:               </th> <td>1.207e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>       <td> 3298.7604</td> <td>  262.145</td> <td>   12.584</td> <td> 0.000</td> <td> 2784.099</td> <td> 3813.422</td>
</tr>
<tr>
  <th>summer</th>          <td> -761.1027</td> <td>  107.169</td> <td>   -7.102</td> <td> 0.000</td> <td> -971.505</td> <td> -550.701</td>
</tr>
<tr>
  <th>fall</th>            <td> -473.7153</td> <td>  109.947</td> <td>   -4.309</td> <td> 0.000</td> <td> -689.570</td> <td> -257.860</td>
</tr>
<tr>
  <th>winter</th>          <td> -899.3182</td> <td>  122.283</td> <td>   -7.354</td> <td> 0.000</td> <td>-1139.393</td> <td> -659.243</td>
</tr>
<tr>
  <th>holiday</th>         <td> -686.1154</td> <td>  203.301</td> <td>   -3.375</td> <td> 0.001</td> <td>-1085.251</td> <td> -286.980</td>
</tr>
<tr>
  <th>workingday</th>      <td>  124.9209</td> <td>   73.267</td> <td>    1.705</td> <td> 0.089</td> <td>  -18.921</td> <td>  268.763</td>
</tr>
<tr>
  <th>misty</th>           <td> -379.3985</td> <td>   87.553</td> <td>   -4.333</td> <td> 0.000</td> <td> -551.289</td> <td> -207.508</td>
</tr>
<tr>
  <th>rain_snow_storm</th> <td>-1901.5399</td> <td>  223.640</td> <td>   -8.503</td> <td> 0.000</td> <td>-2340.605</td> <td>-1462.475</td>
</tr>
<tr>
  <th>temp</th>            <td>  110.7096</td> <td>    7.043</td> <td>   15.718</td> <td> 0.000</td> <td>   96.882</td> <td>  124.537</td>
</tr>
<tr>
  <th>hum</th>             <td>  -17.3772</td> <td>    3.169</td> <td>   -5.483</td> <td> 0.000</td> <td>  -23.600</td> <td>  -11.155</td>
</tr>
<tr>
  <th>windspeed</th>       <td>  -42.5135</td> <td>    6.892</td> <td>   -6.169</td> <td> 0.000</td> <td>  -56.044</td> <td>  -28.983</td>
</tr>
<tr>
  <th>days_since_2011</th> <td>    4.9264</td> <td>    0.173</td> <td>   28.507</td> <td> 0.000</td> <td>    4.587</td> <td>    5.266</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>91.525</td> <th>  Durbin-Watson:     </th> <td>   0.911</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 194.706</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.719</td> <th>  Prob(JB):          </th> <td>5.25e-43</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.079</td> <th>  Cond. No.          </th> <td>3.74e+03</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.74e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



Make an instance of ezinterpret and pass in my model. 


```python
ins = ezinterpret.ez_linear(model)
```

First I am going to look at the feature importance.  This is based on the T-Statistic.


```python
ins.feature_importance();
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/example_of_ezinterpret_files/example_of_ezinterpret_10_0.png)


Next I am going to look at the weight plots.  This is based on the coefficients.  


```python
ins.weight_plot();
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/example_of_ezinterpret_files/example_of_ezinterpret_12_0.png)


We see that `days_since_2011` has a coefficient near 0 even though it is the most important feature.  This is because `days_since_2011` (the raw number) is a big value.  A better way to look at it is with an effect plot which shows the raw data multiplied by the coefficient.  

To run this I need to pass in a dictionary with any one hot encoded features. 


```python
cats = {'weather': ['misty', 'rain_snow_storm'], 'season': ['summer', 'fall', 'winter']}
ins.effect_plot(df, cats);
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/example_of_ezinterpret_files/example_of_ezinterpret_14_0.png)


Now we see the big range that `days_since_2011` and temperate have on number of bikes rented.  We can see why these are the two most important features. 

Finally, I am going to overlay a local prediction on the effect plot. 


```python
test_case = df.loc[302].to_dict()
```


```python
ins.effect_plot_with_local_pred(df, cats, test_case, 'cnt');
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/example_of_ezinterpret_files/example_of_ezinterpret_17_0.png)


Here our model predicts there to be 3,501 bikes rented when there were actually 3,331.  The red `x`'s are where the values for the specific case lie with regards to the rest of the data. 
