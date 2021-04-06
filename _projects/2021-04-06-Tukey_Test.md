---
layout: project
title: Example and Visualization of Tukey Test
description: Example of Tukey Test
category: Statistics
---

A Tukey Test is a way to compare more than 2 datasets and see which are statistically significant.  I will demo this using the [Auto MPG Dataset from UCI](http://archive.ics.uci.edu/ml/datasets/Auto+MPG) 


```python
import pandas as pd 

# load in dataset 
df = pd.read_csv('auto-mpg.csv')

# update orgin column 
df.loc[df['origin'] == 1, 'origin'] = 'US'
df.loc[df['origin'] == 2, 'origin'] = 'Germany'
df.loc[df['origin'] == 3, 'origin'] = 'Japan'

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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>US</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>US</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>US</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>US</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>US</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>



I want to see if the number of cylinders is statistically different depending on the origin of the car.  


```python
from statsmodels.stats.multicomp import MultiComparison

cardata = MultiComparison(df['cylinders'], df['origin'])
results = cardata.tukeyhsd()
results.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD, FWER=0.05</caption>
<tr>
  <th>group1</th>  <th>group2</th> <th>meandiff</th> <th>p-adj</th>  <th>lower</th>   <th>upper</th> <th>reject</th>
</tr>
<tr>
  <td>Germany</td>  <td>Japan</td>  <td>-0.0559</td>  <td>0.9</td>  <td>-0.5805</td> <td>0.4688</td>  <td>False</td>
</tr>
<tr>
  <td>Germany</td>   <td>US</td>    <td>2.0919</td>  <td>0.001</td> <td>1.6595</td>  <td>2.5242</td>  <td>True</td> 
</tr>
<tr>
   <td>Japan</td>    <td>US</td>    <td>2.1477</td>  <td>0.001</td>  <td>1.735</td>  <td>2.5605</td>  <td>True</td> 
</tr>
</table>



We see that the number of cylinders in cars that originated in Germany and Japan are not statistically significant, we also see that cars that originated in the US have a statistcally different number of cylinders than cars that originated in Japan or Germany.  

Now lets visualize this. 


```python
results.plot_simultaneous();
```

![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Tukey_Test_files/Tukey_Test_5_0.png)


The X-Axis is the number of cylinders and we see why the US had a statistically significant result, due to having a much higher mean number of cylinders.  

We can also highlight one of the groups using `comparison_name`.  I'm going to highlight Japan.  


```python
results.plot_simultaneous(comparison_name = 'Japan');
```

![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Tukey_Test_files/Tukey_Test_7_0.png)


This shows that Germany intersects the confidence interval of Germany and this is why they were not statistically different.  
