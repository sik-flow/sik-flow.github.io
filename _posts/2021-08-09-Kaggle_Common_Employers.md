---
layout: post
title: Most Common Employers of Kaggle Competitors
description: Who are the most common employers of Kaggle Competitors?
---

## Background 

I recently noticed that of the top 10 ranked Kagglers (in Competitions) 7 of them either worked at H2O.ai or Nvidia.  I wanted to explore this to see if this was a trend where these 2 companies employed more highly ranked Kaggle Competitors than other companies.

## Data 
I scraped the Top 1000 ranked Kaggle competitors from [here](https://www.kaggle.com/rankings).  From there I was able to pull the employer and job title from each competitor.  

> Note: not all Kaggle competitors listed their employer or job title, these competitors were ignored from this analysis 

## Insights 
I first looked at the top employers of Kaggle competitors ranked in the Top 1000.  

![Most Common Employers](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_posts/Images/Kaggle_Competitors/Most_Common_Employers.png)

We see that H2O.ai employs over 3.5% of the total Kagglers ranked in the top 1000 and DeNa, not Nvidia, employs the 2nd most Kagglers of any company. 

Next, I wanted to know what are the median rankings Kagglers by company.  For example we know Nvidia employs a lot of Kagglers ranked in the top 10, do they employ higher ranked Kagglers than other companies? 

![Median Kaggle Ranking](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_posts/Images/Kaggle_Competitors/Median_Kaggle_Ranking.png)

We see the highest ranked Kagglers work at Nvidia, followed by H2O.ai.  DeNA employs a greater number of Kagglers ranked in the top 1000 than Nvidia, but Nvidia's Kagglers have much higher median ranking.  

Finally, since the data included job title I wanted to look at the most common job titles of Kaggle Competitors ranked in the top 1000. 

![Most Common Job Title](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_posts/Images/Kaggle_Competitors/Most_Common_Job_Title.png)

We see that over 20% of Kagglers Ranked in the Top 1000 list a job title of `Data Scientist`, which is not surprising.  The next most common job title is `Student`.  What I found interesting, Kaggle has a reputation for requiring deep learning to win competitions and deep learning is not listed in any of the top 10 job titles. 