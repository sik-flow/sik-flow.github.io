---
layout: post
title: Using Monte Carlo Simulation to Determine who Makes More Money at University of Missouri
---

{{ page.title }}
================

<p class="meta">26 October 2019 - Kansas City</p>

I recently came across the following PDF [University of Missouri Employee Salaries](https://mospace.umsystem.edu/xmlui/bitstream/handle/10355/67263/AnnualSalaryReport2018-2019.pdf?sequence=1&isAllowed=y).
![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/Overiew_PDF.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/Overiew_PDF.png)

This PDF contains salary information for over 20,000 University of Missouri employees.  I wanted to see if I could accomplish the following:
1. Parse the PDF into a format that I could analyze using Python
2. Using each employees first name and/or middle name, make a "guess" on their gender 
3. Use a statistical test to determine whether a certain gender makes more money on average 


# Part 1: Parsing the PDF

I used the following notebook to parse the PDF, [Parse PDF](https://github.com/sik-flow/mizzou_salaries-/blob/master/Parse_PDF.ipynb).  To start I used the library [pdfminer](https://buildmedia.readthedocs.org/media/pdf/pdfminer-docs/latest/pdfminer-docs.pdf) to convert the pdf into a text file.  This process can be seen below:  
![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/pdf_to_text.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/pdf_to_text.png)

After converting the PDF to a text file, I needed a way to split the text file into the individual rows.  
![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/split.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/split.png)

I ended up making a split based on the Business Unit column.  I know that this column is going to be consistent for all the employees at each of the 4 campuses (Columbia, Kansas City, Rolla, St. Louis). This led to having elements that looked like the following:
![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/after_split.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/after_split.png)

I next wrote a function to extract the first, middle (if applicable), and last name from this string (which can be seen below).  I started with grabbing the last name because this was easiest.  The last name was the first element after I did a split on the `,`.  In this case it was `Abadi`.  Grabbing the first and middle name proved to be much more difficult.  This was mainly driven by people not always having middle names and sometimes having multiple middle names.  I wrote a series of conditional statements and using regular expressions to handle the majority of these edge cases.  The regular expression is looking for cases where there are multiple capital letters in a single word.  For example `MartaStaffing` meets this condition.  I am looking for these cases because this is where the name ends and the department for the employee begins.  The function returns 4 outputs:
- first - the employees first name
- middle - the employees middle name (could have multiple middle names or even no middle names)
- last - employees last name
- splitter - the last part of the employee first and/or middle name.  This was used in the next function to know where in the string we stopped parsing. 
![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/get_name.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/get_name.png)

Next I made a function `get_department`.  This function takes where we left off on the previous step (grabbing the name) and using a regular expression looks for when we start getting all capital letters.  All the text to the left of all the capital letters is the department.  We can see an example of this below.  

![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/get_department%20.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/get_department%20.png)

Finally, I grabbed the employees job title and their respective pay rate.  For this I did not have to use a regular expression, but instead was able to do a split based on the `$`.  

![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/get_title.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/get_title.png)

I then repeated this process for the 3 other locations (Kansas City, Rolla, and St. Louis).  I ended up with 23,581 rows in my dataframe.  You may have noticed that I had a `try` and `except` statement in the `get_department` function.  This was due to some edge cases and misspellings in the pdf.  I counted there to be a total of 23,688 rows in the original pdf.  This process was able to successfully parse 99.55% of the rows in the pdf.  The notebook for parsing the pdf can be found [here](https://github.com/sik-flow/mizzou_salaries-/blob/master/Parse_PDF.ipynb). 

![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/error_parse.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/error_parse.png)

My dataset then looked as follows 
![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/original_data.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/original_data.png)

The `Rate` column was a string and there was a mix of hourly rates and salary rates.  I made the rate column a float and added a new column for hourly salary and my dataframe looked as follows. The notebook for accomplishing this can be found [here](https://github.com/sik-flow/mizzou_salaries-/blob/master/Clean_CSVs.ipynb).

![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/final_df.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/final_df.png)

Now that the dataset was relatively clean, I am ready to start with the EDA.

# Part II: EDA 

I started by looking at what the average salary and hourly rate for employees at each of the four campuses. 

![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/salary_by_location.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/salary_by_location.png)

Salaries for employees in Columbia are higher than the other campuses.  This could be driven by the fact that there is a hospital at the Columbia location and hospital employees typically are paid on the higher end.  We see that hourly rate for employees is also highest in Columbia. 

Next I was curious what departments have the highest salaries.  I made a bar plot showing the top salaries by department.  

![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/top_departments.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/top_departments.png)

The top paying departments are medical fields and sports.  This further supports the idea that Columbia is the highest paying campus due to having a hospital on campus and having the biggest athletics program of the 4 campuses.  

Next I used the [gender guesser](https://pypi.org/project/gender-guesser/) to make a guess what an employees gender is based on their name.  This library uses a corpus of 40,000 first names and then makes a guess what the gender is based on this corpus.  The library returns 6 options when a name is put in (`unknown`, `androgynous`, `male`, `female`, `mostly_male`, `mostly_female`).  If the first name did not return `male` or `female` I then would check the middle name to see if that could get a better result. 

<img src = "https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/gender_count.png" height = "400" width = "400" >

The [gender guesser](https://pypi.org/project/gender-guesser/) was not able to find all names, as we can see above, and **I'd like to stress using a program to guess what someones gender is based on their first and middle name is not an ideal way to do this.**  With that being said, I'm going to continue to do this because I'm using this to show off a process (parsing a PDF and using a Monte Carlo Simulation) vs proving one gender is or not paid more or less (due to the uncertainty of the process of guessing genders).  

Next I looked at the average salary based on the guessed gender.  

<img src = "https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/salary_gender.png" height = "400" width = "400" >

We see the average salary for females is less than average male salaries.  This leads to more questions - is the difference statistically different and what does the distribution look like.  

![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/female_distributions.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/female_distributions.png) 

![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/male_distributions.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/male_distributions.png)

I started with looking at the distributions of the two groups as this will help me decide what kind of hypothesis test to run.  Since these distributions are both highly skewed and non-normal, I will not be able to run a T-Test to test statistical significance.  Something that caught my eye was the peak at the beginning of both distributions.  I decided to zoom in on salaries <\$100,000.  We see we have a peak at <\$25,000.  This was driven by Adjunct Professors, there were 806 of them in the dataset and they had an average salary of \$11,740.  

# Part III: Statistical Testing

I'm now ready to begin my statistical test to see if the difference in salaries between the 2 groups is statistically significant.  Like I mentioned earlier, I will not be able to do a T-Test due to the data not being normally distributed.  I am instead going to try a non-parametric test.  

<p>One possible test I could run is the <a href="https://en.wikipedia.org/wiki/Resampling_(statistics)#Permutation_tests">Permutation Test</a>. The Permutation Test will determine statistical significance by taking every single combination from the 2 samples.</p>  

First I wanted to see how many possible combinations there are.  There are 4,989 males that have a salary and 4,962 females that have a salary (for a total of 9,951).  

<img src = "https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/Combinations.png" height = "300" width = "300" >

This means that I have a huge amount of possible combinations.  So large in fact, that scipy was unable to calculate it.  I do not have the computing power and this is way too many possible combinations. 

<p>I am instead going to run a <a href="https://en.wikipedia.org/wiki/Resampling_(statistics)#Monte_Carlo_testing">Monte Carlo Simulation</a>.  A Monte Carlo Simulation runs a large amount of combinations, but not all of them, and thus is able to provide confidence in the results of the test while being more computationally efficient then Permutation Testing.   

Starting out with Monte Carlo Simulation, like all statistical tests, I need to state my null and alternative hypothesis. </p>  

```
Null Hypothesis: The average salary of men is less than or equal to the 
average salary of women
Alternative Hypothesis:  The average salary of men is greater the average 
salary of women
```

![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/monte_carlo.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/monte_carlo.png)

I ran the above for loop for my Monte Carlo Simulation.  I took 1,000,000 samples from the original dataset.  Of those 1,000,000 samples I took the mean of each and took the differences of the two means.  Then compared to see if the mean was larger then my original difference between male and female salaries. <br>
<img src = "https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/Dr.%20Evil.png" height = "300" width = "300" > <br>

The idea is that if 50% of my 1,000,000 samples have a mean difference larger then my original difference I would say that this happened by chance, however if only 1% of the 1,000,000 samples is greater than the original mean I am really confident this difference did not happen by chance.  

![https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/monte_carlo_results.png](https://raw.githubusercontent.com/sik-flow/mizzou_salaries-/master/images/monte_carlo_results.png)

Above is the distribution of the sample differences and the white line on the right is the original difference.  I get a P-Value of 0, this means of the 1,000,000 samples that I randomly selected not a single one had a sample mean difference greater than 15,000.  I am very confident that the results of the original sample difference did not occur by chance. 

To conclude - due to having a P-Value of 0, I can reject my null hypothesis and conclude that people with a guessed gender of male on average are paid more money than people with a guessed gender of women at the University of Missouri.  This is, again, with the huge caveat that coming up with gender based on first and middle name is not a perfect process in identifying someones gender.  

If you made it this far, thanks and let me know if you have any questions - jherman1199@gmail.com
