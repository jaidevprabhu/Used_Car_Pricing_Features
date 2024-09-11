# What drives the price of a car?

![](images/kurt.jpeg)

**OVERVIEW**

In this application, you will explore a dataset from kaggle. The original dataset contained information on 3 million used cars. The provided dataset contains information on 426K cars to ensure speed of processing.  Your goal is to understand what factors make a car more or less expensive.  As a result of your analysis, you should provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.

### CRISP-DM Framework

<center>
    <img src = images/crisp.png width = 50%/>
</center>


To frame the task, throughout our practical applications we will refer back to a standard process in industry for data projects called CRISP-DM.  This process provides a framework for working through a data problem.  Your first step in this application will be to read through a brief overview of CRISP-DM [here](https://mo-pcco.s3.us-east-1.amazonaws.com/BH-PCMLAI/module_11/readings_starter.zip).  After reading the overview, answer the questions below.

### Business Understanding

From a business perspective, we are tasked with identifying key drivers for used car prices.  In the CRISP-DM overview, we are asked to convert this business framing to a data problem definition.  Using a few sentences, reframe the task as a data task with the appropriate technical vocabulary. 

The task of identifying key drivers for used car prices can be reframed as a data task of **predictive modeling or regression analysis**. We will use historical data on used car prices and their associated attributes (such as make, model, year, mileage, condition, etc.) to **build a model that can predict used car prices based on these attributes**. The **key drivers** will be the **attributes that have the most significant impact on the predicted prices**, as identified through **feature importance analysis or by examining the coefficients of the regression model**. 


The **Manheim Index** is increasingly recognized by both financial and economic analysts as the premier indicator of pricing trends in the used vehicle market, but should not be considered indicative or predictive of any individual remarketer's results.

[Manheim](https://site.manheim.com/en/services/consulting/used-vehicle-value-index.html)


```python

```


```python

```

### Data Understanding

After considering the business understanding, we want to get familiar with our data.  The following steps will get us to know the dataset and identify any quality issues within.  

**Data Collection and Initial Inspection**

* **Gather the data:** We have teh data given to us as a subset of the Kaggle public dataset in CSV format. 
* **Data dictionary:** A data dictionary is not available so we decide to loo at pandas dataframe analysis to look at each column (feature) and their data types.
* **First look:** :
    * `df.head()` - View the first few rows to get a sense of the data structure and values.
    * `df.info()` - Get summary statistics, including column names, data types, and number of non-null values.
    * `df.describe()` - Get descriptive statistics for numerical columns (mean, median, min, max, etc.).

```python
df.info()
```

<center>
    <img src = images/initial_data_info.png width = 50% / >
</center>


```python

```


```python

```

**Data Exploration and Quality Assessment**

* **Missing values:**
    * `df.isnull().sum()` - Check for missing values in each column.
    * Visualize missing data patterns using heatmaps or bar charts.
    * Consider strategies for handling missing data (e.g., imputation, deletion).
* **Outliers:**
    * Visualize distributions using histograms or box plots to identify potential outliers.
    * Investigate outliers: Are they errors or legitimate extreme values?
    * Decide on appropriate handling (e.g., removal, transformation).
* **Data types and consistency:**
    * Ensure data types are appropriate for each column.
    * Check for inconsistencies within categorical variables (e.g., different spellings for the same category).
* **Relationships between variables:**
    * Calculate correlations between numerical variables.
    * Visualize relationships using scatter plots or pair plots.
    * Explore potential multicollinearity issues (high correlation between features)
* **Univariate and Bivariate Analysis:**
    * Explore the distributions of key variables (e.g., price, mileage, age) to understand their central tendency and spread.
    * Analyze the relationship between key predictor variables (e.g., mileage, Model Year) and the target variable (price).


```python

```

### Data Preparation

After our initial exploration and fine tuning of the business understanding, it is time to construct our final dataset prior to modeling.  Here, we want to make sure to handle any integrity issues and cleaning, the engineering of new features, any transformations that we believe should happen (scaling, logarithms, normalization, etc.), and general preparation for modeling with `sklearn`. 

**Data Cleaning and Preparation**

* **Address missing values:**
    * Impute missing values using appropriate techniques (e.g., mean, median, mode, regression imputation).
    * Drop rows or columns with excessive missing values as necessary.
* **Handle outliers:**
    * Remove or transform outliers based on domain knowledge and analysis.
* **Correct data types:**
    * Convert columns to appropriate data types if needed.
* **Standardize categorical variables:**
    * Ensure consistent representation of categoriees (e.g., age of the car, interaction terms).


```python

```


```python

```


```python

```


```python

```

### Modeling

With our final dataset in hand, it is now time to build some models.  Here, you should build a number of different regression models with the price as the target.  In building your models, you should explore different parameters and be sure to cross-validate your findings.


```python

```


```python

```


```python

```


```python

```

### Evaluation

With some modeling accomplished, we aim to reflect on what we identify as a high quality model and what we are able to learn from this.  We should review our business objective and explore how well we can provide meaningful insight on drivers of used car prices.  Your goal now is to distill your findings and determine whether the earlier phases need revisitation and adjustment or if you have information of value to bring back to your client.


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

### Deployment

Now that we've settled on our models and findings, it is time to deliver the information to the client.  You should organize your work as a basic report that details your primary findings.  Keep in mind that your audience is a group of used car dealers interested in fine tuning their inventory.


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
