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
    <img src = images/initial_data_info.png width = 100% / >
</center>


**Data Exploration and Quality Assessment**

The following plan will be followed for this phase. 

* **Missing values:**
    * `df.isnull().sum()` - Check for missing values in each column.
    * Visualize missing data patterns using heatmaps or bar charts.
    * Consider strategies for handling missing data (e.g., imputation, deletion).
 
We notice that there are null values in several columns. For those columns such as `price` , `year` and `odometer` - that have NULL/NaN or 0 values are unusable thus entire row is dropped.

The values of `odometer` and `year` (Model Year) are converted to integer from float for easier representation. 
The following features are coverted from `object` to `category`

```
categorical_cols = ['manufacturer', 'condition', 'cylinders', 'fuel', 
                    'title_status', 'transmission', 'drive', 
                    'size', 'type', 'paint_color']
```

### Motivation

**Memory Efficiency**: Categorical types use less memory compared to object types, especially when there are many repeated string values. This can be significant in large datasets.

**Performance Improvement**: Operations on categorical data can be faster than on object data. For example, filtering, grouping, and aggregating can be more efficient.

**Machine Learning Algorithms**: Many ML algorithms can benefit from categorical data. Categorical variables can be encoded using techniques like one-hot encoding or label encoding, which can improve the performance of algorithms that require numerical input.

**Data Integrity**: By converting to categorical, we can ensure that only the defined categories are used, which can help prevent errors in data entry or processing.

**Null or Invalid category names handling**
Any category improperly filled will be filled as `UNKNOWN` or `other` based on existing data entry for the column

* **Outliers:**
    * Visualize distributions using histograms or box plots to identify potential outliers.
    * Investigate outliers: Are they errors or legitimate extreme values?
    * Decide on appropriate handling (e.g., removal, transformation).

 We use the IQR method to remove outliers for the numerical columns. 
 We use a threshold method to remove low propensity category values for categorical columns.

**Ordered Category**
The `condition` column has categories that can be ordered from least desireable (UNKNOWN and salvage) to most desireable ( like new and new ).
```
ordered_conditions_values = ['UNKNOWN', 'salvage', 'fair', 'good', 'excellent', 'like new', 'new']
```


* **Data types and consistency:**
    * Ensure data types are appropriate for each column.
    * Check for inconsistencies within categorical variables (e.g., different spellings for the same category).
 
**Duplicate vehicles in different locations**
We notice that the same vehicle is listed in different locations at the same time, to avoid this skewing the model, we decide to drop these duplicate entries
```
# Drop rows that have the same values for 'odometer' and 'VIN'
# This will keep only the first occurrence of each unique combination of 'odometer' and 'VIN'
df = df.drop_duplicates(subset=['odometer', 'VIN'], keep='first')
```

**Drop VIN and ID**
These are for identification only - do not add value to modelling

```
# Get unique names from the model column
unique_model_values = pd.Series(df['model'].unique()).dropna().tolist()  # This creates a list of unique values

# Let's check the number of unique model names 
print(len(unique_model_values))
20036
```

**Drop column `model` - too many variants**
We notice that many car models are named with slight variations - it's difficult to meaningfully to group them in any discernable manner in a short timeframe without any deep knowledge into Automobile subject matter. We thus decide that Manufacturer name and Year should constitutue adequate uniqueness.


### Review Cleaned Data

<center>
    <img src = images/cleaned_data_info.png width = 100% / >
</center>

### Cleaned Data Summary

<center>
    <img src = images/cleaned_data_summary.png width = 100% / >
</center>


* **Relationships between variables:**
    * Calculate correlations between numerical variables.
    * Visualize relationships using scatter plots or pair plots.
    * Explore potential multicollinearity issues (high correlation between features)


### Distribution of numerical variables - `odometer` and `year` (Model Year)

<center>
    <img src = images/numeric_hist.png width = 100% / >
</center>

### Histogram of the target variable Price

<center>
    <img src = images/price_distribution.png width = 100% / >
</center>




* **Univariate and Bivariate Analysis:**
    * Explore the distributions of key variables (e.g., price, mileage, age) to understand their central tendency and spread.
    * Analyze the relationship between key predictor variables (e.g., mileage, Model Year) and the target variable (price).

Univariate Analysis chart is presented here. 

<center>
    <img src = images/univariate_analysis.png width = 100% / >
</center>

The Bivariate chart is available in the Notebook. 

### Data Preparation

After our initial exploration and fine tuning of the business understanding, it is time to construct our final dataset prior to modeling.  Here, we want to make sure to handle any integrity issues and cleaning, the engineering of new features, any transformations that we believe should happen (scaling, logarithms, normalization, etc.), and general preparation for modeling with `sklearn`. 

**Data Cleaning and Preparation**

The cleaned data was used in the univariate and bivariate analysis. 



### Modeling

With our final dataset in hand, it is now time to build some models.  Here, you should build a number of different regression models with the price as the target.  In building your models, you should explore different parameters and be sure to cross-validate your findings.

**Create training set**
For Vehicle characteristics we will analyze it in an non-regional manner, so we drop state and region from X
Also the target variable price is dropped.

```
X = df_cleaned.drop(['state', 'region', 'price'], axis = 1)

y = df_cleaned['price']
```

We create a machine learning pipeline with three stages

1. Scaling numerical columns,
2. Encode categorical columns,
3. Create a Column Transformer stage that will be used by the pipeline
4. Fit a Ridge regression model

```
numerical_transformer = StandardScaler()
# Create encoders
ordinal_transformer = OrdinalEncoder()
non_ordinal_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('non_ord', non_ordinal_transformer, non_ordinal_features),
    ],
    remainder='drop'  # Drop any remaining columns not specified
)

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])
```


For CrossValidation and to tune for hyperparameters we perform a `GridSearchCV`.

```
# Define the parameter grid for Ridge regression
param_grid = {
    # Create 15 exponentially spaced samples from 0.01 to 100
    'regressor__alpha': np.logspace(np.log10(0.01), np.log10(100), num=15), 
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error', 
    verbose = 4, 
    n_jobs=-1,
)
```


### Evaluation

With some modeling accomplished, we aim to reflect on what we identify as a high quality model and what we are able to learn from this.  We should review our business objective and explore how well we can provide meaningful insight on drivers of used car prices.  Your goal now is to distill your findings and determine whether the earlier phases need revisitation and adjustment or if you have information of value to bring back to your client.


### Deployment

Now that we've settled on our models and findings, it is time to deliver the information to the client.  You should organize your work as a basic report that details your primary findings.  Keep in mind that your audience is a group of used car dealers interested in fine tuning their inventory.


