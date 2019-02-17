import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
# %matplotlib inline

""" 
Read in the breast cancer data from csv file and store it in the dataframe bcdata 
"""
bcdata = pd.read_csv("/Users/tsao/Desktop/IS 289 Data Analysis Project/bcdata.csv", encoding="utf-8-sig")

""" 
Applies a function to the 'Bare nuclei' column of the breast cancer dataset that casts all values of that column to a float data type
if possible, and otherwise to NaN 
"""
bcdata['Bare nuclei'] = bcdata['Bare nuclei'].apply(lambda x: float(x) if x != "?" else np.nan)

"""
Get a visual count of how many null (NaN) values there are in each column of the entire data frame
"""
bcdata.isnull().sum()

"""
Get a sum count of the total number of NaN values in the dataframe
"""
bcdata.isnull().sum().sum()

# """
# Create a new dataframe for just the rows of data with missing values
# """
# miss_vals_df = bcdata[bcdata.isnull().any(axis=1)]

"""
Create a copy of the bcdata dataframe
"""
bcdata_copy = bcdata.copy(deep=True)

"""
Create a new column of the dataframe that indicates True/False depending on whether the value of the 'bare nuclei' column is an integer
"""
bcdata_copy['Miss Non Miss'] = bcdata_copy['Bare nuclei'].apply(lambda x: x.is_integer())

"""
Look at various methods to interpolate missing values using pandas interpolate method
** DISCOVERY : the interpolate method only looks at values across one attribute (one variable/column of data), so assumes that there
is some sort of dependent relationship from one value of that attribute to the next. This, however, does not apply to this dataset
"""
est_1 = bcdata_copy.interpolate(method='linear')
est_1_missrows = est_1[est_1['Miss Non Miss']==False]

est_2 = bcdata_copy.interpolate(method='nearest')
est_2_missrows = est_2[est_2['Miss Non Miss']==False]

est_3 = bcdata_copy.interpolate(method='values')
est_3_missrows = est_3[est_3['Miss Non Miss']==False]

est_4 = bcdata_copy.interpolate(method='index')
est_4_missrows = est_4[est_4['Miss Non Miss']==False]

for i in range(len(est_1_missrows)):
	print(est_1_missrows.values[i][6], est_2_missrows.values[i][6], est_3_missrows.values[i][6], est_4_missrows.values[i][6])

"""
Store all NaN values in a separate dataframe to be used later to predict missing 'Bare nuclei' values
"""
miss_vals = bcdata[bcdata.isnull().any(axis=1)]
miss_vals = miss_vals.drop(['Bare nuclei', 'Class label', 'Sample ID'], axis=1)

"""
Create a new dataframe X that drops any row of data with missing values
Create another dataframe XX from the X dataframe that contains all the columns of bcdata EXCEPT the sample ID column
Amend the X dataframe by removing the 'class label,' 'sample ID,' and 'bare nuclei' columns
"""
X = bcdata.dropna()
XX = X.drop(['Sample ID'], axis=1)
X = X.drop(['Class label', 'Sample ID', 'Bare nuclei'], axis=1)

"""
Create a new series y that contains only the bare nuclei column of the breast cancer data
Create a new series z that contains only the class label column of the breast cancer data
"""
y = bcdata['Bare nuclei']
z = bcdata.dropna()
z = z['Class label']


"""
Create new clean y dataframe that excludes all incomplete records
"""
y_clean = y.dropna()

"""
Determine the best guess for missing data values (imputation) using
1) Linear Regression to determine correlation between attributes
	a) Use sklearn to create an X_predict dataframe that includes all of the predictor variables (excluding the 
	outcome variable ('Bare nuclei') and the rows of data with any missing values)
	b) Use sklearn to create a y_predict array that includes only the outcome variable (excluding any rows of data
	where the value of that variable is missing)
	c) Run the model.fit(X_predict, y_predict) to apply multiple linear regression
	d) Run model.summary() to see the regression results
2) Use 
"""

"""
Linear Regression to determine correlation between attribute with missing values and remaining attributes
"""
from sklearn import datasets, linear_model

"""
Store the data frame of the predictor variables to be used for the multiple linear regression in a matrix/numpy array
"""
X_known = X.values

"""
Store the data frame of the outcome variable to be used for the multiple linear regression in an array/series
"""
y_known = y_clean

from sklearn.linear_model import LinearRegression
"""
Run the linear model on the data using the X_known and y_known data frames
"""
model1 = LinearRegression()
model1.fit(X_known, y_known)

"""
Print/look at regression results to estimate correlation
"""
print(model1.intercept_, model1.coef_)

"""
Use the statsmodels Python module to be able to create summaries and visuals
If p-value (P > |t|) is > .05, the independent variable is considered insignificant (i.e. considered not correlated to dependent variable)
"""
import statsmodels.api as sm
X_predict = miss_vals.values

model2 = sm.OLS(y_known, X_known)
results = model2.fit()
print(results.summary())
ypred = results.predict(X_known)
plt.scatter(XX['Bare nuclei'], ypred, alpha=.1)
plt.xlabel('Actual Bare nuclei')
plt.ylabel('Predicted Bare nuclei')
plt.show()
# XX = XX[XX['Class label'] == 4]
XX = XX.drop(['Class label'], axis=1)
plt.scatter(XX['Normal nucleoli'], XX['Bare nuclei'], alpha=.1)
plt.xlabel('Normal nucleoli')
plt.ylabel('Bare nuclei')
# plt.show()

"""
Zip provides a list of tuples, each with predicted bare nuclei value, and actual
"""
import math
total = 0
for pred, actual in zip(ypred, XX['Bare nuclei']):
	total += abs(1 - pred/actual)
	print(pred/actual)

print(total/len(ypred))

"""
Creates a segmented line chart/color map. Each value across the x-axis represents one attribute
"""
plt.imshow(XX, aspect='auto')
plt.colorbar()
# plt.show()

"""
Find the predicted values for missing 'Bare nuclei' attribute using their associated X values
"""
X_predict = miss_vals.values
y_predict = results.predict(X_predict)


"""
Split the X dataframe into training and testing data
"""
from sklearn.model_selection import train_test_split  
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.20) 

"""

"""
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, z_train) 
svm_y_pred = svclassifier.predict(X_test)
"""
Print out a tuple at each index position for the actual classification and the predicted classification
"""
for i in range(len(z_test)):
	print(svm_y_pred[i], z_test.values[i])

"""
Pair wise plot - a scatter plot of every pair of variables that gives a rough visual of how correlated (or not) every variable 
is with every other
"""
import seaborn as sns
sns.pairplot(X, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws':{'alpha':0.1}})
# plt.show()

"""
Based off the pair plot visualizations, it appears that 'bare nuclei' might be closest correlated with X2, X3, X4
Do a linear regression with just these three independent variables
"""
model3 = sm.OLS(y_known, np.transpose(np.transpose(X_known)[2:5]))
results3 = model3.fit()
print(results3.summary())

"""
ypred is the prediction of 'bare nuclei' for model3
"""
ypred = results3.predict(np.transpose(np.transpose(X_known)[2:5]))
plt.scatter(XX['Bare nuclei'], ypred, alpha=.1)
plt.xlabel('NEW Actual Bare nuclei')
plt.ylabel('Predicted Bare nuclei')
plt.show()

"""
Perform cross-validation for SVM classification model
"""

def bouncingBall(h, bounce, window):
    # your code
    # window = height (in meters) where mom is
    # bounce = % of preceding height after each bounce
    # h = original height (in meters) of where the ball dropped from
    nums_bounce = 0
    while h > window:
        h = h(bounce)
        nums_bounce += 2
    nums_bounce = nums_bouce - 1
    
    return nums_bounce



# """
# Zip provides a list of tuples, each with predicted bare nuclei value, and actual
# """
# import math
# total = 0
# for pred, actual in zip(ypred, XX['Bare nuclei']):
# 	total += abs(1 - pred/actual)
# 	print(pred/actual)

# print(total/len(ypred))


""" 
A function to cast a column of values into integer data type and pandas NaN data type if not possible
"""
# def cast_to_int(x):
# 	try:
# 		x = float(x)
# 	except:
# 		x = np.nan
# 	return x



# lm = linear_model.LinearRegression()
# model = lm.fit(X_train, y_train)
# predictions = lm.predict(X_test)


# plt.scatter("Clump Thickness", "Mitoses")