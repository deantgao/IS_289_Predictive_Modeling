import clean_data
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import seaborn as sns

"""
Create a new dataframe that includes only the columns of interest to use as independent variables in linear regression
"""
def given_to_impute(file, encoding):

	bcdata = clean_data.miss_to_nan(file, encoding)

	#Drop any rows with missing values
	bcdata = bcdata.dropna()

	#Drop the class label, sample ID, and bare nuclei columns
	bcdata = bcdata.drop(['Class label', 'Sample ID', 'Bare nuclei'], axis=1)

	#Creat a numpy array that contains all the values of the above subset dataframe
	all_given = bcdata.values

	return all_given

"""
Create a new dataframe series that includes only the column of interest to use as dependent variable in linear regression
"""
def missing_to_impute(file, encoding):

	bcdata = clean_data.miss_to_nan(file, encoding)

	#Drop any rows with missing values
	bcdata = bcdata.dropna()

	#Create a new dataframe series that contains only the bare nuclei column of data
	bn_given = bcdata['Bare nuclei']

	return bn_given

"""
Use the sklearn built-in linear regression tool to regress the bare nuclei variable on all the other attributes to determine potential 
correlation between variables
"""
def sklearn_lin_reg(file, encoding):

	#Call the functions that return the attributes with all known values and the bare nuclei attribute with missing values
	all_given = given_to_impute(file, encoding)
	bn_given = missing_to_impute(file, encoding)

	model = LinearRegression()
	model.fit(all_given, bn_given)

	print(model1.intercept_, model1.coef_)

"""
Use the stats model built-in ordinary least squares (linear regression) tool to predict what the missing bare nuclei values given
a linear model determined by regressing bare nuclei on all other known attributes
"""
def statsmod_lin_reg(file, encoding):

	#Create a new dataframe to use in this function that calls the function that returns a dataframe of just the rows of missing data
	bn_to_predict = clean_data.subset_miss_data(file, encoding)
	
	#Create a new dataframe that calls the function that returns a dataframe of the original bcdata with all missing values replaced with NaN
	miss_to_nan = clean_data.miss_to_nan(file, encoding)

	#Create an array that calls the function that returns a numpy array without rows with missing values and including only the predictor attrbi
	all_given = given_to_impute(file, encoding)
	bn_given = missing_to_impute(file, encoding)
	
	#Instantiate an ordinary least squares model that takes in the parameters of 
	#(all the known values of the bare nuclei attribute, all the rows of known values for remaining attributes for rows with no NaN)
	model = sm.OLS(bn_given, all_given)
	
	#Fits an OLS model to the input data
	results = model.fit()

	#Print out a stats summary of the model fit
	print(results.summary())

	return results, miss_to_nan

"""
1) Use the stats model library to fit a linear regression model using known values of bare nuclei, and known values of all the other attributes
*This will fit a model that uses all known values for its predictive capacity, but will allow for the prediction of known 
values of bare nuclei, which allows to check the model for its accuracy
2) Use the model to predict the missing bare nuclei values using the known values of the remaining attributes
"""
def statsmod_lin_reg_predict(file, encoding):

	results = statsmod_lin_reg(file, encoding)[0]
	miss_to_nan = statsmod_lin_reg(file, encoding)[1]
	all_given = given_to_impute(file, encoding)
	known_to_predict = clean_data.subset_miss_data_x(file, encoding)

	#Predict the bare nuclei values (that are already known) using the values across the other attributes
	bn_known_preds = results.predict(all_given)

	#Create a scatter plot of the known bare nuclei values, and the bare nuclei values that were predicted by the model
	plt.scatter(clean_data.drop_miss(file, encoding)['Bare nuclei'], bn_known_preds, alpha=.1)

	#Label the x-axis of the scatter plot
	plt.xlabel('Actual Bare nuclei')

	#Label the y-axis of the scatter plot
	plt.ylabel('Predicted Bare nuclei')

	#Invoke the visualization of the scatter plot
	plt.show()

	#Predict the missing bare nuclei values using the known values of the other attributes
	bn_unknown_preds = results.predict(known_to_predict)
	print(bn_unknown_preds)

	#Create a subset of the previously created dataframe, with all missing values dropped, that includes only the records 
	#with a malignant diagnosis
	mal_cases = miss_to_nan[miss_to_nan['Class label']==4]

	#Among the records with malignant diagnosis, create a scatter plot of the normal nucleoli values and the bare nuclei values
	#to determine if there appears to be any correlation
	plt.scatter(mal_cases['Normal nucleoli'], mal_cases['Bare nuclei'], alpha=.1)
	plt.xlabel('Normal nucleoli')
	plt.ylabel('Bare nuclei')
	plt.show()

"""
Plots a color map to visualize any patterns in values between attributes
"""
def color_map(file, encoding):

	drop_miss = clean_data.drop_miss(file, encoding)
	bcdata = drop_miss.drop(['Class label', 'Sample ID'], axis=1)

	#Creates a segmented line chart/color map. Each value across the x-axis represents one attribute
	plt.imshow(bcdata, aspect='auto')
	plt.colorbar()
	plt.xlabel('BC Data Attributes')
	plt.ylabel('Patient Record')
	plt.show()

"""
Pair wise plot - a scatter plot of every pair of variables that gives a rough visual of how correlated (or not) every variable 
is with every other
"""
def pair_plot(file, encoding):	
	
	bcdata = clean_data.drop_miss(file, encoding)
	bcdata = bcdata.drop(['Sample ID', 'Class label'], axis=1)
	print(bcdata)

	#Instantiate a pairplot for all data except for Sample ID
	sns.pairplot(bcdata, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws':{'alpha':0.1}})
	plt.show()

"""
STATS NOTES:
- Closer to 1 r-squared is better
- If p-value (P>|t|) is less than .05, indicates significance
"""



