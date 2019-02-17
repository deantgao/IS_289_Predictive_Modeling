import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

"""
Read in the data from csv
"""
def import_data(file, encoding):

	#Read in the data from csv file
	bcdata = pd.read_csv(file)

	return bcdata

"""
Replace all unknown values, originally represented as "?" with an NaN missing value
"""
def miss_to_nan(file, encoding):

	#Call the function that imports data from csv
	bcdata = import_data(file, encoding)

	#Replace all values that have a "?" placeholder value with a NaN missing value
	bcdata['Bare nuclei'] = bcdata['Bare nuclei'].apply(lambda x: float(x) if x != "?" else np.nan)

	return bcdata

"""
Create a new dataframe with just the rows that contain missing values
"""
def subset_miss_data(file, encoding):

	#Call the function that imports data where missing values have been converted to NaN
	bcdata = miss_to_nan(file, encoding)

	#Get a summary of all the columns and the number of missing values in each
	bcdata.isnull().sum()
	#Get a summary of total number of missing values
	bcdata.isnull().sum().sum()

	#Create the subset dataframe with only rows with missing values
	miss_rows = bcdata[bcdata.isnull().any(axis=1)]
	return miss_rows

"""
Create a new dataframe from the subset dataframe of rows with missing values, and keep only the rows with non-missing values
"""
def subset_miss_data_x(file, encoding):

	bcdata = subset_miss_data(file, encoding)

	#Among the rows with missing values, drop the class label, sample ID, and bare nuclei columns
	known_to_predict = bcdata.drop(['Class label', 'Sample ID', 'Bare nuclei'], axis=1)

	return known_to_predict


"""
Create a new column of the original dataframe that indicates True/False depending on whether the value of the 'bare nuclei' column is an integer
"""
def new_bool_col(file, encoding):

	#Call the function that returns the read in data from csv
	bcdata = miss_to_nan(file, encoding)

	#Add a new column to the copy of the dataset that is either True or False on the condition of bare nuclei value as integer
	bcdata['Miss Non Miss'] = bcdata['Bare nuclei'].apply(lambda x: x.is_integer())

	return bcdata

"""
Drop all rows with missing values from the original dataframe
"""
def drop_miss(file, encoding):

	#Call the function that converts missing values to NaN type
	bcdata = miss_to_nan(file, encoding)

	#Drop any rows that contain an NaN
	bcdata = bcdata.dropna()

	return bcdata

def rem_cols_keepnan(file, encoding):

	bcdata = import_data(file, encoding)

	#Store a new dataframe that 




