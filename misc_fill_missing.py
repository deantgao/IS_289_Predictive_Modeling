import clean_data
import pandas as pd

"""
Fill in all missing values with the mode value of the known bare nuclei values
"""
def fill_w_mode(file, encoding):

	#Create a dataframe with all the rows with missing values dropped
	bcdata_no_miss = clean_data.drop_miss(file, encoding)

	#Create a dataframe with all missing values replaced with the NaN datatype
	bcdata_w_miss = clean_data.miss_to_nan(file, encoding)

	#Print out a series that displays each possible value of bare nuclei, and the frequency of each value
	print(type(bcdata_w_miss['Bare nuclei'].value_counts()))

	#Create a numpy array that lists every value of the bare nuclei attribute that has a mode frequency
	mode = bcdata_no_miss['Bare nuclei'].mode().values

	#Creates a new dataframe with all of the missing values filled in with the previously determined'
	#mode value
	bcdata = bcdata_w_miss.fillna(mode[0])

	#Print out a list of tuples, each tuple representing each value of the bare nuclei attribute
	#containing the original frequency of that value and the adjusted frequency following replaced values
	print(zip(bcdata_w_miss['Bare nuclei'].value_counts().values, bcdata['Bare nuclei'].value_counts().values))

	return bcdata


"""
Fill in all missing values with the mean value of the known bare nuclei values
"""
def fill_w_avg(file, encoding):

	bcdata_no_miss = clean_data.drop_miss(file, encoding)
	bcdata_w_miss = clean_data.miss_to_nan(file, encoding)
	print(bcdata_w_miss['Bare nuclei'].value_counts())

	mean = int(bcdata_no_miss['Bare nuclei'].mean())

	bcdata = bcdata_w_miss.fillna(mean)
	print(zip(bcdata_w_miss['Bare nuclei'].value_counts().values, bcdata['Bare nuclei'].value_counts().values))

	return bcdata
