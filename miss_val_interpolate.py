import clean_data

"""
Function that uses pandas interpolate method to interpolate missing values
"""
def lin_interpol(file, encoding):

	#Call the function that returns the bcdata with the added T/F boolean column
	bcdata = clean_data.new_bool_col(file, encoding)

	#Call the linear interpolate method of the bcdata frame
	est_1 = bcdata.interpolate(method='linear')

	#Captures only the rows with newly interpolated values
	est_1_miss = est_1[est_1['Miss Non Miss']==False]

	return est_1_miss

def nearest_interpol(file, encoding):
	
	bcdata = clean_data.new_bool_col(file, encoding)

	est_2 = bcdata.interpolate(method='nearest')

	est_2_miss = est_2[est_2['Miss Non Miss']==False]

	return est_2_miss

def vals_interpol(file, encoding):
	
	bcdata = clean_data.new_bool_col(file, encoding)

	est_3 = bcdata.interpolate(method='values')

	est_3_miss = est_3[est_3['Miss Non Miss']==False]

	return est_3_miss

def index_interpol(file, encoding):

	bcdata = clean_data.new_bool_col(file, encoding)

	est_4 = bcdata.interpolate(method='index')

	est_4_miss = est_4[est_4['Miss Non Miss']==False]

	return est_4_miss

def display_interpols(file, encoding):

	#Call each of the interpolate method functions in order the get the returned interpolated values
	est_1_miss = lin_interpol(file, encoding)
	est_2_miss = nearest_interpol(file, encoding)
	est_3_miss = vals_interpol(file, encoding)
	est_4_miss = index_interpol(file, encoding)

	#Print out a tuple - each tuple provides the interpolated 'bare nuclei' value across each of the different methods for each
	#row originally containing a missing value 
	for i in range(len(est_1_miss)):
		print(est_1_miss.values[i][6], est_2_miss.values[i][6], est_3_miss.values[i][6], est_4_miss.values[i][6])