"""
This applies the Support Vector Machine Classification ALgorithm to each of the various dataframes where missing values have 
been imputed, interpolated, deleted, or averaged
"""

import clean_data
import miss_vals_impute
import miss_vals_interpol
import misc_fill_missing
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  

"""
Train and test an SVM model fit to the complete breast cancer data using values that were originally missing but were interpolated
using the linear interpolate method in sklearn
"""
def svc_lin_interpol(file, encoding):

	#Creates a new dataframe with the first item of what is returned from the linear method interpolation
	bcdata = miss_vals_interpol.lin_interpol(file, encoding)[0]

	#Subsets the above dataframe, split into the bare nuclei values and all other attribute values
	all_data = bcdata.drop(['Miss Non Miss', 'Sample ID', 'Class label'], axis=1)
	class_data = bcdata['Class label']

	#Split the subset data (bare nuclei and all other attributes) into training and testing sets 
	all_train, all_test, class_train, class_test = train_test_split(all_data, class_data, test_size = 0.20) 

	#Instantiate the SVC model using linear kernel
	svclassifier = SVC(kernel='linear')  
	#Fit SVC model to the training data
	svclassifier.fit(all_train, class_train) 
	#Use the SVC model fit to the training data to predict the class label of the test set
	svm_y_pred = svclassifier.predict(all_test)

	#Create a list of tuples, where each tuple is a side-by-side comparison of the actual class label, compared with the predicted class
	#label of the SVM
	test_v_pred = zip(class_test.values, svm_y_pred)
	print(test_v_pred)
	
	#Determine the approximate accuracy of the SVM model fit to this data
	#Initialize a count that will track the number of times the model predicts an inaccurate class label
	count = 0
	#Iterate through the list of zipped tuples
	for tup in test_v_pred:
		#If the first item of the tuple does not equal the second item, increment the count up by 1
		if tup[0] != tup[1]:
			count += 1

	return class_test.values, svm_y_pred, count

"""
Train and test an SVM model fit to the complete breast cancer data using values that were originally missing but were interpolated
using the nearest interpolate method in sklearn
"""
def svc_nearest_interpol(file, encoding):

	#Creates a new dataframe with the first item of what is returned from the nearest method interpolation
	bcdata = miss_vals_interpol.nearest_interpol(file, encoding)[0]

	#Subsets the above dataframe, split into the bare nuclei values and all other attribute values
	all_data = bcdata.drop(['Miss Non Miss', 'Sample ID', 'Class label'], axis=1)
	class_data = bcdata['Class label']

	#Split the subset data (bare nuclei and all other attributes) into training and testing sets 
	all_train, all_test, class_train, class_test = train_test_split(all_data, class_data, test_size = 0.20) 

	#Instantiate the SVC model using linear kernel
	svclassifier = SVC(kernel='linear')  
	#Fit SVC model to the training data
	svclassifier.fit(all_train, class_train) 
	#Use the SVC model fit to the training data to predict the class label of the test set
	svm_y_pred = svclassifier.predict(all_test)

	#Create a list of tuples, where each tuple is a side-by-side comparison of the actual class label, compared with the predicted class
	#label of the SVM
	test_v_pred = zip(class_test.values, svm_y_pred)
	print(test_v_pred)
	
	#Determine the approximate accuracy of the SVM model fit to this data
	#Initialize a count that will track the number of times the model predicts an inaccurate class label
	count = 0
	#Iterate through the list of zipped tuples
	for tup in test_v_pred:
		#If the first item of the tuple does not equal the second item, increment the count up by 1
		if tup[0] != tup[1]:
			count += 1

	return class_test.values, svm_y_pred, count

"""
Train and test an SVM model fit to the complete breast cancer data using values that were originally missing but were replaced
with the average value of the bare nuclei values
"""
def svc_use_avg(file, encoding):

	#Creates a new dataframe that is the original bcdata dataframe with all missing values filled with the
	#mean value of the bare nuclei values
	bcdata = misc_fill_missing.fill_w_avg(file, encoding)

	#Subsets the above dataframe, split into the bare nuclei values and all other attribute values
	all_data = bcdata.drop(['Sample ID', 'Class label'], axis=1)
	class_data = bcdata['Class label']

	#Split the subset data (bare nuclei and all other attributes) into training and testing sets 
	all_train, all_test, class_train, class_test = train_test_split(all_data, class_data, test_size = 0.20) 

	#Instantiate the SVC model using linear kernel
	svclassifier = SVC(kernel='linear')  
	#Fit SVC model to the training data
	svclassifier.fit(all_train, class_train) 
	#Use the SVC model fit to the training data to predict the class label of the test set
	svm_y_pred = svclassifier.predict(all_test)

	#Create a list of tuples, where each tuple is a side-by-side comparison of the actual class label, compared with the predicted class
	#label of the SVM
	test_v_pred = zip(class_test.values, svm_y_pred)
	print(test_v_pred)
	
	#Determine the approximate accuracy of the SVM model fit to this data
	#Initialize a count that will track the number of times the model predicts an inaccurate class label
	count = 0
	#Iterate through the list of zipped tuples
	for tup in test_v_pred:
		#If the first item of the tuple does not equal the second item, increment the count up by 1
		if tup[0] != tup[1]:
			count += 1

	return class_test.values, svm_y_pred, count

"""
Train and test an SVM model fit to the complete breast cancer data using values that were originally missing but were replaced
with the mode value of the bare nuclei values
"""
def svc_use_mode(file, encoding):

	#Creates a new dataframe that is the original bcdata dataframe with all missing values filled with the
	#mode value of the bare nuclei values
	bcdata = misc_fill_missing.fill_w_mode(file, encoding)

	#Subsets the above dataframe, split into the bare nuclei values and all other attribute values
	all_data = bcdata.drop(['Sample ID', 'Class label'], axis=1)
	class_data = bcdata['Class label']

	#Split the subset data (bare nuclei and all other attributes) into training and testing sets 
	all_train, all_test, class_train, class_test = train_test_split(all_data, class_data, test_size = 0.20) 

	#Instantiate the SVC model using linear kernel
	svclassifier = SVC(kernel='linear')  
	#Fit SVC model to the training data
	svclassifier.fit(all_train, class_train) 
	#Use the SVC model fit to the training data to predict the class label of the test set
	svm_y_pred = svclassifier.predict(all_test)

	#Create a list of tuples, where each tuple is a side-by-side comparison of the actual class label, compared with the predicted class
	#label of the SVM
	test_v_pred = zip(class_test.values, svm_y_pred)
	print(test_v_pred)
	
	#Determine the approximate accuracy of the SVM model fit to this data
	#Initialize a count that will track the number of times the model predicts an inaccurate class label
	count = 0
	#Iterate through the list of zipped tuples
	for tup in test_v_pred:
		#If the first item of the tuple does not equal the second item, increment the count up by 1
		if tup[0] != tup[1]:
			count += 1

	return class_test.values, svm_y_pred, count

"""
Train and test an SVM model fit to the subset of the breast cancer data where there were no missing values
"""
def svc_drop_miss(file, encoding):

	#Creates a new dataframe that is the original bcdata dataframe with all rows with missing values dropped
	bcdata = clean_data.drop_miss(file, encoding)

	#Subsets the above dataframe, split into the bare nuclei values and all other attribute values
	all_data = bcdata.drop(['Sample ID', 'Class label'], axis=1)
	class_data = bcdata['Class label']

	#Split the subset data (bare nuclei and all other attributes) into training and testing sets 
	all_train, all_test, class_train, class_test = train_test_split(all_data, class_data, test_size = 0.20) 

	#Instantiate the SVC model using linear kernel
	svclassifier = SVC(kernel='linear')  
	#Fit SVC model to the training data
	svclassifier.fit(all_train, class_train) 
	#Use the SVC model fit to the training data to predict the class label of the test set
	svm_y_pred = svclassifier.predict(all_test)

	#Create a list of tuples, where each tuple is a side-by-side comparison of the actual class label, compared with the predicted class
	#label of the SVM
	test_v_pred = zip(class_test.values, svm_y_pred)
	print(test_v_pred)
	
	#Determine the approximate accuracy of the SVM model fit to this data
	#Initialize a count that will track the number of times the model predicts an inaccurate class label
	count = 0
	#Iterate through the list of zipped tuples
	for tup in test_v_pred:
		#If the first item of the tuple does not equal the second item, increment the count up by 1
		if tup[0] != tup[1]:
			count += 1

	return class_test.values, svm_y_pred, count




# """
# Print out a tuple at each index position for the actual classification and the predicted classification
# """
# for i in range(len(z_test)):
# 	print(svm_y_pred[i], z_test.values[i])

# """
# Based off the pair plot visualizations, it appears that 'bare nuclei' might be closest correlated with X2, X3, X4
# Do a linear regression with just these three independent variables
# """
# model3 = sm.OLS(y_known, np.transpose(np.transpose(X_known)[2:5]))
# results3 = model3.fit()
# print(results3.summary())

# """
# ypred is the prediction of 'bare nuclei' for model3
# """
# ypred = results3.predict(np.transpose(np.transpose(X_known)[2:5]))
# plt.scatter(XX['Bare nuclei'], ypred, alpha=.1)
# plt.xlabel('NEW Actual Bare nuclei')
# plt.ylabel('Predicted Bare nuclei')
# plt.show()

