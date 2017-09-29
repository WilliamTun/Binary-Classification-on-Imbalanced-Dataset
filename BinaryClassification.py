from __future__ import division
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
mydata = pd.read_csv('dataCSV.csv') #index_col=False



# This function boot stram samples from the data you give it 
# and can create a given number of boot strap samples
def bootstrap_it(dataIn, num_iter, num_samples):
	b_samples = list()
	while len(b_samples) < num_samples:
		sampled_data = dataIn.sample(n=num_iter, replace = True) 
		b_samples.append(sampled_data)
	return b_samples

BootOrNot = raw_input("Create BootStrap Dataset? (Y/N): ") 

if (BootOrNot == "Y"):
	sam_size = int(raw_input("enter total num rows to collect: ")) 
	mydata_untouched = mydata  # for LOOCV later
	mydata2 = bootstrap_it(mydata, sam_size, 1)
	mydata = mydata2[0] 


# 1- change Clinic to numeric representation

def dataPreProcess(dataIn):
	# 1- change Clinic to numeric representation
	dataIn['Clinic'] = dataIn['Clinic'].map({'GH': 1.0, 'CS': 2.0, 'Not cancer clinic': 3.0})
	#dataIn['Cancer'] = dataIn['Cancer'].map({'Yes': 1, 'No': 0})
	dataIn['Sex'] = dataIn['Sex'].map({'F': 1.0, 'M': 0.0})
	dataIn['Blood in urine'] = dataIn['Blood in urine'].map({'No': 0.0, 'Yes': 1.0, 'MH': 2.0})

	dataIn = dataIn[dataIn['Cancer'].notnull()]
	# 2- for now, drop Op, Grade, Stage, Clinic
	list1 = ['Project', 'Cytoplasm Area percentage of Cell']  # list of columns to drop
	dataIn = dataIn.drop(list1, 1) # dataframe.drop allows column to be dropped. 1 = columns, 0 - rows

# CHECK TYPE OF EACH COLUMN - float64, object
#print(X.dtypes) 

# Select columns with specific datatype
#X = X.select_dtypes(include=['object'])
#X = X.select_dtypes(include=['float64']) # SELECT FOR FLOATS ONLY - WITHOUT THIS, LOGISITC REGRESSION CANNOT RUN. 

	dataIn[dataIn['Cytoplasm Area percentage of MCM2 +ve CELL']=='undefined'] = 0.0  
	dataIn[dataIn['Cytoplasm Area percentage of MCM2 panCK +ve CELL']=='undefined'] = 0.0  
	dataIn['Cytoplasm Area percentage of MCM2 +ve CELL'] = dataIn['Cytoplasm Area percentage of MCM2 +ve CELL'].astype('float64')
	dataIn['Cytoplasm Area percentage of MCM2 panCK +ve CELL'] = dataIn['Cytoplasm Area percentage of MCM2 panCK +ve CELL'].astype('float64')

	for index, row in dataIn['Sex'].iteritems():
		if row != 0.0 and row != 1.0:
			dataIn['Sex'][index] = 2.0
	for index, row in dataIn['Clinic'].iteritems():
		if row != 1.0 and row != 2.0 and row != 3.0:
			dataIn['Clinic'][index] = 0.0
	return(dataIn)


mydata = dataPreProcess(mydata)


if (BootOrNot == "Y"):
	mydata_untouched = dataPreProcess(mydata_untouched)






'''
mydata_untouched['Clinic_num'] = mydata_untouched['Clinic'].map({'GH': 1, 'MH': 2})
mydata_untouched['Cancer'] = mydata_untouched['Cancer'].map({'Yes': 1, 'No': 0})
mydata_untouched= mydata_untouched[mydata_untouched['Cancer'].notnull()]
# 2- for now, drop Op, Grade, Stage, Clinic
list1 = ['scene_name', 'Op', 'Grade', 'Stage', 'Clinic']  # list of columns to drop
mydata_untouched = mydata_untouched.drop(list1, 1) # dataframe.drop allows column to be dropped. 1 = columns, 0 - rows


# 1- change Clinic to numeric representation
mydata['Clinic_num'] = mydata['Clinic'].map({'GH': 1, 'MH': 2})
mydata['Cancer'] = mydata['Cancer'].map({'Yes': 1, 'No': 0})
mydata = mydata[mydata['Cancer'].notnull()]
# 2- for now, drop Op, Grade, Stage, Clinic
list1 = ['scene_name', 'Op', 'Grade', 'Stage', 'Clinic']  # list of columns to drop
mydata = mydata.drop(list1, 1) # dataframe.drop allows column to be dropped. 1 = columns, 0 - rows

'''



# Does the data differ between the two clinics? 
# Clinics SHOULD have same procedure for gathering data - but human error may still introduce bias
# Clinic based questions:
# Do false positive or negative rates change from clinic to clinic? 
# Does one model perform better from one clinic over the other clinic?

'''
split_via_Clinic = raw_input("Input to use (ALL/GH/CS/NotCancerClinic): ") # train data on all data/specific clinic? 
if (split_via_Clinic == 'GH'):
	mydata = mydata.drop(mydata[mydata.Clinic == 1].index)
elif (split_via_Clinic == 'CS'):
	mydata = mydata.drop(mydata[mydata.Clinic == 2].index)
else:
	mydata = mydata.drop(mydata[mydata.Clinic == 3].index)

'''
### NOTE - PROBLEM WITH GH is that all but 1 case is cancer - YES


#print(mydata['Cancer'].value_counts())

# Step 4, split data into training set and test set
from sklearn.cross_validation import train_test_split 
# Split data into labels and features
X = mydata.drop('Cancer', 1)           #must get rid of GOOD column
Y = mydata['Cancer']                   #response column only GOOD



'''
# Split into training and test set 

test_size_question = raw_input("Enter fraction to which the data should be split to create the TEST set:\n(0.33 for balanced or 0.1 for imbalanced labels): ")
test_size_question = float(test_size_question)

'''





#X = X.select_dtypes(include=['object'])



# Random split of test + training set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state = 666)


#LogOutPut = DoLogReg(X_train, y_train, X_test)
#y_predictions = LogOutPut[0]
#resub_predictions = LogOutPut[1]

from sklearn.linear_model import LogisticRegression

#mydata = mydata.astype(float)
#mydata.loc[mydata['Cytoplasm Area percentage of MCM2 +ve CELL'] == 'undefined'] = 0.0
#mydata.loc[mydata['Cytoplasm Area percentage of PanCK +ve CELL'] == 'undefined'] = 0.0
#mydata.loc[mydata['Cytoplasm Area percentage of MCM2 panCK +ve CELL'] == 'undefined'] = 0.0


#print(mydata["Cytoplasm Area percentage of MCM2 +ve CELL"])

'''
model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)
Log_y_predictions = model.predict(X_test)
Log_resub_predictions = model.predict(X_train)
print(Log_y_predictions)
'''

## without NaN generated by clinic and sex columns, regression can work



'''
# STRATIFIED SHUFFLE SPLIT of Test and Training set 
from sklearn.cross_validation import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(y = Y, n_iter=1, test_size=test_size_question, random_state=666)
for train_index, test_index in sss:
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

# it is current splitting columns, not rows

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
'''




# Step 5, normalize data into values between 0 and 1 
from sklearn.preprocessing import StandardScaler
normalize_or_not = raw_input("Normalize features?(Y/N): ")
if (normalize_or_not == 'Y'):
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

# Step 6, Dimensionality with feature extraction: PCA - if data is linearly separable
from sklearn.decomposition import PCA

PCA_or_not = raw_input("Perform PCA? (Y / N): ")
if (PCA_or_not == 'Y'):
	pca = PCA(n_components = None)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	explained_variance = pca.explained_variance_ratio_ #contain percentage of variance explained by each of the extracted principle componenets
	cumulative_explained_variance = np.cumsum(explained_variance)
	print(explained_variance)
	plt.plot(cumulative_explained_variance)
	plt.title('Cumulative variance explained by principle componenets')
	plt.ylabel('Variance explained')
	plt.xlabel('Principle componenet index')
	plt.show()

	num_PrincipleComponents = int(raw_input("Desired number of Principle Components: ")) 
	pca = PCA(n_components = num_PrincipleComponents)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	explained_variance = pca.explained_variance_ratio_ #contain percentage of variance explained by each of the extracted principle componenets
	cumulative_explained_variance = np.cumsum(explained_variance)
	print(explained_variance)
	plt.plot(cumulative_explained_variance)
	plt.title('Cumulative variance explained by principle componenets')
	plt.ylabel('Variance explained')
	plt.xlabel('Principle componenet index')
	plt.show()





# SVM packages
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search

# Tree methods
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#K nearest neightbout
from sklearn.neighbors import KNeighborsClassifier 

#Traditional approaches
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Ensembl methods: Adaboost
from sklearn.ensemble import AdaBoostClassifier

# Ensembl methods: gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

# Logistic Regression Function
def DoLogReg(X_trainIn, y_trainIn, X_testIn):
	model = LogisticRegression(random_state = 0)
	model.fit(X_trainIn, y_trainIn)
	Log_y_predictions = model.predict(X_testIn)
	Log_resub_predictions = model.predict(X_trainIn)
	return(Log_y_predictions, Log_resub_predictions)

def DoNaiveBayes(X_trainIn, y_trainIn, X_testIn):
	model = GaussianNB()
	model.fit(X_trainIn, y_trainIn)
	NB_y_predictions = model.predict(X_testIn)
	NB_resub_predictions = model.predict(X_trainIn)
	return(NB_y_predictions, NB_resub_predictions)

def DoSVM(X_trainIn, y_trainIn, X_testIn):
	if SVM_kernel == 'poly':
		model = SVC(C = optimal_C, kernel = SVM_kernel, degree=SVM_degree, gamma=optimal_gamma, random_state = 0)
	else:
		model = SVC(C = optimal_C, kernel = SVM_kernel, gamma=optimal_gamma, random_state = 0)	
		model.fit(X_trainIn, y_trainIn)
		SVM_y_predictions = model.predict(X_testIn)
		SVM_resub_predictions = model.predict(X_trainIn)
	return(SVM_y_predictions, SVM_resub_predictions)

def DoDTree(X_trainIn, y_trainIn, X_testIn):
	model = DecisionTreeClassifier(criterion= DT_InfoGain_criterion, max_depth = DTreebestMaxDepth, max_features= DTreebestMaxFeat)
	model.fit(X_trainIn, y_trainIn)
	D_y_predictions = model.predict(X_testIn)
	D_resub_predictions = model.predict(X_trainIn)
	return(D_y_predictions, D_resub_predictions)

def DoRForest(X_trainIn, y_trainIn, X_testIn):
	model = RandomForestClassifier(criterion=RF_InfoGain_criterion, n_estimators = RForbestNDepth, max_features= RForbestMaxFeat)
	model.fit(X_trainIn, y_trainIn)
	RF_y_predictions = model.predict(X_testIn)
	RF_resub_predictions = model.predict(X_trainIn)
	return(RF_y_predictions, RF_resub_predictions)

def Do_KNearest(X_trainIn, y_trainIn, X_testIn):
	model = KNeighborsClassifier(n_neighbors = optimized_K_param)  # Fit KNearest with optimized K parameter
	model.fit(X_trainIn, y_trainIn)
	KN_y_predictions = model.predict(X_testIn)
	KN_resub_predictions = model.predict(X_trainIn)
	return(KN_y_predictions, KN_resub_predictions)

def DO_AdaBoost(X_trainIn, y_trainIn, X_testIn):
	model = AdaBoostClassifier(n_estimators= n_estimators_best)
	model.fit(X_trainIn, y_trainIn)
	Ay_predictions = model.predict(X_testIn)
	Ay_resub_predictions = model.predict(X_trainIn)
	return(Ay_predictions, Ay_resub_predictions)

def DO_GradBoost(X_trainIn, y_trainIn, X_testIn):
	model = GradientBoostingClassifier(n_estimators=n_estimators_best, learning_rate=1.0, max_depth=max_depth_best, random_state=0) 
	model.fit(X_train, y_train)
	Gy_predictions = model.predict(X_test)
	Gy_resub_predictions = model.predict(X_train)
	return(Gy_predictions, Gy_resub_predictions)


def AverageOutResults(ListIn):
	sumindex = [sum(elts) for elts in zip(*ListIn)] # sums every corresponding index in a list of lists
	max_num_in_sumindex = float(max(sumindex))  # the maximum index of a list
	# average out the combined predictions 
	prediction = [x / max_num_in_sumindex for x in sumindex]
	for i in range(0, len(prediction)):
		if prediction[i] > 0.5: 
			prediction[i] = int(1)
		else:
			prediction[i] = int(0)
#convert new_ans from list to numpy array
	output_predictions = np.asarray(prediction)
	return(output_predictions)



# Now ask for input
model_choice = raw_input("Enter Model \n (SVM/ANN/RForest/DTree/KNearest/LogReg/NaiveBayes/AdaBoost/GradientBoosting/Stack): ") # or `input("Some...` in python 3

if (model_choice == 'SVM'):
	SVM_kernel = raw_input("Kernel to use (linear/poly/rbf/sigmoid): ") 

	if SVM_kernel == 'poly':
		SVM_degree = int(raw_input("degree Polynomial (3): ")) #Degree: = polynomial degree
#		model = SVC(kernel = SVM_kernel, degree = SVM_degree, random_state = 0)
#		model.fit(X_train, y_train)
#		y_predictions = model.predict(X_test)
#	else: 
#		model = SVC(kernel = SVM_kernel, random_state = 0)
#		model.fit(X_train, y_train)
#		y_predictions = model.predict(X_test)
	
	optimize_SVM = raw_input("Optimize parameters? Y (gridsearch))/N (manual): ")
	if optimize_SVM == 'Y':
		param_grid = {'C':[0.1,1,10], 'gamma':[1,0.1,0.01]} # put in values of c and gamma parameters we wish to test
		fresh = raw_input("reuse model? Y/N: ")
		if(fresh == 'N'):
			model = GridSearchCV(SVC(), param_grid, verbose=3) #make sure verbose is not 0
			model.fit(X_train, y_train)
			y_predictions = model.predict(X_test)
			resub_predictions = model.predict(X_train)
		else:
			if SVM_kernel == 'poly':
				model = SVC(kernel = SVM_kernel, degree=SVM_degree, random_state = 0)
			else:
				model = SVC( kernel = SVM_kernel, random_state = 0)
			model = GridSearchCV(model, param_grid, verbose=3) #make sure verbose is not 0
			model.fit(X_train, y_train)
			y_predictions = model.predict(X_test)
#			print("\nBest parameters:\n", model.best_params_)
#			print("\nBest estimators:\n", model.best_estimator_)
			print("best C parameter:", model.best_params_['C'])
			print("best gamma parameter:", model.best_params_['gamma'])
			optimal_C = model.best_params_['C']
			optimal_gamma = model.best_params_['gamma']

			SVM_OutPut = DoSVM(X_train, y_train, X_test)
			y_predictions = SVM_OutPut[0]
			resub_predictions = SVM_OutPut[1]


	if optimize_SVM == 'N':
		optimal_C =  float(raw_input("C parameter eg. [0.1, 1, 10, 100]: "))
		optimal_gamma = float(raw_input("gamma parameter eg. [1, 0.1, 0.01, 0.001]: "))
		SVM_OutPut = DoSVM(X_train, y_train, X_test)
		y_predictions = SVM_OutPut[0]
		resub_predictions = SVM_OutPut[1]


elif (model_choice == 'DTree'):
	DT_InfoGain_criterion = raw_input("DTree Info gain: (entropy/gini): ")
	optimize_DTree = raw_input("Optimize parameters? Y (gridsearch))/N (manual): ")
	if optimize_DTree == 'Y':
		
		parameters = {'max_depth':[3,4,5,6,7,8,9,10], 'max_features':[3,4,5,6,7,8,9,10],}
		model = grid_search.GridSearchCV(DecisionTreeClassifier(criterion = DT_InfoGain_criterion), parameters)
		model.fit(X_train, y_train)
		y_predictions = model.predict(X_test)
		print("\nBest parameters:\n")
		print(model.best_params_)
		print("\nBest estimators:\n")
		print(model.best_estimator_)
		DTreebestMaxDepth = model.best_params_['max_depth']
		DTreebestMaxFeat = model.best_params_['max_features']
		
		D_OutPut = DoDTree(X_train, y_train, X_test)
		y_predictions = D_OutPut[0]
		resub_predictions = D_OutPut[1]


	if optimize_DTree == 'N':
		DTreebestMaxDepth = int(raw_input("MaxDepth Param eg. [3,4,5,6]: "))
		DTreebestMaxFeat = int(raw_input("MaxFeat Param eg [3,4,5,6]: "))
		D_OutPut = DoDTree(X_train, y_train, X_test)
		y_predictions = D_OutPut[0]
		resub_predictions = D_OutPut[1]

elif (model_choice == 'RForest'):
	#num_estimators = int(raw_input("Enter num estimators (200): ")) 
	#model = RandomForestClassifier(n_estimators=num_estimators)
	#model.fit(X_train, y_train)
	#y_predictions = model.predict(X_test)
	#
	RF_InfoGain_criterion = raw_input("RForest info gain: (entropy/gini): ")
	optimize_RFor = raw_input("Optimize parameters? Y (gridsearch))/N (manual): ")
	if optimize_RFor == 'Y':
		parameters = {'n_estimators':[80, 90, 100, 110, 120], 'max_features':[3,4,5]}  # max features to do with overfitting # n estimators - compulsory variable
		model = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
		model.fit(X_train, y_train)
		y_predictions = model.predict(X_test)
		print("\nBest parameters:\n")
		print(model.best_params_)
		print("\nBest estimators:\n")
		print(model.best_estimator_)
		RForbestNDepth = model.best_params_['n_estimators']
		RForbestMaxFeat = model.best_params_['max_features']
		RF_OutPut = DoRForest(X_train, y_train, X_test)
		y_predictions = RF_OutPut[0]
		resub_predictions = RF_OutPut[1]
	if optimize_RFor == 'N':
		RForbestNDepth=  int(raw_input("n_estimators EG. 80, 90, 100: "))
		RForbestMaxFeat = int(raw_input("max_features eg. 3,4,5: "))
		#RFinforGain = raw_input("max_features eg. 3,4,5: ")
		RF_OutPut = DoRForest(X_train, y_train, X_test)
		y_predictions = RF_OutPut[0]
		resub_predictions = RF_OutPut[1]

### If we use random forrest in R, the 
### important(random forrest)
### varImpPlot # get yourself a model - you get plots of variable importance 


elif (model_choice == 'KNearest'):	
	# TEST K PARAMETER: 
	error_rate = [] #intialize error rate array to collect error rates from models with df KNN K parameter for loop
	##check k values from 1 to 40
	for i in range(1,40):
		knn = KNeighborsClassifier(n_neighbors=i) #create a KNN model per iteration
		knn.fit(X_train, y_train)
		pred_i = knn.predict(X_test)
		error_rate.append(np.mean(pred_i != y_test)) #take numpy mean and append error rate to error rate array
	plt.figure(figsize=(10,6))
	plt.plot(range(1,40), error_rate, color = 'blue',linestyle='dashed', marker='o', markerfacecolor= 'red', markersize = 10)
	plt.title('Error Rate vs K value')
	plt.xlabel('K')
	plt.ylabel('Error Rate')
	plt.show()
	#FIT K Nearest Neightbour model with best K value
	optimized_K_param = int(raw_input("Enter K parameter: "))
	KN_OutPut = Do_KNearest(X_train, y_train, X_test)
	y_predictions = KN_OutPut[0]
	resub_predictions = KN_OutPut[1]


elif (model_choice == 'LogReg'):
	LogOutPut = DoLogReg(X_train, y_train, X_test)
	y_predictions = LogOutPut[0]
	resub_predictions = LogOutPut[1]

elif (model_choice == 'NaiveBayes'):
	NB_output = DoNaiveBayes(X_train, y_train, X_test)
	y_predictions = NB_output[0]
	resub_predictions = NB_output[1]
elif (model_choice == 'ANN'):
	#Artificial Neural Network packages
	import keras 
	from keras.models import Sequential
	from keras.layers import Dense
	X_shape = X_train.shape  # IMPORTANT FOR INPUT LAYER PARAMETER!!!!
	num_features = X_shape[1]
	#initialize ANN by defining it as a sequence of layers
	model = Sequential()
	#Adding input layer
	#activation function parameter set to 'relu' = rectifier function 
	#uniform = intialization method
	model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = num_features))
	# Add one hidden layer
	model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
	# Add output layer
	model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
	# Apply stochastic gradient descent
	model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
	# Choose number of epochs 
	batch_param = int(raw_input("Enter No.samples per gradient update.(10): "))
	nb_epoch_param = int(raw_input("Enter No.epochs to train model(50): "))
	model.fit(X_train, y_train, batch_size=batch_param, nb_epoch = nb_epoch_param)
	y_predictions = model.predict(X_test)
	y_predictions = (y_predictions > 0.5)  #if y_pred is larger than 0.5, it returns CANCER TRUE, if not, returns false
	resub_predictions = model.predict(X_train)
elif (model_choice == 'AdaBoost'):
	optimize_ada = raw_input("Optimize parameters? Y (gridsearch))/N (manual): ")
	if (optimize_ada == 'Y'):
		parameters = {'n_estimators':[25,50,100,150,175]}
		model = grid_search.GridSearchCV(AdaBoostClassifier(), parameters)
		model.fit(X_train, y_train)
		y_predictions = model.predict(X_test)
		print("\nBest parameters:\n")
		print(model.best_params_)
		print("\nBest estimators:\n")
		print(model.best_estimator_)
		n_estimators_best = model.best_params_['n_estimators']
		Ada_out = DO_AdaBoost(X_train, y_train, X_test)
		y_predictions = Ada_out[0]
		resub_predictions = Ada_out[1]
	if (optimize_ada == 'N'):
		n_estimators_best = int(raw_input("n estimators Param eg. [100]: "))
		Ada_out = DO_AdaBoost(X_train, y_train, X_test)
		y_predictions = Ada_out[0]
		resub_predictions = Ada_out[1]

elif (model_choice == 'GradientBoosting'):
	optimize_GB = raw_input("Optimize parameters? Y (gridsearch))/N (manual): ")
	if (optimize_GB == 'Y'):
		parameters = {'n_estimators':[25,50,100,150,175], 'max_depth': [3,4,5,6,7]}
		model = grid_search.GridSearchCV(GradientBoostingClassifier(), parameters)
		model.fit(X_train, y_train)
		y_predictions = model.predict(X_test)
		print("\nBest parameters:\n")
		print(model.best_params_)
		print("\nBest estimators:\n")
		print(model.best_estimator_)
		n_estimators_best = model.best_params_['n_estimators']
		max_depth_best = model.best_params_['max_depth']
		Grad_out = DO_GradBoost(X_train, y_train, X_test)
		y_predictions = Grad_out[0]
		resub_predictions = Grad_out[1]
	if (optimize_GB == 'N'):
		n_estimators_best = int(raw_input("n estimators Param eg. [100]: "))
		max_depth_best = int(raw_input("max depth: eg. [3]" ))
		Grad_out = DO_GradBoost(X_train, y_train, X_test)
		y_predictions = Grad_out[0]
		resub_predictions = Grad_out[1]



elif (model_choice == 'Stack'):
	# 160025769 wrote the Stacking
	# 130007105 helped with the implementation of Stacking 

	model_choice_all = raw_input("Enter models - space delimited: ")  # enter all models to be used for stacking - model codes should be delimited by spaces
	chosen_models = model_choice_all.split()  # split the string by spaces
	#  SET RELEVANT PARAMETERS FOR CHOSEN MODELS
	if any("DTree" in s for s in chosen_models):
		DTreebestMaxDepth = int(raw_input("DTree: MaxDepth Param eg. [3,4,5,6]: "))
		DTreebestMaxFeat = int(raw_input("DTree: MaxFeat Param eg [3,4,5,6]: "))
		DT_InfoGain_criterion = raw_input("DTree Info gain: (entropy/gini): ")
	if any("RForest" in s for s in chosen_models):
		RForbestNDepth=  int(raw_input("RForest: n_estimators EG. 80, 90, 100: "))
		RForbestMaxFeat = int(raw_input("RForest: max_features eg. 3,4,5: "))
		RF_InfoGain_criterion = raw_input("RForest info gain: (entropy/gini): ")
	if any("KNearest" in s for s in chosen_models):
		optimized_K_param = int(raw_input("K-Nearest: Enter K parameter: "))
	if any("AdaBoost" in s for s in chosen_models):
		n_estimators_best = int(raw_input("AdaBoost: n estimators Param eg. [100]: "))
	if any("GradientBoosting" in s for s in chosen_models):
		GB_n_estimators_best = int(raw_input("Gboost: n estimators Param eg. [100]: "))
		GB_max_depth_best = int(raw_input("Gboost: max depth: eg. [3]" ))
	if any("SVM" in s for s in chosen_models):
		SVM_kernel = raw_input("Kernel to use (linear/poly/rbf/sigmoid): ") 
		if SVM_kernel == 'poly':
			SVM_degree = int(raw_input("degree Polynomial (3): ")) #Degree: = polynomial degree
			optimal_C =  float(raw_input("C parameter eg. [0.1, 1, 10, 100]: "))
			optimal_gamma = float(raw_input("gamma parameter eg. [1, 0.1, 0.01, 0.001]: "))
	if any("ANN" in s for s in chosen_models):
		import keras 
		from keras.models import Sequential
		from keras.layers import Dense
		batch_param = int(raw_input("ANN: Enter No.samples per gradient update.(10): "))
		nb_epoch_param = int(raw_input("ANN: Enter No.epochs to train model(50): "))


#			model = GradientBoostingClassifier(n_estimators=n_estimators_best, learning_rate=1.0, max_depth=max_depth_best, random_state=0) 
#			model.fit(KX_train, Ky_train)
#			Ky_predictions = model.predict(KX_test)

	def DoStacking(model_choice_In, X_trainIn, y_trainIn, X_testIn):


#print(X_train.head(n=5))
#print(y_train.head(n=5))
		mydataStack = pd.concat([y_trainIn, X_trainIn], axis=1)
#print(result.head(n=5))

		data2 = mydataStack.sample(frac=1) # shuffle data
		all_models = model_choice_all.split()  # split the string by spaces
		Sfolds = np.array_split(mydataStack, len(all_models)) # split data into number of folds equivalent to number of models
		Snum_k = int(len(Sfolds)) 
		collect_predictions = []	#intialize empty variable to collect predictions
		collect_resub_pred = []


		#### Implement S
		###  Loop through different folds subsetted from original dataset
		###  For each iteration collect predictions for stacking
		###  Train a different model for each fold
		###  Average the predictions of all models
		for i in range(0,Snum_k):
			Stest = Sfolds[i]      
			Strain = []         # Initialize list to training each model with
			for j in xrange(len(Sfolds)):
				if j != i:
					Strain.append(Sfolds[j])  
			Strain = pd.concat(Strain) # concatenate the Ktrain list of pandas dataframes into single dataframe 
		# Definte train set v test set + labels v features
			SX_train = Strain.drop('Cancer', 1)                      
			Sy_train = Strain['Cancer'] 

			Smodel_choice = all_models[i]
			print(Smodel_choice)
			if (Smodel_choice == 'LogReg'):

				stack_model = LogisticRegression(random_state = 0)
				stack_model.fit(SX_train, Sy_train)
				stack_predictions = stack_model.predict(X_testIn)
				stack_resub_predictions = stack_model.predict(X_trainIn)

#				LogOutPut = DoLogReg(SX_train, Sy_train, X_testIn)
#				stack_predictions = LogOutPut[0]
#				stack_resub_predictions = LogOutPut[1]


			if (Smodel_choice == 'NaiveBayes'):
				stack_model = GaussianNB()
				stack_model.fit(SX_train, Sy_train)
				stack_predictions = stack_model.predict(X_testIn)
				stack_resub_predictions = stack_model.predict(X_trainIn)
			if (Smodel_choice == 'DTree'):
				stack_model = DecisionTreeClassifier(criterion = DT_InfoGain_criterion, max_depth = DTreebestMaxDepth, max_features= DTreebestMaxFeat)
				#stack_model = DecisionTreeClassifier(max_depth = 5, max_features= 3)
				stack_model.fit(SX_train, Sy_train)
				stack_predictions = stack_model.predict(X_testIn)
				stack_resub_predictions = stack_model.predict(X_trainIn)
			if (Smodel_choice == 'RForest'):
				stack_model = RandomForestClassifier(n_estimators = RForbestNDepth, max_features= RForbestMaxFeat)
				stack_model.fit(SX_train, Sy_train)
				stack_predictions = stack_model.predict(X_testIn)
				stack_resub_predictions = stack_model.predict(X_trainIn)
			if (Smodel_choice == 'KNearest'):
				stack_model = KNeighborsClassifier(n_neighbors = optimized_K_param)  # Fit KNearest with optimized K parameter
				stack_model.fit(SX_train, Sy_train)
				stack_predictions = stack_model.predict(X_testIn)
				stack_resub_predictions = stack_model.predict(X_trainIn)
			if (Smodel_choice == 'AdaBoost'):
				stack_model = AdaBoostClassifier(n_estimators= n_estimators_best)
				stack_model.fit(SX_train, Sy_train)
				stack_predictions = stack_model.predict(X_testIn)
				stack_resub_predictions = stack_model.predict(X_trainIn)
			if (Smodel_choice == 'GradientBoosting'):
				stack_model = GradientBoostingClassifier(n_estimators=GB_n_estimators_best, learning_rate=1.0, max_depth=GB_max_depth_best, random_state=0) 
				stack_model.fit(SX_train, Sy_train)
				stack_predictions = stack_model.predict(X_testIn)
				stack_resub_predictions = stack_model.predict(X_trainIn)
			if (Smodel_choice == 'SVM'):
				if SVM_kernel == 'poly':
					stack_model = SVC(C = optimal_C, kernel = SVM_kernel, degree=SVM_degree, gamma=optimal_gamma, random_state = 0)
				else:
					stack_model = SVC(C = optimal_C, kernel = SVM_kernel, gamma=optimal_gamma, random_state = 0)
				stack_model.fit(SX_train, Sy_train)
				stack_predictions = stack_model.predict(X_testIn)	
				stack_resub_predictions = stack_model.predict(X_trainIn)
	#		if (Smodel_choice == 'ANN'):
	#			sc = StandardScaler()
	#			SsX_train = sc.fit_transform(X_train)
	#			SsX_shape = SsX_train.shape  # IMPORTANT FOR INPUT LAYER PARAMETER!!!!
	#			num_features = SsX_shape[1]
	#			#initialize ANN by defining it as a sequence of layers
	#			stack_model = Sequential()
	#			stack_model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = num_features))
	#			# Add one hidden layer
	#			stack_model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
	#			# Add output layer
	#			stack_model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
	#			# Apply stochastic gradient descent
	#			stack_model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
	#			# Choose number of epochs 
	#			stack_model.fit(SsX_train, y_train, batch_size=batch_param, nb_epoch = nb_epoch_param)
	#			stack_predictions = stack_model.predict(X_test)
	#			stack_predictions = (stack_predictions > 0.5) 
			collect_predictions.append(stack_predictions)
			collect_resub_pred.append(stack_resub_predictions) 

		### generalisation predictions
		output_predictions = AverageOutResults(collect_predictions)  
		### REsubtitution predictions
		resub_output_predictions = AverageOutResults(collect_resub_pred)
		return(output_predictions, resub_output_predictions)

	stack_predictions = DoStacking(model_choice_all, X_train, y_train, X_test)
	y_predictions = stack_predictions[0]
	resub_predictions = stack_predictions[1]


# This is used in Bagging function
def ModelSelector(model_choiceIn, XtrainIn, ytrainIn, XtestIn):
	if (model_choiceIn == 'LogReg'):
		LogOutPut = DoLogReg(XtrainIn, ytrainIn, XtestIn)
		mod_predictions = LogOutPut[0]
	if (model_choiceIn == 'SVM'):
		SVM_OutPut = DoSVM(XtrainIn, ytrainIn, XtestIn)
		mod_predictions = SVM_OutPut[0]
	if (model_choiceIn == 'DTree'):
		D_OutPut = DoDTree(XtrainIn, ytrainIn, XtestIn)
		mod_predictions = D_OutPut[0]
	if (model_choiceIn == 'RForest'):
		RF_OutPut = DoRForest(XtrainIn, ytrainIn, XtestIn)
		mod_predictions = RF_OutPut[0]
	if (model_choiceIn == 'KNearest'):	
		KN_OutPut = Do_KNearest(XtrainIn, ytrainIn, XtestIn)
		mod_predictions = KN_OutPut[0]
	if (model_choiceIn == 'NaiveBayes'):
		NB_output = DoNaiveBayes(XtrainIn, ytrainIn, XtestIn)
		mod_predictions = NB_output[0]
	if (model_choiceIn == 'Stack'):
		st_output = DoStacking(model_choice_all, XtrainIn, ytrainIn, XtestIn)
		mod_predictions = st_output[0]
	if (model_choiceIn == 'AdaBoost'):
		Ada_out = DO_AdaBoost(XtrainIn, ytrainIn, XtestIn)
		mod_predictions = Ada_out[0]
	if (model_choiceIn == 'GradientBoosting'):
		G_out = DO_GradBoost(XtrainIn, ytrainIn, XtestIn)
		mod_predictions = G_out[0]

	return(mod_predictions)



Baggits = raw_input("Shall we apply chosen model to Bagging procedure? (Y/N) ") 
if (Baggits == "Y"):

	# this function is called in the Do_Bagging function - to return predictions
	def ApplyModel(SampIn, ToPredictUpon):
		samp_X = SampIn.drop('Cancer', 1)           #must get rid of GOOD column
		samp_Y = SampIn['Cancer']                   #response column only GOOD
		samp_X_train, samp_X_test, samp_y_train, samp_y_test = train_test_split(samp_X, samp_Y, test_size= 0.33, random_state = 101)
		samp_predictions = ModelSelector(model_choice,samp_X_train, samp_y_train, ToPredictUpon)
		return(samp_predictions)
		
##  In K fold cross validation - shall we sample from original dataset - or k folds made to look like original dataset?
#	def BaggIt(X_trainIn, y_trainIn, X_testIn, y_testIn):
		# turn all inputs into single dataframe again to sample from. 

		#Booty2 = pd.concat([y_testIn, X_testIn], axis=1)
		#Bootsy = [Booty1, Booty2]
		#myBootyData = pd.concat(Bootsy)
	sam_size = int(raw_input("enter total num rows to collect: ")) 
	boot_size = int(raw_input("enter total num bootstraps: ")) 


	def Do_Bagging(X_trainIn, y_trainIn, X_testIn):
		Booty1 = pd.concat([y_trainIn, X_trainIn], axis=1)
		samples = bootstrap_it(Booty1, sam_size, boot_size) 
		array_ans = map(lambda x: ApplyModel(x, X_testIn), samples)	 # create list of predictions from bagged samples - bagging
		ans = AverageOutResults(array_ans)
		return(ans)

	y_predictions = Do_Bagging(X_train, y_train, X_test)
	resub_predictions = Do_Bagging(X_train, y_train, X_train)






# Classification Report
from sklearn.metrics import confusion_matrix, classification_report
report_CM = raw_input("Return CM/classification report? (Y/N)")

###       Predicted 
#			-   +	
# Truth  -  d   c
#        +  b   a 

# accuracy = a + b / all
# misclassification rate = b + c / all
# true positive (sensitivity) = d / c + d 
# false positive = b / a + b
# true negative (specificity) =  a / a + b
# false negative = c / c + d
# precision = d / b + d 

def DoConfMat(yTestIn, yPredIn):
	Confusion_M1 = confusion_matrix(yTestIn, yPredIn) #.ravel()
	tn, fp, fn, tp = confusion_matrix(yTestIn, yPredIn).ravel()
	ALL1 = tn + fp + fn + tp 
	stat_accuracy1 = (tp + tn) / ALL1
	stat_misclassification_rate1 = (fp + fn) / ALL1
	stat_false_negative1 = fn / (tp + fn) # false negative rate
	stat_false_positive1 = fp / (fp + tn) # false positive rate
	stat_sensitivity1 = tp / (tp + fn) # false negative rate
	stat_specificity1 = tn / (tn + fp)
	stat_precision1  = tp/(tp+fp)
	return((Confusion_M1, stat_accuracy1, stat_misclassification_rate1, stat_sensitivity1, stat_false_positive1, stat_specificity1, stat_false_negative1, stat_precision1))

'''
	Confusion_M1 = confusion_matrix(yTestIn, yPredIn)

	#### THIS IS PROBABLY VERY WRONG
	if len(Confusion_M1) == 1:
		D1 = float(Confusion_M1[0][0])
		stat_accuracy1 = 1  #???
		stat_misclassification_rate1 = 0 
		stat_sensitivity1 = 1     #???
		stat_false_positive1 = 0
		stat_specificity1 = 1
		stat_false_negative1 = 0
		stat_precision1 = 1 ## ??? 
		return((Confusion_M1, stat_accuracy1, stat_misclassification_rate1, stat_sensitivity1, stat_false_positive1, stat_specificity1, stat_false_negative1, stat_precision1))
	#### ^ THIS IS PROBABLY WRONG
		
	if len(Confusion_M1) == 2:
		A1 = float(Confusion_M1[1][1]) # true positive / yes / 1
		B1 = float(Confusion_M1[1][0]) # false negative
		C1 = float(Confusion_M1[0][1]) # false positive
		D1 = float(Confusion_M1[0][0])  # true negative / no / 0 
		
		ALL1 = A1 + B1 + C1 + D1 
		stat_accuracy1 = (A1+D1)/ALL1  
		stat_misclassification_rate1 = (B1 + C1)/ALL1
	# either we do do not calculation or return by zero
		if((A1+B1)>0):
			if (A1 == 0):
				stat_sensitivity1 = 0 
				stat_false_negative1 = 1
			elif (B1 == 0):
				stat_sensitivity1 = 1 
				stat_false_negative1 = 0
			else: 
				stat_sensitivity1 = A1/(A1+B1)    # correct - predicted yes / all actual yes's
				stat_false_negative1 = B1/(A1+B1)  # correct
		else:
			stat_sensitivity1 = 0 
			stat_false_negative1 = 0


		if((C1+D1)>0):
			if (C1 == 0):
				stat_specificity1 = 0
				stat_false_positive1 = 1 
			elif (D1 == 0): 
				stat_specificity1 = 1
				stat_false_positive1 = 0
			else:
				stat_specificity1 = D1/(C1+D1)     
				stat_false_positive1= C1/(C1+D1)
		else:
			stat_specificity1 = 0
			stat_false_positive1 = 0

	### ^ These are all correct

		if((A1+C1)>0):
			stat_precision1  = A1/(A1+C1)
		else:
			stat_precision1 = 0

'''


if (report_CM == 'Y'):
	classification_stats = DoConfMat(y_test, y_predictions)

	conf_mat = classification_stats[0]
	conf_mat = pd.DataFrame(data=conf_mat, index=['Actual No', 'Actual Yes'], columns=['Predicted No', 'Predicted Yes']) 
	print(conf_mat)
	stat_accuracy = classification_stats[1] 
	stat_misclassification_rate = classification_stats[2]
	stat_sensitivity = classification_stats[3]
	stat_false_positive = classification_stats[4]
	stat_specificity = classification_stats[5]
	stat_false_negative = classification_stats[6]
	stat_precision  = classification_stats[7]

	print("\nAccuracy: ", stat_accuracy)
	print("misclassification rate: ", stat_misclassification_rate)
	print("Sensitivity (True Positives): ", stat_sensitivity)
	print("False Positive:", stat_false_positive)
	print("Specificity (True Negatives): ", stat_specificity)
	print("** False Negatives ** : ", stat_false_negative)
	print("Precision: :", stat_precision)
	print("\nClassification Report\n")


	### Classification report
	print("Predictions vs Training set (Resubstitution error)") 
	print(classification_report(y_train, resub_predictions))

	print("Predictions vs Test set") 
	print(classification_report(y_test, y_predictions))


	print("\nConfusion Matrix\n")
	Confusion_M = confusion_matrix(y_test, y_predictions)
	print(Confusion_M)
	A = float(Confusion_M[0][0])
	B = float(Confusion_M[0][1])
	C = float(Confusion_M[1][0])
	D = float(Confusion_M[1][1])
	ALL = A + B + C + D 
	stat_accuracy = (A+B)/ALL  
	stat_misclassification_rate = (B + C)/ALL 
	stat_sensitivity = D/(C+D)
	stat_false_positive = B/(A+B)
	stat_specificity = A/(A+B)
	stat_false_negative = C/(C+D)
	stat_precision  = D/(B+D)
	'''

'''
	print("Accuracy: ", stat_accuracy)
	print("misclassification rate: ", stat_misclassification_rate)
	print("Sensitivity (True Positives): ", stat_sensitivity)
	print("False Positive:", stat_false_positive)
	print("Specificity (True Negatives): ", stat_specificity)
	print("** False Negatives ** : ", stat_false_negative)
	print("Precision: :", stat_precision)
	print("\nClassification Report\n")
	print(classification_report(y_test, y_predictions))
'''
'''
#ROC curve
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from scipy import interp
import random

ROC_input = raw_input("Return ROC curve? (Y/N): ")

def ROC_visual():
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc='lower right')
	plt.xlim([-0.1,1.2])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')	
	plt.show()

def cross_validate_roc(mydataIn, num_k_param, model_choice_param):
	mydataIn = mydataIn.sample(frac=1) # randomly shuffle data, frac parameter define what fraction of data to return - frac= 1  means we return all data shuffled
	folds = np.array_split(mydataIn, num_k_param)
	num_k = int(len(folds)) 
	# Initialize MEAN ROC CURVE
	mean_true_positive_rate = 0
	mean_false_positive_rate = np.linspace(0, 1, 100)
	all_true_positive_rate = []

	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random Luck')

	CV_accuracy = []
	CV_misclassification = []
	CV_sensitivity = []
	CV_false_positive = []
	CV_specificity = []
	CV_false_negative = []
	CV_precision = []

	# K fold cross validation
	for i in range(0,num_k):
		Ktest = folds[i]
		Ktrain = []
		# returns a list of pandas dataframes from all k folds - apart from k fold used in Ktest
		for j in xrange(len(folds)):
			if j != i:
				Ktrain.append(folds[j])  
		Ktrain = pd.concat(Ktrain) # concatenate the Ktrain list of pandas dataframes into single dataframe 



		##### SHALL WE BOOTSTRAP AND BOOS THAT DAMN KTRAIN??? - and shall we bootstrap it from the K set - or the original dataset?!?!?1


		# Definte train set v test set + labels v features
		KX_train = Ktrain.drop('Cancer', 1)         
		KX_test = Ktest.drop('Cancer',1)              
		Ky_train = Ktrain['Cancer'] 
		Ky_test = Ktest['Cancer'] 

		if (Baggits != "Y"):
			if (model_choice == 'LogReg'):
				LogOutPut = DoLogReg(KX_train, Ky_train, KX_test)
				Ky_predictions = LogOutPut[0]
			if (model_choice == 'NaiveBayes'):
				NB_output = DoNaiveBayes(KX_train, Ky_train, KX_test)
				Ky_predictions = NB_output[0]
			if (model_choice == 'KNearest'):	
			#FIT K Nearest Neightbour model with best K value
				KN_OutPut = Do_KNearest(KX_train, Ky_train, KX_test)
				Ky_predictions = KN_OutPut[0]
			if (model_choice == 'DTree'):
				D_OutPut = DoDTree(KX_train, Ky_train, KX_test)
				Ky_predictions = D_OutPut[0]
			if (model_choice == 'RForest'):
				RF_OutPut = DoRForest(KX_train, Ky_train, KX_test)
				Ky_predictions = RF_OutPut[0]
	#		if (model_choice == 'SVM'):
	#			print("Cross validating a LINEAR kernel SVM")
	#			model = SVC(kernel = 'linear', random_state = 0)
	#			model.fit(KX_train, Ky_train)
	#			Ky_predictions = model.predict(KX_test)
	#			print("Was unable to get CV-SVM working... debugging required")
			if (model_choice == 'SVM'):
				if SVM_kernel == 'poly':
					model = SVC(C = optimal_C, kernel = SVM_kernel, degree=SVM_degree, gamma=optimal_gamma, random_state = 0)
				else:
					model = SVC(C = optimal_C, kernel = SVM_kernel, gamma=optimal_gamma, random_state = 0)
				model.fit(KX_train, Ky_train)
				Ky_predictions = model.predict(KX_test)
			if (model_choice == 'ANN'):
				#KX_shape = KX_train.shape  # IMPORTANT FOR INPUT LAYER PARAMETER!!!!
				#num_features = KX_shape[1]
				#model = Sequential()
				#model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = num_features))
				#model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
				#model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
				#model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
				#model.fit(KX_train, Ky_train, batch_size=batch_param, nb_epoch = nb_epoch_param)
				#Ky_predictions = model.predict(X_test)
				#Ky_predictions = (Ky_predictions > 0.5) 
				print("Was unable to get CV-ANN working... debugging required")
			if (model_choice == 'AdaBoost'):
				model = AdaBoostClassifier(n_estimators= n_estimators_best)
				model.fit(KX_train, Ky_train)
				Ky_predictions = model.predict(KX_test)
			if (model_choice == 'GradientBoosting'):
				model = GradientBoostingClassifier(n_estimators=n_estimators_best, learning_rate=1.0, max_depth=max_depth_best, random_state=0) 
				model.fit(KX_train, Ky_train)
				Ky_predictions = model.predict(KX_test)
			if (model_choice == 'Stack'):
				stack_predictions = DoStacking(model_choice_all, KX_train, Ky_train, KX_test)
				Ky_predictions = stack_predictions[0]
				### cHECK STACKING, Because, the model is not trained via model.fit(KX_train, Ky_train) .. but rather stack_model.fit(SX_train, Sy_train)
				#### NEED TO EDIT DoStacking function later on in order to permit this function
		if (Baggits == "Y"):
#			y_predictions = BaggIt(X_train, y_train, X_test, y_test)
			Ky_predictions = Do_Bagging(KX_train, Ky_train, KX_test)  # K-fold * BAG FROM THE K FOLD. 

#

		false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(Ky_test, Ky_predictions)  # added thresholds without TEST!!!!! - will it mess up program?!?!?!?
		roc_auc = metrics.auc(false_positive_rate,true_positive_rate)

		Kclassification_stats = DoConfMat(Ky_test, Ky_predictions)
		CV_accuracy.append(Kclassification_stats[1])
		CV_misclassification.append(Kclassification_stats[2])
		CV_sensitivity.append(Kclassification_stats[3])
		CV_false_positive.append(Kclassification_stats[4])
		CV_specificity.append(Kclassification_stats[5])
		CV_false_negative.append(Kclassification_stats[6])
		CV_precision.append(Kclassification_stats[7])



		#mean ROC
		mean_true_positive_rate += interp(mean_false_positive_rate, false_positive_rate, true_positive_rate)
		mean_true_positive_rate[0] = 0
		plt.plot(false_positive_rate, true_positive_rate, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

	#calculate mean AUC
	mean_true_positive_rate /= num_k
	mean_true_positive_rate[-1] = 1
	mean_auc = auc(mean_false_positive_rate, mean_true_positive_rate)

	#
	if(len(CV_accuracy) > 0):
		stat_accuracy2 = sum(CV_accuracy)/len(CV_accuracy)
	else: 
		stat_accuracy2 = 0 

	if(len(CV_misclassification) > 0):
		stat_misclassification_rate2 = sum(CV_misclassification)/len(CV_misclassification) 
	else: 
	 	stat_misclassification_rate2 = 0

	if(len(CV_sensitivity) > 0):	
		stat_sensitivity2 = sum(CV_sensitivity)/len(CV_sensitivity)	
	else: 
		stat_sensitivity2 = 0

	if(len(CV_false_positive) > 0):
		stat_false_positive2 = sum(CV_false_positive)/len(CV_false_positive) 
	else:
		stat_false_positive2 = 0	
	
	if(len(CV_specificity) > 0):
		stat_specificity2 = sum(CV_specificity)/len(CV_specificity) 
	else: 
		stat_specificity2 = 0 

	if(len(CV_false_negative) > 0):
		stat_false_negative2 = sum(CV_false_negative)/len(CV_false_negative) 
	else: 
		stat_false_negative2 = 0

	if(len(CV_precision) > 0):
		stat_precision2 = sum(CV_precision)/len(CV_precision)
	else:
		stat_precision2 = 0

	print("K fold mean Accuracy: ", stat_accuracy2)
	print("K fold mean Misclassification rate: ", stat_misclassification_rate2)
	print("K fold Sensitivity (True Positives): ", stat_sensitivity2)
	print("K fold False Positive:", stat_false_positive2)
	print("K fold Specificity (True Negatives): ", stat_specificity2)
	print("K fold ** False Negatives ** : ", stat_false_negative2)
	print("K fold Precision: :", stat_precision2)

	# Visualisation
	
	plt.plot(mean_false_positive_rate, mean_true_positive_rate, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
	plt.plot([0,1],[0,1],'r--')
	ROC_visual()

	return(mean_auc, stat_accuracy2, stat_misclassification_rate2, stat_sensitivity2, stat_false_positive2, stat_specificity2, stat_false_negative2, stat_precision2)



if (ROC_input == 'Y') :
	false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_predictions)  
	roc_auc = metrics.auc(false_positive_rate,true_positive_rate)

	# visualisation
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random Luck')
	plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
	ROC_visual()

import matplotlib.pyplot as plt

### APPLY LEAVE ONE OUT CROSS VALIDATION 
###       Predicted 
#			-   +	
# Truth  -  d   c
#        +  b   a 


# accuracy = a + b / all
# misclassification rate = b + c / all
# true positive (sensitivity) = d / c + d 
# false positive = b / a + b
# true negative (specificity) =  a / a + b
# false negative = c / c + d
# precision = d / b + d 


LOOCV_question = raw_input("Apply LOOCV? (Y/N): ")	
if (LOOCV_question == "Y"):
	def LOOCV(mydataIn, model_choice_param):
		indexes = mydataIn.index.values
		#print(type(indexes))

		# Initialize empty array to collect confusion matrix values
		A_count = []
		B_count = []
		C_count = []
		D_count = []

		for i in range(0,len(indexes)):
			all_other_indexes = np.delete(indexes, i) # ALL INDEXES to be left in 
			leave_these_IN = mydataIn.loc[all_other_indexes, :] #SLICE ROWS BY GIVEN INDEX
			leave_this_OUT = mydataIn.loc[[indexes[i]]] # SLICE ROW to be left OUT 
			Ktest = leave_this_OUT
			Ktrain = leave_these_IN



			#### IMPORTANT!!!!!






			### Ktrain - shall we bootstrap Ktrain? 

			Ktrain = bootstrap_it(Ktrain, sam_size, 1)
			Ktrain = Ktrain[0] 

			### ????  ^^^






			KX_train = Ktrain.drop('Cancer', 1)         
			KX_test = Ktest.drop('Cancer',1)              
			Ky_train = Ktrain['Cancer'] 
			Ky_test = Ktest['Cancer'] 
			if (Baggits != "Y"):
				if (model_choice == 'LogReg'):
					LogOutPut = DoLogReg(KX_train, Ky_train, KX_test)
					Ky_predictions = LogOutPut[0]
				if (model_choice == 'NaiveBayes'):
					model = GaussianNB()
					model.fit(KX_train, Ky_train)
					Ky_predictions = model.predict(KX_test)
				if (model_choice == 'KNearest'):	
					KN_OutPut = Do_KNearest(KX_train, Ky_train, KX_test)
					Ky_predictions = KN_OutPut[0]
				if (model_choice == 'DTree'):
					D_OutPut = DoDTree(KX_train, Ky_train, KX_test)
					Ky_predictions = D_OutPut[0]
				if (model_choice == 'RForest'):
					RF_OutPut = DoRForest(KX_train, Ky_train, KX_test)
					Ky_predictions = RF_OutPut[0]

#			if (model_choice == 'SVM'):
#			print("Cross validating a LINEAR kernel SVM")
#				model = SVC(kernel = 'linear', random_state = 0)
#				model.fit(KX_train, Ky_train)
#				Ky_predictions = model.predict(KX_test)
#				print("one iteration")
				if (model_choice == 'SVM'):
					if SVM_kernel == 'poly':
						model = SVC(C = optimal_C, kernel = SVM_kernel, degree=SVM_degree, gamma=optimal_gamma, random_state = 0)
					else:
						model = SVC(C = optimal_C, kernel = SVM_kernel, gamma=optimal_gamma, random_state = 0)
					model.fit(KX_train, Ky_train)
					Ky_predictions = model.predict(KX_test)
				if (model_choice == 'AdaBoost'):
					model = AdaBoostClassifier(n_estimators= n_estimators_best)
					model.fit(KX_train, Ky_train)
					Ky_predictions = model.predict(KX_test)
				if (model_choice == 'GradientBoosting'):
					model = GradientBoostingClassifier(n_estimators=n_estimators_best, learning_rate=1.0, max_depth=max_depth_best, random_state=0) 
					model.fit(KX_train, Ky_train)
					Ky_predictions = model.predict(KX_test)
				if (model_choice == 'Stack'):
					stack_predictions = DoStacking(model_choice_all, KX_train, Ky_train, KX_test)
					Ky_predictions = stack_predictions[0]
			### cHECK STACKING, Because, the model is not trained via model.fit(KX_train, Ky_train) .. but rather stack_model.fit(SX_train, Sy_train)
			#### NEED TO EDIT DoStacking function later on in order to permit this function
			if (Baggits == "Y"):
				Ky_predictions = Do_Bagging(X, Y, KX_test)


			K_confusionMatrix = confusion_matrix(Ky_test, Ky_predictions)

			if len(K_confusionMatrix) == 1: 
				if (K_confusionMatrix[0]) == 1:
					D_count.append(K_confusionMatrix[0]) 
			if len(K_confusionMatrix) == 2:
				if (K_confusionMatrix[0][0]) == 1:
					D_count.append(K_confusionMatrix[0][0]) 
				if(K_confusionMatrix[0][1]) == 1:
					C_count.append(K_confusionMatrix[0][1])
				if(K_confusionMatrix[1][0]) == 1:
					B_count.append(K_confusionMatrix[1][0])
				if(K_confusionMatrix[1][1]) == 1:
					A_count.append(K_confusionMatrix[1][1])

			# predict [0] but confusion matrix returns:  [[1]] ... this means it is D... 

		A1 = float(len(A_count))
		B1 = float(len(B_count))
		C1 = float(len(C_count))
		D1 = float(len(D_count))
		ALL1 = A1 + B1 + C1 + D1 

		print("True Postive Count:", A1)
		print("False Negative Count:", B1)
		print("False Positive Count:", C1)
		print("True Negative Count:", D1)
		print("Total number of Predictions:", ALL1)

		# NEW
		C1D1 = C1 + D1 
		A1B1 = A1 + B1 
		A1C1 = A1 + C1 

		print("REAL number of Negatives:", C1D1)
		print("REAL number of Positives:", A1B1)
		print("Predicted number of negatives: ", A1C1)

		## Original
		stat_accuracy3 = (A1+D1)/ALL1  
		stat_misclassification_rate3 = (B1 + C1)/ALL1

		if (A1B1 > 0.0): 
			if (A1 == 0.0):
				stat_sensitivity3 = 0.0 
				stat_false_negative3 = 1.0 
			elif (B1 == 0.0):
				stat_sensitivity3 = 1.0 
				stat_false_negative3 = 0.0  
			else: 
				stat_sensitivity3 = A1/(A1+B1)  
				stat_false_negative3 = B1/(A1+B1)
		else:
			stat_sensitivity3 = 0.0 
			stat_false_negative3 = 0.0 




#		else: 	
#			stat_sensitivity1 = 1.0 
#			stat_false_negative1 = 0.0  

		if (C1D1 > 0.0):
			if (C1 == 0.0):
				stat_specificity3 = 1.0
				stat_false_positive3 = 0.0
			elif (D1 == 0.0):
				stat_specificity3 = 0.0
				stat_false_positive3 = 1.0
			else: 
				stat_false_positive3 = C1/(C1+D1)
				stat_specificity3 = D1/(C1+D1)
		else:
			stat_specificity3 = 0.0
			stat_false_positive3 = 0.0




		if (A1C1 > 0.0):
			if (A1 == 1.0):
				stat_precision3 = 1.0
			elif (A1 == 0.0):
				stat_precision3 = 0.0
			else:
				stat_precision3  = A1/(A1+C1)
		else: 
			stat_precision3 = 0.0

		return(stat_accuracy3, stat_misclassification_rate3, stat_sensitivity3
			, stat_false_positive3, stat_specificity3, stat_false_negative3, stat_precision3)

	if (BootOrNot == "Y"):
		LOOCV_stats = LOOCV(mydataIn = mydata_untouched, model_choice_param= model_choice)
	else:
		LOOCV_stats = LOOCV(mydataIn = mydata, model_choice_param= model_choice)

	#Collect LOOCV stats and put into pandas dataframe
	Stat_value = []
	for i in range(0,7):
		Stat_value.append(LOOCV_stats[i])
	Stat = ["acc", "misclas", "sensitivity", "fpr", "specificity", "fnr", "precision"]
	collected_LOOCV_statistics = pd.DataFrame(
    		{'Stat': Stat,
     		'Value': Stat_value
    		})
	print(collected_LOOCV_statistics)

	#PLOT
	DF = collected_LOOCV_statistics.set_index('Stat')
	ax = DF.plot(kind='bar', title ="LOOCV statistics", figsize=(8, 8), fontsize=10)
	ax.set_xlabel("Stat", fontsize=10)
	ax.set_ylabel("Value", fontsize=10)
	plt.show()
	print_or_not = raw_input("Print results? (Y/N): ")
	if (print_or_not == "Y"):
		outputName = raw_input("FILE NAME: ")
		collected_LOOCV_statistics.to_csv(outputName, sep='\t')



CV_question = raw_input("Cross Validate the test/train? (Y/N): ")
if (CV_question == "Y"):
	iterate_k_question = raw_input("Iterate across various 3-4-5,6 fold cross validation? (Y/N): ")
	if(iterate_k_question == "N"):
		fold_question = int(raw_input("How many K folds? (5) : ")) 
		cross_validate_roc(mydataIn=mydata, num_k_param=fold_question, model_choice_param=model_choice)    ### FUNCTION DEFINED ABOVE
	else:
		collect_AUC = []
		collect_accuracy = []
		collect_misclass = [] 
		collect_sensitivity = [] 
		collect_false_positive = []
		collect_specificity = []
		collect_false_negative = []
		collect_precision = []
		k_list = [2,3,4,5,6]
		for i in range(len(k_list)):
			print("\n", "K value: ", k_list[i])
			cross_validate_roc(mydataIn=mydata, num_k_param=k_list[i], model_choice_param=model_choice)
			output_evaluation_stats = cross_validate_roc(mydataIn=mydata, num_k_param=k_list[i], model_choice_param=model_choice) 
			out_AUC = output_evaluation_stats[0]
			out_accuracy = output_evaluation_stats[1]
			out_misclass = output_evaluation_stats[2]
			out_sensitivity = output_evaluation_stats[3]
			out_false_positive = output_evaluation_stats[4]
			out_specificity = output_evaluation_stats[5]
			out_false_negative = output_evaluation_stats[6]
			out_precision = output_evaluation_stats[7]
			collect_AUC.append(out_AUC)
			collect_accuracy.append(out_accuracy)
			collect_misclass.append(out_misclass)
			collect_sensitivity.append(out_sensitivity) 
			collect_false_positive.append(out_false_positive)
			collect_specificity.append(out_specificity)
			collect_false_negative.append(out_false_negative)
			collect_precision.append(out_precision)		


		collected_evaluation_statistics = pd.DataFrame(
    		{'AUC': collect_AUC,
     		'Accuracy': collect_accuracy,
     		'Misclassification': collect_misclass,
     		'Sensitivity': collect_sensitivity,
     		'FalsePositive': collect_false_positive,
     		'Specificity': collect_specificity,
     		'FalseNegative': collect_false_negative,
     		'Precision': collect_precision,
     		'K fold': k_list
    		})
		print(collected_evaluation_statistics)
		print_or_not2 = raw_input("Print results? (Y/N): ")
		if (print_or_not2 == "Y"):
			outputName = raw_input("FILE NAME: ")
			collected_evaluation_statistics.to_csv(outputName, sep='\t')
#		plt.scatter(collected_evaluation_statistics['K fold'], collected_evaluation_statistics['AUC'], s=10)
		plt.ylim(0.0,1.0)
		plt.xlabel('K fold')
		plt.ylabel('statistic')
		plt.title('Model Evaluation')
		plt.plot(collected_evaluation_statistics['K fold'],collected_evaluation_statistics['AUC'])
		plt.plot(collected_evaluation_statistics['K fold'],collected_evaluation_statistics['Accuracy'])
		plt.plot(collected_evaluation_statistics['K fold'],collected_evaluation_statistics['Misclassification'])
		plt.plot(collected_evaluation_statistics['K fold'],collected_evaluation_statistics['Precision'])
		plt.legend(loc='upper left', frameon=False)
		plt.show()
		
		plt.gcf().clear()

		plt.ylim(0.0,1.0)
		plt.xlabel('K fold')
		plt.ylabel('statistic')
		plt.title('Model Evaluation')
		plt.plot(collected_evaluation_statistics['K fold'],collected_evaluation_statistics['Sensitivity'])
		plt.plot(collected_evaluation_statistics['K fold'],collected_evaluation_statistics['FalsePositive'])
		plt.plot(collected_evaluation_statistics['K fold'],collected_evaluation_statistics['Specificity'])
		plt.plot(collected_evaluation_statistics['K fold'],collected_evaluation_statistics['FalseNegative'])
		plt.legend(loc='upper left', frameon=False)
		plt.show() 



		









# Can we find a way of ranking independent variables in terms of predictive importance? 
# PCA/Random forrest can help rank variables
# Step wise selection via subsetting out independent variables
# Shall we simulate data / introduce Nas to further test predictive power? 
## eg. simulate data with high low medium value. 


# op - measurements before or after diagnosis? 
# is test precision, better or worse as training precision?
# if we think we are acceptable numbers for test - we need cross validation - might be a fluke due to test and train
# if we get similar confusion matrices every time during cross validation - things are ok
# 

# ROC curve - each column against Cancer - it is not an approved way of identifying important variables - but it seems to work
# gives us a basis for further experimentation 
# compare ROC curve for each of these vs ROC curve for FULL model 
# individual covariates with highest ROC curve contributes to most between cancer yes or no.  




