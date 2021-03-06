import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf
from tensorflow.keras import layers
from copy import copy

def transform_data(data_orig):
	"""
	Function that takes experimental data and gives us the
	dependent/independent variables for analysis.

	Parameters
	----------
	data : Pandas DataFrame or string.
		If this is a DataFrame, it should have the columns `contrast1` and
		`answer` from which the dependent and independent variables will be
		extracted. If this is a string, it should be the full path to a csv
		file that contains data that can be read into a DataFrame with this
		specification.

	Returns
	-------
	X : array
		train data, information about each passenger
		features: ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\
					'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
	y : array
		label of survival of each passenger
	"""
	if isinstance(data_orig, str):
		data_orig = pd.read_csv(data_orig)

	data = data_orig

	num_rows,num_variables = data.shape
	all_columns = data.columns.tolist()
	clean_data(data,all_columns,ignore_na=False,fill_mode="prob")
	expand_features(data)
	variables = ['Pclass','Sex',"Fare","Age","SibSp","Parch","Embarked","Fam_size",\
				 "cabin_no","ticket_no","friend","Fare_person","Child"]
	X = pd.get_dummies(data[variables])

	## normalise features to zero man and unit variance
	scaler = preprocessing.StandardScaler().fit(X)
	X_scaled = scaler.transform(X)
	X = pd.DataFrame(X_scaled, columns=X.columns)

	if "Survived" in data.columns:
		y = data['Survived']
	else:
		y = None

	return X, y


def expand_features(data):
	"""
	build new features from existing ones
	Fam_size : SibSp + Parch
	friend : sharing ticket or cabin without having family
	"""
	## combine num of siblings and parents to feature of family size
	data["Fam_size"] = data["SibSp"] + data["Parch"]

	## add friend category defined as either sharing a ticket with someone not family
	## or share a room with someone not registered as family
	friends = np.zeros((data['PassengerId'].size))
	for i,iid in enumerate(data["PassengerId"]):
		ticket_temp = data.loc[data["PassengerId"]==iid, "ticket_no"].values[0]
		cabin_temp = data.loc[data["PassengerId"]==iid, "cabin_no"].values[0]
		if data.loc[data["ticket_no"]==ticket_temp, "Ticket"].count()>1:
			if data.loc[data["PassengerId"]==iid,"Fam_size"].values[0]==0:
				friends[i] = 1
		elif cabin_temp!=0:## corresponds to NaN cabin value
			if data.loc[data["cabin_no"]==cabin_temp, "cabin_no"].count()>1:
				if data.loc[data["PassengerId"]==iid,"Fam_size"].values[0]==0:
					friends[i] = 1
	data["friend"] = pd.Series(friends,dtype=int)

	## fare per person
	fare_per_person = np.zeros((data['PassengerId'].size))
	for i,ifare in enumerate(data["ticket_no"].unique()):
		shared_ticket_temp = data.loc[data["ticket_no"]==ifare,"Fare"]
		fare_per_person[i] = 1.*shared_ticket_temp.values[0]/shared_ticket_temp.count()
	data["Fare_person"] = pd.Series(fare_per_person,dtype=float)

	## add child feature
	data.loc[data['Age'] <= 9, 'Child'] = 1
	data.loc[data['Age'] > 9, 'Child'] = 0


def clean_data(data,variables,ignore_na=True,fill_mode="mode"):
	"""clean data from nans and format 'Ticket'/'Cabin' values to integer"""
	num_rows,num_cols = data.shape
	if ignore_na:
		data = data.dropna(axis=0,how="any",subset=variables)
		num_rows,num_cols = data.shape
	else:
		for variable in variables:
			if data[variable].isna().sum()>0:
				if fill_mode=="mode":
					data[variable] = data[variable].fillna(data[variable].mode()[0])
				else:
					if variable=="Age":
						nan_idx = np.where(data["Age"].isna())[0]
						norm = [1.,1.]#[8.,3.]
						new_age = np.zeros((num_rows))
						for idx in nan_idx:
							err = (data[["SibSp","Pclass"]]-data[["SibSp","Pclass"]].iloc[idx])
							err /= norm
							total_err = np.sqrt(np.sum(err**2,axis=1))
							if len(data["Age"][total_err==0].mode())>0:
								new_age[idx] = data["Age"][total_err==0].mode()[0]
							else:
								new_age[idx] = data["Age"][total_err<4].mode()[0]
						data["Age"].fillna(pd.Series(new_age),inplace=True)

					if variable=="Embarked":
						# data[variable] = data[variable].fillna(data[variable].mode()[0])
						data = data.dropna(axis=0,how="any",subset=["Embarked"])
						num_rows,num_cols = data.shape

					if variable=="Fare":
						nan_idx = np.where(data["Fare"].isna())[0]
						new_fare = np.zeros((num_rows))
						for idx in nan_idx:
							temp_age = data["Age"].iloc[idx]
							temp_embarked = data["Embarked"].iloc[idx]
							temp_pclass = data["Pclass"].iloc[idx]

							similar_cases = data.loc[(data["Embarked"]==temp_embarked) &\
													 (data["Pclass"]==temp_pclass) &\
													 (data["Age"]>(temp_age-15)) &\
													 (data["Age"]<(temp_age+15)) &\
													 (data["Fare"].isna()==False), "Fare"]
							new_fare[idx] = similar_cases.median()
						data["Fare"].fillna(pd.Series(new_fare),inplace=True)


	## format ticket values
	duplicates = []
	ticket_no = np.zeros((num_rows))
	for i,uniq in enumerate(data['Ticket'].unique()):
		ticket_no[data['Ticket']==uniq] = i
	## take into account four cases where 'Ticket' is given as LINE
	ticket_no[data['Ticket']=="LINE"] = i + np.arange(np.sum(data['Ticket']=="LINE"))
	data["ticket_no"] = pd.Series(ticket_no,dtype=int)

	## format cabin values, set NaN cabin values to zero
	duplicates = []
	cabin_no = np.zeros((num_rows))
	for i,uniq in enumerate(data['Cabin'].unique()):
		if uniq=="nan":
			cabin_no[data['Cabin']==uniq] = 0
		cabin_no[data['Cabin']==uniq] = i + 1
	data["cabin_no"] = pd.Series(cabin_no,dtype=int)
	




def build_keras_model(input_shape,optimizer="adam",init="glorot_normal"):
	model = tf.keras.Sequential()
	model.add(layers.Dense(8, activation='relu', kernel_initializer=init,\
						   input_shape=(input_shape,)))#kernel_regularizer=tf.keras.regularizers.L2(0.01)
	model.add(layers.Dense(16, activation='relu', kernel_initializer=init))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(16, activation='relu', kernel_initializer=init))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation='sigmoid'))

	# compile model
	model.compile(optimizer=optimizer,
					loss='binary_crossentropy',	#'mse' 'binary_crossentropy'
					metrics=['accuracy'])		#'mae'	'categorical_accuracy'
					
	return model


class Model(object):
	"""Class for applying classification method to data"""
	def __init__(self, classific_method="LogisticRegression"):
		""" Initialize a model object.

		Parameters
		----------
		data : Pandas DataFrame
			Data from passengers from Titanic accident

		classific_method : callable, optional
			A method that fits x and y.
			Default: :classific_method:`LogisticRegression`
		"""
		self.classific_method = classific_method

	def classify(self, x, y):
		"""
		Fit a Model to data.

		Parameters
		----------
		x : float or array
			The independent variable: passenger features
		y : float or array
			The dependent variable (survived or not)

		Returns
		-------
		fit : :class:`Fit` instance
			A :class:`Fit` object that contains the parameters of the model.

		"""
		if self.classific_method=="LogisticRegression":
			clf = LogisticRegression().fit(x,y)
			score = clf.score(x,y)
			params = {"coef" : clf.coef_, "intercept" : clf.intercept_}

		elif self.classific_method=="RidgeClassifier":
			clf = RidgeClassifier().fit(x,y)
			score = clf.score(x,y)
			params = clf.get_params()

		elif self.classific_method=="MLPClassifier":
			clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),\
								random_state=1,max_iter=1000)
			clf.fit(x, y)
			params = {"coefs" : clf.coefs_}
			score = clf.score(x,y)

		elif self.classific_method=="RandomForestClassifier":
			# clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=2)
			
			# model = RandomForestClassifier(random_state=2)
			# grid_parameters = {'n_estimators': [i for i in range(300, 601, 50)],\
			# 					'min_samples_split' : [2, 10, 20, 30, 40]}
			# grid = GridSearchCV(estimator=model, param_grid=grid_parameters)
			# grid_result = grid.fit(x, y)

			# n_estimator = grid_result.best_params_['n_estimators']
			# min_samples_split = grid_result.best_params_['min_samples_split']
			

			clf = RandomForestClassifier(random_state=2,n_estimators=400,\
										 min_samples_split=30, max_depth=20)
			clf.fit(x,y)
			score = clf.score(x,y)
			params = {}#{"params" : grid_result.best_params_}

		elif self.classific_method=="NeuralNetwork":
			seed = 7
			np.random.seed(seed)
			input_shape = x.shape[1]


			clf = build_keras_model(input_shape,optimizer="adam",init="glorot_normal")

			n_epochs = 200
			n_sub_epochs = 10
			sub_epoch_size = len(x) // n_sub_epochs
			# for epoch_number in range(50):
			# 	for sub_epoch in range(n_sub_epochs):
			# 		X = x[sub_epoch * sub_epoch_size: (sub_epoch + 1) * sub_epoch_size]
			# 		Y = y[sub_epoch * sub_epoch_size: (sub_epoch + 1) * sub_epoch_size]
			# 		hist = clf.fit(X,Y,epochs=1);
			hist=clf.fit(x, y, epochs=n_epochs, batch_size=sub_epoch_size, verbose=0)
			acc = hist.history['accuracy']
			loss = hist.history['loss']
			score = acc[-1]
			params = {"acc" : acc, "loss" : loss}

		return clf, score, params




if __name__=="__main__":
	print("pass")