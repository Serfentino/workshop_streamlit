import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px


@st.cache
def loadData():
	df = pd.read_csv("avocado.csv").replace({"conventional": 0, "organic": 1})
	return df

# Basic preprocessing required for all the models.  
def preprocessing(df):
	# Assign X and y
	X = df.iloc[:, 2:-3].values
	y = df.iloc[:, -3].values

	# X and y has Categorical data hence needs Encoding
	le = LabelEncoder()
	y = le.fit_transform(y.flatten())

	# 1. Splitting X,y into Train & Test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
	return X_train, X_test, y_train, y_test, le


# Training Decission Tree for Classification
@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, y_train, y_test):
	# Train the model
	tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
	tree.fit(X_train, y_train)
	y_pred = tree.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)

	return score, report, tree

# Training Neural Network for Classification.
@st.cache(suppress_st_warning=True)
def neuralNet(X_train, X_test, y_train, y_test):
	# Scalling the data before feeding it to the Neural Network.
	scaler = StandardScaler()  
	scaler.fit(X_train)  
	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test)
	# Instantiate the Classifier and fit the model.
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	score1 = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)
	
	return score1, report, clf

# Training KNN Classifier
@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train, X_test, y_train, y_test):
	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)

	return score, report, clf


# Accepting user data for predicting its Member Type
def accept_user_data(): 
	AveragePrice = st.text_input("What is the  AveragePrice ")
	total  = st.text_input("What is the Total Volume")
	sku4046 = st.text_input("How many sku4046 sales? ") 
	user_prediction_data = np.array([AveragePrice,total,sku4046,sku4225,sku4770,smll_bags,large_bags,xlarge_bags]).reshape(1,-1)
	return user_prediction_data


# Loading the data for showing visualization of vehicals starting from various start locations on the world map.


def main():
	st.title("Classification of Avocado Type with Streamlit!")
	data = loadData()
	X_train, X_test, y_train, y_test, le = preprocessing(data)

	# Insert Check-Box to show the snippet of the data.
	if st.checkbox('Show Raw Data'):
		st.subheader("Showing raw data---->>>")	
		st.write(data.head())


	# ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["NONE","Decision Tree", "Neural Network", "K-Nearest Neighbours"])

	if(choose_model == "Decision Tree"):
		score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
		st.text("Accuracy of Decision Tree model is: ")
		st.write(score,"%")
		st.text("Report of Decision Tree model is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
				user_prediction_data = accept_user_data() 		
				pred = tree.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
		except:
			pass

	elif(choose_model == "Neural Network"):
		score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
		st.text("Accuracy of Neural Network model is: ")
		st.write(score,"%")
		st.text("Report of Neural Network model is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
				user_prediction_data = accept_user_data()
				scaler = StandardScaler()  
				scaler.fit(X_train)  
				user_prediction_data = scaler.transform(user_prediction_data)	
				pred = clf.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
		except:
			pass

	elif(choose_model == "K-Nearest Neighbours"):
		score, report, clf = Knn_Classifier(X_train, X_test, y_train, y_test)
		st.text("Accuracy of K-Nearest Neighbour model is: ")
		st.write(score,"%")
		st.text("Report of K-Nearest Neighbour model is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
				user_prediction_data = accept_user_data() 		
				pred = clf.predict(user_prediction_data)
				st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
		except:
			pass
	
	


	# Visualization Section

	# plt.hist(data['Member type'], bins=5)
	# st.pyplot()

if __name__ == "__main__":
	main()