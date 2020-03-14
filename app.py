import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px


@st.cache
def loadData():
	df = pd.read_csv("avocado.csv").replace({"conventional": 0, "organic": 1})
	return df

# Basic preprocessing 
def preprocessing(df):
	# Assign X and y
	X = df.iloc[:, 2:-3].values
	y = df.iloc[:, -3].values

	
	# 1. Splitting X,y into Train & Test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)
	return X_train, X_test, y_train, y_test


# Training Decission Tree for Classification
@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, y_train, y_test):
	# Initiate the Classifier and fit the model.
	tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
	tree.fit(X_train, y_train)
	y_pred = tree.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)

	return score, report, tree

# Training Random Forest for Classification.
@st.cache(suppress_st_warning=True)
def Random_Forest(X_train, X_test, y_train, y_test):
	 
	
	# Initiate the Classifier and fit the model.
	clf = RandomForestClassifier(n_estimators=40,random_state=0)
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


# Accepting user data for predicting its Class
def accept_user_data():
    AveragePrice = st.text_input("What is the  AveragePrice ")
    total  = st.text_input("What is the Total Volume")
    sku4046 = st.text_input("How many sku4046 sales? ")
    sku4225 = st.text_input("How many sku4225 sales? ")
    sku4770 = st.text_input("How many sku4770 sales? ")
    Total_Bags = st.text_input("How many total Bags sales? ")
    smll_bags = st.text_input("How many smll_bags sales? ")
    large_bags = st.text_input("How many large_bags sales? ")
    xlarge_bags = st.text_input("How many xlarge_bags sales? ")
    user_prediction_data = np.array([AveragePrice,total,sku4046,sku4225,sku4770,Total_Bags,smll_bags,large_bags,xlarge_bags]).reshape(1,-1)
    return user_prediction_data



def main():
	st.title("Classification of Avocado Type with Streamlit!")
	data = loadData()
	X_train, X_test, y_train, y_test = preprocessing(data)

	# Insert Check-Box to show the snippet of the data.
	if st.checkbox('Wanna see the data?'):
		st.subheader("Gud here we are")
		st.write(data.head(30))
	# ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["NONE","Decision Tree", "Random_Forest", "K-Nearest Neighbours"])

	if(choose_model == "Decision Tree"):
		score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
		st.text("Accuracy of Decision Tree model is: ")
		st.write(score,"%")
		st.text("Report of Decision Tree model is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input?")):
				user_prediction_data = accept_user_data() 		
				pred = tree.predict_proba(user_prediction_data)
				st.write("The Predicted Class is: ", pred)
		except:
			pass

	elif(choose_model == "Random_Forest"):
		score, report, clf = Random_Forest(X_train, X_test, y_train, y_test)
		st.text("Accuracy of Random_Forest model is: ")
		st.write(score,"%")
		st.text("Report of Random_Forest model is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input?")):
				user_prediction_data = accept_user_data()
				
				
				
				pred = clf.predict_proba(user_prediction_data)
				st.write("The Predicted Class is: ", pred)
		except:
			pass

	elif(choose_model == "K-Nearest Neighbours"):
		score, report, clf = Knn_Classifier(X_train, X_test, y_train, y_test)
		st.text("Accuracy of K-Nearest Neighbour model is: ")
		st.write(score,"%")
		st.text("Report of K-Nearest Neighbour model is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input?")):
				user_prediction_data = accept_user_data() 		
				pred = clf.predict(user_prediction_data)
				st.write("The Predicted Class is: ", pred)
		except:
			pass
	
	


if __name__ == "__main__":
	main()