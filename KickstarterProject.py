#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import seaborn as sns   
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#importing functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#importing Classification ML Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


#style for the visualizations
sns.set_style('darkgrid')


# # Data Collection
# 

# In[3]:



project_data =pd.read_csv(r"C:\Users\HOME\Desktop\Kickstarter project\kickstarter_data_full.csv")


# # Data Cleaning

# In[4]:


project_data.head()


# In[5]:


project_data.columns


# In[6]:


#Selecting the relevant columns for the use case, available input features for the user

df = project_data[['goal', 'country', 'deadline', 'launched_at', 'staff_pick', 'backers_count', 'usd_pledged', 'category', 'state']]


# In[7]:


df.head()


# In[8]:


#2) Statistical analysis 
df.info()


# In[9]:


df.shape


# In[10]:


#summary statistics rounded off to 1 decimal place
df.describe().round(1)


# In[11]:


#3) Handling null values


# In[12]:


df.isnull().sum()


# In[13]:


#Removing rows which contain null value. We require information about the category of the project.
df = df.dropna()


# In[14]:


df.isnull().sum()


# In[15]:


df['state'].unique().tolist()


# In[16]:


#The purpose of the model is to determine wether the project will fail or be successful.
#4) Thus only the data related to failed or successful projects will be used for training.

df = df[(df['state'] == 'successful') | (df['state'] == 'failed')]


# In[17]:


df['state'].unique().tolist()


# In[18]:


#Saved cleaned data to an excel file. This excel file is used for data analysis in PowerBI.
#set the file path where you want to save the Excel file
# file_path = r'C:\Users\HOME\Desktop\Kickstarter project\my_data1.xlsx'

# # write the data frame to an Excel file
# df.to_excel(file_path, index=False)

# print(f"Data frame saved as Excel file to {file_path}")


# # Data Preprocessing

# In[19]:


#Making the data less complex by extracting the date from deadline and launched at columns


# In[20]:


# Convert the 'launched_at' column to datetime objects
df['launched_at'] = pd.to_datetime(df['launched_at'], format='%Y-%m-%d %H:%M:%S')

# Convert the datetime objects to strings in 'dd-mm-yyyy' format
df['launched_at'] = df['launched_at'].apply(lambda x: x.strftime('%d-%m-%Y'))


# In[21]:


# Convert the 'deadline' column to datetime objects
df['deadline'] = pd.to_datetime(df['deadline'], format='%Y-%m-%d %H:%M:%S')

# Convert the datetime objects to strings in 'dd-mm-yyyy' format
df['deadline'] = df['deadline'].apply(lambda x: x.strftime('%d-%m-%Y'))


# In[ ]:





# In[22]:


df.head()


# In[23]:


#Label encoding


# In[24]:


df.head()


# In[25]:


from sklearn.preprocessing import LabelEncoder

# Assume you have a DataFrame called 'df' with categorical columns: 'state', 'country', and 'category'
categorical_cols = ['state', 'country', 'category','staff_pick','deadline','launched_at']

# Initialize LabelEncoder
le = LabelEncoder()

# Loop through categorical columns and encode them
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

    



# # Exploratory Data Analysis

# In[26]:


#Relational Analysis


# In[83]:


#Fig 2: correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
plt.title('Fig 2: Correlation Matrix',fontsize=15)


# In[82]:


#Fig 3: Pairplot
#numerical columns and state column is visualized in the pairplot
pairplot_cols = ['goal', 'backers_count', 'usd_pledged','state']
pairplot_df= df[pairplot_cols]
plt.figure(figsize=(15,15))
sns.pairplot(data=pairplot_df, hue='state')
plt.suptitle('Fig 3: Pairplot', fontsize=15, y=1.03, horizontalalignment='center')


# In[81]:


#Data Imbalance
#to check if the data imbalanced
pot_lbl = df.state.value_counts()

#Fig 4: Data imbalance barplot 
plt.figure(figsize=(8,5))
sns.barplot(x=pot_lbl.index, y=pot_lbl)
plt.xlabel('State',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.suptitle('Fig 4: Data imbalance barplot', fontsize=15, y=1.03, horizontalalignment='center')


# In[67]:


#As the Data imbalance is not large it does not need to be fixed
#It will need to be taken into consideration during evaluating the ML model which will be used on the data


# # Data Preparation

# In[29]:


# 1.Train test split


# In[30]:


from sklearn.model_selection import train_test_split

# Define the features (independent variables) and the target variable
X = df.drop('state', axis=1)
y = df['state']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[31]:


print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)


# In[32]:


#2. Feature engineering


# In[33]:


#Feature Scaling


# In[34]:


#features of the given dataset fluctuate significantly within their ranges
#They are recorded in various units of measurement
#We will use Standard Scaler
''' standard scaler '''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[35]:


#Feature selection


# In[36]:


# lets see feature importance
from sklearn.ensemble import ExtraTreesClassifier
x = df.drop(['state'],axis=1) #independent variables
y =df.state #dependent variables
Ext = ExtraTreesClassifier()
Ext.fit(x,y)
print(Ext.feature_importances_)


# In[80]:


#plotting the importance of features
plt.suptitle('Fig 15: Feature Importance', fontsize=15, y=1.03, horizontalalignment='center')
feature = pd.Series(Ext.feature_importances_,index=x.columns)
feature.sort_values(ascending=True).nlargest(10).plot(kind='barh')


# In[38]:


# define a threshold for feature importance score
threshold = 0.05

# get the names of features with importance score less than the threshold
to_drop = feature[feature < threshold].index.tolist()

# drop the less important features from the dataset
X_selected = X.drop(to_drop, axis=1)

# print the remaining features
print(X_selected.columns)


# In[39]:


#Create new training and testing data which includes only the selected features


# In[40]:


# create a new DataFrame containing only the selected columns
df1 = df.drop('country', axis=1)


# In[41]:


from sklearn.model_selection import train_test_split

# Define the features (independent variables) and the target variable
X = df1.drop('state', axis=1)
y = df1['state']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# # Model Building 

# In[68]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Define the list of models to evaluate
models = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(), GaussianNB(), SVC(), GradientBoostingClassifier()]

# Define empty lists to store the performance metrics of each model
models_acc = []
models_prec = []
models_recall = []
models_f1 = []
models_roc_auc = []

# Iterate over the models and evaluate their performance
for model in models:
    # 1. Model Training
    model.fit(X_train, y_train)
    
    # Predict the labels of the test data
    pred = model.predict(X_test)
    
    #2.Model Evaluation
    # Compute the performance metrics and append them to the corresponding lists
    models_acc.append(accuracy_score(y_test, pred))
    models_prec.append(precision_score(y_test, pred))
    models_recall.append(recall_score(y_test, pred))
    models_f1.append(f1_score(y_test, pred))
    models_roc_auc.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)))



    


# In[69]:


''' creating dataframe '''
res = pd.DataFrame({
    "Model Name": ['LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'GaussianNB', 'SVC','GradientBoosting'],
    'Accuracy': models_acc, 
    'Precision': models_prec,
    'Recall': models_recall,
    'F1-Score': models_f1,
    'ROC-AUC': models_roc_auc
    
})


# In[70]:


res.sort_values(by=['Accuracy'], ascending=False)


# In[86]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 10))

# define colors for each metric
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# sort models by accuracy
res = res.sort_values(by=["Accuracy"], ascending=False)

# plot horizontal bar chart with different colors for each metric
res.plot(x="Model Name", y=["Accuracy", "Precision", "Recall", "F1-Score","ROC-AUC"], kind="barh", ax=ax, color=colors, width=0.75)

# add title and axis labels
ax.set_title("Fig 16: Model Performance Metrics", fontsize=18, fontweight="bold")

ax.set_xlabel("Score", fontsize=14)
ax.set_ylabel("Model Name", fontsize=14)

# add grid lines
ax.grid(axis="x", linestyle="dashed", alpha=0.7)

plt.show()


# In[ ]:


#KNeighboursClassifier has the best performance and is thereby selected 

import pickle
from sklearn.neighbors import KNeighborsClassifier



# Create a KNeighborsClassifier model
model = KNeighborsClassifier(n_neighbors=2)

# Train the model on the training data
model.fit(X_train, y_train)

# Save the model to a pickle file
filename = 'model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the training data
train_data = pd.read_csv('train_data.csv')

# Fit and save the label encoders
deadline_le = LabelEncoder()
deadline_le.fit(train_data['deadline'])
with open('deadline_le.pkl', 'wb') as f:
    pickle.dump(deadline_le, f)

launched_at_le = LabelEncoder()
launched_at_le.fit(train_data['launched_at'])
with open('launched_at_le.pkl', 'wb') as f:
    pickle.dump(launched_at_le, f)

staff_pick_le = LabelEncoder()
staff_pick_le.fit(train_data['staff_pick'])
with open('staff_pick_le.pkl', 'wb') as f:
    pickle.dump(staff_pick_le, f)
