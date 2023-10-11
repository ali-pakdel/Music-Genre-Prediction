#!/usr/bin/env python
# coding: utf-8

# # Machine Learning
# ## AI-CA#4
# 
# ### Ali Pakdel Samadi
# ### 810198368

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("dataset.csv")
train_df


# # Phase Zero: EDA and Visualization
# 
# ### 1. Describe and Info functions

# In[60]:


train_df.describe()


# In[61]:


train_df.info()


# ### 2. Percentage of missing values in each column

# In[62]:


train_df.isnull().sum()/len(train_df) * 100


# ### 3. Histograms

# In[63]:


cols = ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 
        'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence', 'music_genre']

for col in cols:
    ax = sns.histplot(train_df[col], color="b")
    plt.title(col)
    plt.show()


# # Phase One: Preprocessing
# 
# ### 2.  Fill NaNs

# In[64]:


num_cols = ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 
        'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

obj_cols = ['artist_name', 'track_name', 'key', 'mode', 'music_genre']

print(train_df.isna().sum())

for col in num_cols:
    mean = train_df[col].mean()
    train_df[col] = train_df[col].replace(to_replace= np.nan, value= mean)

for col in obj_cols:
    mode = train_df[col].mode()
    train_df[col] = train_df[col].replace(to_replace= np.nan, value= mode[0])


    
print(train_df.isna().sum())


# ### 4. Standardization

# In[65]:


stan_df = train_df.copy()

for col in num_cols:
    mean = stan_df[col].mean()
    std = stan_df[col].std()
    stan_df[col] = (stan_df[col] - mean) / std

display(stan_df)


# ### 5. Encode object columns

# In[66]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

one_hot_cols = ['key', 'mode']

one_hot_df = stan_df[['key', 'mode']]

enc = OneHotEncoder()
one_hot_enc = enc.fit_transform(stan_df[one_hot_cols]).toarray()
enc_cols = enc.get_feature_names_out(one_hot_cols)
one_hot_df = pd.DataFrame(one_hot_enc, columns = enc_cols)

stan_df = stan_df.drop('key', 1)
stan_df = stan_df.drop('mode', 1)
stan_df = stan_df.drop('track_name', 1)
stan_df = stan_df.drop('music_genre', 1)

for col in enc_cols:
    stan_df[col] = one_hot_df[col]
    
    
enc = OrdinalEncoder()
enc.fit(stan_df[['artist_name']])
stan_df[['artist_name']] = enc.transform(stan_df[['artist_name']])

display(stan_df)


# ### 6. Artist Name column

# In[67]:


stan_df = stan_df.drop('artist_name', 1)


# ### 7. Information Gains

# In[68]:


from sklearn.feature_selection import mutual_info_classif

information_gain =  mutual_info_classif(stan_df, train_df['music_genre'], discrete_features=False)

counter = 0
for col in stan_df.columns:
    print(col, ' = ' ,information_gain[counter])
    counter += 1

plt.bar(stan_df.columns, information_gain)
plt.xticks(rotation='vertical')
plt.title('Information Gains')
plt.show()


# ### 8. Removing not usefull features

# In[69]:


counter = 0
for col in stan_df.columns:
    if information_gain[counter] < 0.1:
        stan_df = stan_df.drop(col, 1)
    counter += 1

stan_df


# # Phase Two: Model Training, Evaluation and Hyper Parameter Tuning
# 
# ### 2. Train-Test Split

# In[70]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(stan_df, train_df['music_genre'], test_size=0.3, random_state=42)

display(x_train, y_train)
display(x_test, y_test)


# ### 3. KNN

# In[71]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_list_n = []
test_list_n = []

test_preds = []

for k in range(1, 16):
    neigh = KNeighborsClassifier(n_neighbors= k)

    neigh_fit = neigh.fit(x_train, y_train)
    
    train_pred = neigh_fit.predict(x_train)
    train_list_n.append(accuracy_score(y_train, train_pred))

    test_pred = neigh_fit.predict(x_test)
    test_preds.append(test_pred)
    test_list_n.append(accuracy_score(y_test, test_pred))

plt.plot(np.linspace(1, 15, 15), train_list_n, label='Train')
plt.plot(np.linspace(1, 15, 15), test_list_n, label='Test')
plt.legend(loc="best")
plt.title('KNN Accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()


# ### 4. Decision Tree -> Max Depth

# In[72]:


from sklearn.tree import DecisionTreeClassifier

def __DecisionTree(depth, min_leaf):
    
    tree = DecisionTreeClassifier(max_depth= depth, min_samples_leaf= min_leaf)

    tree_fit = tree.fit(x_train, y_train)

    train_pred_tree = tree_fit.predict(x_train)
    test_pred_tree = tree_fit.predict(x_test)
    
    return accuracy_score(y_train, train_pred_tree), accuracy_score(y_test, test_pred_tree), test_pred_tree

train_list_tree_d = []
test_list_tree_d = []
test_preds_tree_d = []

for k in range(1, 16):
    tr, te, pred = __DecisionTree(k, 1)
    train_list_tree_d.append(tr)
    test_list_tree_d.append(te)
    test_preds_tree_d.append(pred)
    
plt.plot(np.linspace(1, 15, 15), train_list_tree_d, label='Train')
plt.plot(np.linspace(1, 15, 15), test_list_tree_d, label='Test')
plt.legend(loc="best")
plt.title('Decision Tree Accuracy By Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.show()


# ### 4. Decision Tree -> Min Samples Leaf

# In[73]:


train_list_tree_l = []
test_list_tree_l = []
test_preds_tree_l = []

for l in range(1, 16):
    tr, te, pred = __DecisionTree(None, l)
    train_list_tree_l.append(tr)
    test_list_tree_l.append(te)
    test_preds_tree_l.append(pred)
    
plt.plot(np.linspace(1, 15, 15), train_list_tree_l, label='Train')
plt.plot(np.linspace(1, 15, 15), test_list_tree_l ,label='Test')
plt.legend(loc="best")
plt.title('Decision Tree Accuracy By Min Leaf')
plt.xlabel('Min Sample Leaf')
plt.ylabel('Accuracy')
plt.show()


# ### 6. Recall, Precision, Accuracy, F1 Score

# ### For KNN

# In[74]:


from sklearn.metrics import classification_report

best_index = test_list_n.index(max(test_list_n))

print("Best K:", best_index + 1)
print(classification_report(y_test, test_preds[best_index]))
print("Accuracy for train: ", train_list_n[best_index])
print("Accuracy for test: ", test_list_n[best_index])


# ### For Decision Tree by Max Depth

# In[75]:


best_index1 = test_list_tree_d.index(max(test_list_tree_d))

print("Best Max-Depth:", best_index1 + 1)
print(classification_report(y_test, test_preds_tree_d[best_index1]))
print("Accuracy for train: ", train_list_tree_d[best_index1])
print("Accuracy for test: ", test_list_tree_d[best_index1])


# ### For Decision Tree by Min Sample Leaf

# In[76]:


best_index2 = test_list_tree_l.index(max(test_list_tree_l))

print("Best Min-Sample-Leaf:", best_index2 + 1)
print(classification_report(y_test, test_preds_tree_l[best_index2]))
print("Accuracy for train: ", train_list_tree_l[best_index2])
print("Accuracy for test: ", test_list_tree_l[best_index2])


# ### For Decision Tree by Max Depth and Min Sample Leaf

# In[77]:


train_acc, test_acc, pred = __DecisionTree(best_index1 + 1, best_index2 + 1)

print(classification_report(y_test, pred))
print("Accuracy for train: ", train_acc)
print("Accuracy for test: ", test_acc)


# # Phase Three: Ensemble Methods 
# 
# ### 1. Random Forest by Max_Depth

# In[78]:


from sklearn.ensemble import RandomForestClassifier

def __RandomForest(depth, n_estimator, min_leaf):
    
    forest = RandomForestClassifier(max_depth= depth, n_estimators= n_estimator, min_samples_leaf= min_leaf)

    forest_fit = forest.fit(x_train, y_train)

    train_pred_forest = forest_fit.predict(x_train)

    test_pred_forest = forest_fit.predict(x_test)
    
    return accuracy_score(y_train, train_pred_forest), accuracy_score(y_test, test_pred_forest), test_pred_forest

train_list_forest_d = []
test_list_forest_d = []
test_preds_forest_d = []

for depth in range(1, 16):
    tr, te, pred = __RandomForest(depth, 100, 1)
    train_list_forest_d.append(tr)
    test_list_forest_d.append(te)
    test_preds_forest_d.append(pred)
    
plt.plot(np.linspace(1, 15, 15), train_list_forest_d, label='Train')
plt.plot(np.linspace(1, 15, 15), test_list_forest_d ,label='Test')
plt.legend(loc="best")
plt.title('Random Forest Accuracy By Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.show()


# ### Random Forest by n_estimators

# In[91]:


train_list_forest_n = []
test_list_forest_n = []
test_preds_forest_n = []

for n in range(1, 51):
    tr, te, pred = __RandomForest(None, n, 1)
    train_list_forest_n.append(tr)
    test_list_forest_n.append(te)
    test_preds_forest_n.append(pred)
    
plt.plot(np.linspace(1, 50, 50), train_list_forest_n, label='Train')
plt.plot(np.linspace(1, 50, 50), test_list_forest_n ,label='Test')
plt.legend(loc="best")
plt.title('Random Forest Accuracy By n-Estimators')
plt.xlabel('n-Estimators')
plt.ylabel('Accuracy')
plt.show()


# ### Random Forest by min_samples_leaf

# In[80]:


train_list_forest_l = []
test_list_forest_l = []
test_preds_forest_l = []

for l in range(1, 16):
    tr, te, pred = __RandomForest(None, 100, l)
    train_list_forest_l.append(tr)
    test_list_forest_l.append(te)
    test_preds_forest_l.append(pred)
    
plt.plot(np.linspace(1, 15, 15), train_list_forest_l, label='Train')
plt.plot(np.linspace(1, 15, 15), test_list_forest_l ,label='Test')
plt.legend(loc="best")
plt.title('Random Forest Accuracy By Min Sample Leaf')
plt.xlabel('Min Simple Leaf')
plt.ylabel('Accuracy')
plt.show()


# ### For Max_Depth

# In[81]:


best_index3 = test_list_forest_d.index(max(test_list_forest_d))

print("Best Max-Depth:", best_index3)
print(classification_report(y_test, test_preds_forest_d[best_index3]))
print("Accuracy for train: ", train_list_forest_d[best_index3])
print("Accuracy for test: ", test_list_forest_d[best_index3])


# ### For N_Estimators

# In[92]:


best_index4 = test_list_forest_n.index(max(test_list_forest_n))

print("Best n:", best_index4)
print(classification_report(y_test, test_preds_forest_n[best_index4]))
print("Accuracy for train: ", train_list_forest_n[best_index4])
print("Accuracy for test: ", test_list_forest_n[best_index4])


# ### For Min_Samples_Leaf

# In[83]:


best_index5 = test_list_forest_l.index(max(test_list_forest_l))

print("Best min_leaf:", best_index5)
print(classification_report(y_test, test_preds_forest_l[best_index5]))
print("Accuracy for train: ", train_list_forest_l[best_index5])
print("Accuracy for test: ", test_list_forest_l[best_index5])


# ### 4. Confusion Matrix

# In[94]:


from sklearn.metrics import ConfusionMatrixDisplay

forest = RandomForestClassifier(max_depth= 12, n_estimators=41 , min_samples_leaf= 7)
forest_fit = forest.fit(x_train, y_train)
pred_forest = forest_fit.predict(x_test)
print(classification_report(y_test, pred_forest))
ConfusionMatrixDisplay.from_estimator(forest, x_train, y_train)

