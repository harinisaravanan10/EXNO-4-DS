# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df.head()
```

<img width="358" height="247" alt="image" src="https://github.com/user-attachments/assets/8ade0c84-de4f-4514-9ef5-bcf885d519c4" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```

<img width="157" height="252" alt="image" src="https://github.com/user-attachments/assets/6b3a00b0-de84-4700-b687-59387d701066" />

df.dropna()

<img width="352" height="503" alt="image" src="https://github.com/user-attachments/assets/a50befaa-d605-4e5f-b360-f8e80a530af9" />


```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```

<img width="170" height="179" alt="image" src="https://github.com/user-attachments/assets/9a3e84ac-72aa-4a28-93e0-07a5a93ef039" />


```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```

<img width="333" height="251" alt="image" src="https://github.com/user-attachments/assets/15047fb0-c25c-4ab5-8035-afa6330e2317" />


```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

<img width="433" height="434" alt="image" src="https://github.com/user-attachments/assets/66cc5560-57a8-42f8-849d-1114487bcb97" />


```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="433" height="454" alt="image" src="https://github.com/user-attachments/assets/6189fee3-3d34-4b4d-ac9d-2dcac6c287af" />


```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

<img width="402" height="531" alt="image" src="https://github.com/user-attachments/assets/020601ba-9da6-433b-9656-86ddfabe9c3a" />


```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```

<img width="386" height="245" alt="image" src="https://github.com/user-attachments/assets/671606f2-b4ec-447b-8d82-64d79fcca4b6" />

```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```

<img width="462" height="454" alt="image" src="https://github.com/user-attachments/assets/3530a5ff-0068-4670-af00-b81c7f2e74f4" />


```
df_null_sum=df.isnull().sum()
df_null_sum
```

<img width="267" height="574" alt="image" src="https://github.com/user-attachments/assets/217a9e6a-39cc-4661-ae48-afacee5abb3f" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

<img width="1101" height="528" alt="image" src="https://github.com/user-attachments/assets/89a357aa-2c3f-46f8-9181-2556fa451b9c" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="907" height="519" alt="image" src="https://github.com/user-attachments/assets/f0d7c66c-2613-4f3a-82e7-40efd87b39d0" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

<img width="454" height="91" alt="image" src="https://github.com/user-attachments/assets/c81179c7-55e3-4b91-8832-a7adfedb6e09" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```

<img width="476" height="452" alt="image" src="https://github.com/user-attachments/assets/f7f81d37-81ca-432e-80e8-b1afbe959b70" />

```
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

<img width="1202" height="530" alt="image" src="https://github.com/user-attachments/assets/3519d6fa-827e-4d5f-8eb9-82d16286ca12" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="1007" height="525" alt="image" src="https://github.com/user-attachments/assets/d57267e7-a5eb-44ed-998f-70747709c311" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```

<img width="897" height="106" alt="image" src="https://github.com/user-attachments/assets/067bfba1-7cb0-4e14-8548-30ad8d0999f9" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

<img width="476" height="99" alt="image" src="https://github.com/user-attachments/assets/c1df7bd8-55f3-4a93-b203-4bc5b7195601" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```

<img width="582" height="56" alt="image" src="https://github.com/user-attachments/assets/06771189-9a5e-4b47-a120-a6cfe20e0dcc" />

```
!pip install skfeature-chappers
```

<img width="1582" height="386" alt="image" src="https://github.com/user-attachments/assets/9eb8cc0b-5668-4ec8-855c-fe989d090033" />

```
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```

<img width="1028" height="520" alt="image" src="https://github.com/user-attachments/assets/d6ab7a4a-a0d7-4142-8002-ea4a3423ebed" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 k_anova = 5
 selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
 X_anova = selector_anova.fit_transform(X, y)
 selected_features_anova = X.columns[selector_anova.get_support()]
 print("\nSelected features using ANOVA:")
 print(selected_features_anova)
```

<img width="901" height="143" alt="image" src="https://github.com/user-attachments/assets/ceee7443-aa03-4013-b6b0-6e0b3ddc31c2" />

```
 # Wrapper Method
 import pandas as pd
 from sklearn.feature_selection import RFE
 from sklearn.linear_model import LogisticRegression
 df=pd.read_csv("/content/income(1) (1).csv")
 # List of categorical columns
 categorical_columns = [
 'JobType',
 'EdType',
 'maritalstatus',
 'occupation',
 'relationship',
 'race',
 'gender',
 'nativecountry'
 ]
 # Convert the categorical columns to category dtype
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```

<img width="1187" height="576" alt="image" src="https://github.com/user-attachments/assets/2d285f60-bfe6-40dc-9ea4-9b7d3eedf0b2" />

```
X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 logreg = LogisticRegression()
 n_features_to_select =6
 rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
 rfe.fit(X, y)
```

<img width="1373" height="1049" alt="image" src="https://github.com/user-attachments/assets/e5ba0ed2-12d6-40c7-a1de-7d6888e4de92" />
<img width="311" height="188" alt="image" src="https://github.com/user-attachments/assets/160febc4-fc99-4f49-a8c4-6780abd51c14" />

# RESULT:
  The given data and perform Feature Scaling and Feature Selection process and save the data to a file is succussfully verified.
