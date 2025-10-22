## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="406" height="416" alt="501905965-2446f85c-e9f1-4ce8-95bc-894c81b47abf" src="https://github.com/user-attachments/assets/cbe17f42-bd30-4d7e-bb60-1e8e7bbd72da" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```
<img width="331" height="241" alt="501906109-d6337fbb-c262-4df3-99b2-07cdfa11084a" src="https://github.com/user-attachments/assets/6e5d0fa2-6c6f-4dad-866c-4e3ab7bf857d" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="490" height="428" alt="501906301-b16dd1c5-52b4-4cdb-81a4-37d3052deda3" src="https://github.com/user-attachments/assets/4ce76940-f6b5-4180-97c5-272c5c50aaf8" />

```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```

<img width="539" height="438" alt="501906489-d62f3b54-8fa6-44d9-83ef-3fca880d69e1" src="https://github.com/user-attachments/assets/2930d9ad-42ea-4dec-8329-b98e8182c777" />

```
 from sklearn.preprocessing import OneHotEncoder
 ohe=OneHotEncoder()
 df2=df.copy()
 enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
 df2=pd.concat([df2,enc],axis=1)
 df2
```
<img width="799" height="414" alt="501907198-9d25fc9e-2916-4a3b-b833-4b9abd2348e7" src="https://github.com/user-attachments/assets/aca207c8-38c2-4160-b7c5-10a9d569179a" />

```
 pd.get_dummies(df2,columns=["nom_0"])
```
<img width="833" height="498" alt="501907373-200a76b3-b057-4fde-9371-33449fcb2891" src="https://github.com/user-attachments/assets/6108c104-2460-451c-8259-455c2b41cd3f" />

```
 pip install--upgrade category_encoders
```
<img width="824" height="325" alt="501907571-0e2b1754-95f1-4dab-90c5-6aa25246f164" src="https://github.com/user-attachments/assets/7c2129e4-dd9f-418b-a208-742f4bb4f2fe" />

```

 from category_encoders import BinaryEncoder
 df=pd.read_csv("data.csv")
 df
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 df
 dfb=pd.concat([df,nd],axis=1)
 dfb
```
<img width="787" height="393" alt="501907769-a439c635-dd1f-4da0-9d7a-a00a8838d341" src="https://github.com/user-attachments/assets/0ff4c4ac-a7fc-4809-9216-167ac39d6382" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="730" height="433" alt="501908338-c4f5d74b-11d4-40bf-bd29-14047165661d" src="https://github.com/user-attachments/assets/a8e45e1d-03d8-4b33-a5d4-4c468c286578" />

```
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```
<img width="821" height="403" alt="501908485-214b8475-03c0-4645-8f3f-4d67ce83745d" src="https://github.com/user-attachments/assets/3b7ad2a4-6f27-48b7-9c9f-2f5b74ecf37a" />

```
df.skew()
```
<img width="608" height="342" alt="501908592-8bc2556d-9b07-40b2-8f48-e245713f1220" src="https://github.com/user-attachments/assets/e049305a-7196-4ad8-bc30-ecd531dabcd0" />

```
 np.log(df["Highly Positive Skew"])
```
<img width="475" height="746" alt="501908887-a4d8b45e-ec12-4b2d-8141-7588db0b0113" src="https://github.com/user-attachments/assets/0fc8ea0f-9af3-422d-b3c4-94005ea91dc1" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="534" height="757" alt="501909050-cfc33537-42c3-42bb-9c7c-920b4b6e4651" src="https://github.com/user-attachments/assets/ea724683-1d9d-4ef5-be22-057fcf97b464" />

```
 np.sqrt(df["Highly Positive Skew"])
```
<img width="538" height="749" alt="501909252-3b8a4732-234f-40a8-b2e7-696ab71aa2aa" src="https://github.com/user-attachments/assets/ab88304a-f880-4838-bd98-671d347df947" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```
<img width="810" height="400" alt="501909495-27f4136e-ace1-43c5-963e-ee5e6146eea3" src="https://github.com/user-attachments/assets/155681b8-0684-48f9-bfa3-69311225a75d" />

```
 df.skew()
```
<img width="454" height="272" alt="501909599-c6748993-2cbf-4159-a2e5-47814585fa92" src="https://github.com/user-attachments/assets/1f7e77d7-3cc1-49d8-bc8b-347bde88362b" />

```
 df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
 df.skew()
```
<img width="617" height="309" alt="501910013-2e000a8e-8a7f-46c6-95ee-378b84bd0064" src="https://github.com/user-attachments/assets/50a8b605-3f94-44ad-a75d-9f155fe0dcc2" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```
<img width="836" height="579" alt="501910447-af6ddf5d-24de-4d62-b77e-a9d65771da70" src="https://github.com/user-attachments/assets/c0bc5537-cd96-4bcf-b1cb-bfee6d54d1eb" />

```
 import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
<img width="775" height="582" alt="501910688-2fdbab49-2a88-45a3-ba83-a093f0db13f6" src="https://github.com/user-attachments/assets/2f47493f-2061-47b9-99b3-b37539176260" />

```
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()
```
<img width="771" height="581" alt="501911410-50f157fb-b586-4977-8952-62a5c927c009" src="https://github.com/user-attachments/assets/e6c860ac-ff97-4b72-8a53-4a5f2c386685" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
<img width="776" height="558" alt="501911597-3b0b3a9f-0e48-4456-a71f-19781ede905e" src="https://github.com/user-attachments/assets/8d0b687e-e6c2-4a23-9b20-1453fbd82089" />

```
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```
<img width="799" height="601" alt="501911818-4dad7620-a9f1-4925-a1c2-be6942139474" src="https://github.com/user-attachments/assets/278e34f1-93ff-4657-aaf5-5e65ff768213" />

```
dt=pd.read_csv("titanic_dataset.csv")
 dt
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 dt["Age_1"]=qt.fit_transform(dt[["Age"]])
 sm.qqplot(dt['Age'],line='45') 
plt.show()
```
<img width="729" height="529" alt="501912029-5a8773fe-7a3e-4f94-a0af-14449b487f6b" src="https://github.com/user-attachments/assets/adb580aa-f2e1-4b74-a60e-e6a04a6b2773" />

```
 sm.qqplot(df["Highly Negative Skew_1"],line='45')
 plt.show()
```
<img width="736" height="513" alt="501912219-2277e70f-0598-4aa6-95cb-a59d292672e7" src="https://github.com/user-attachments/assets/37ff3213-8159-4b5b-a572-a9110b84fb1d" />
      
# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully

       
