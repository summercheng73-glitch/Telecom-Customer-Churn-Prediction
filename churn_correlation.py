#相對係數矩陣 (統計問題： 如果這兩個變數高度相關，
# 把它們同時放進模型會導致 共線性，讓模型的係數亂跳，失去解釋力。)

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

data_url= "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df=pd.read_csv(data_url)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn_Target']=df['Churn'].map({'Yes':1, 'No':0})

# 把 'tenure', 'MonthlyCharges', 'TotalCharges' 拿出來分析
numerical_data=df[['tenure', 'MonthlyCharges', 'TotalCharges']]

#計算關係數
corr_matrix=numerical_data.corr()

print("相關係數矩陣")
print(corr_matrix)

#繪圖熱力圖
plt.figure(figsize=(8, 6))
# annot=True 會把數字顯示在格子裡，cmap='coolwarm' 是紅藍配色
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix(Multicollinearity Check)')
plt.show()

