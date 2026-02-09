#建立預測模型 
# Logistic Regression：看係數 (Coefficients)，解釋為什麼流失。
# Random Forest：看準確度，用來抓出流失客戶。


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE #處理不平衡的關鍵武器


data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
print("讀取資料準備中")
df=pd.read_csv(data_url)

df['TotalChargs']=pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn_Target']=df['Churn'].map({'Yes':1, 'No':0})


#特徵篩選
# 1. 刪除 gender, PhoneService (卡方檢定不顯著)
# 2. 刪除 TotalCharges (共線性太高)
# 3. 刪除 Churn (原始標籤) 和 customerID (流水號)
drop_cols=['gender', 'PhoneService', 'TotalCharges', 'Churn', 'customerID', 'Churn_Target']
print(f"正在刪除不顯著與共線性特徵:{drop_cols[:-2]}")

#定義X Y

X=df.drop(columns=drop_cols)
y=df['Churn_Target']

# 把剩下的類別變數轉成 0,1 數字，drop_first=True 是為了避免虛擬變數陷阱
X=pd.get_dummies(X, drop_first=True)

print(f"特徵處理完成 現在有{X.shape[1]}個特徵變數")


#切割訓練與測試
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#smote處理不平衡
print("\n正在使用SMOTE平衡資料")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote=smote.fit_resample(X_train, y_train)

print(f"SMOTE前流失樣本數:{sum(y_train == 1)}/SMOTE後流失樣本數:{sum(y_train_smote == 1)}(已平衡)")


#模型一 Logistic Regression
print("\n--模型一:Logistic Regression")
log_model=LogisticRegression(max_iter=1000)
log_model.fit(X_train_smote, y_train_smote)
#計算 Odds Ratio (勝算比)
#係數(Coef) 取 exponential 就會變成 Odds Ratio
#數值 > 1 代表增加流失機率，數值 < 1 代表減少流失機率
coefficients=pd.DataFrame({
    'Feature': X.columns,
    'Odds_Ratio':np.exp(log_model.coef_[0])
})

#排序
print("\n 商業決策表:流失率倍數")
print(coefficients.sort_values(by='Odds_Ratio', ascending=False).head(5))



#模型二 Random Forest (預測型模型)
print("\n---模型二:Random Forest")
rf_model=RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

y_pred_rf=rf_model.predict(X_test)

print("分類報告:")
print(classification_report(y_test, y_pred_rf))

#繪圖 混淆矩陣圖表
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confustion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()








