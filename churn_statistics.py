#統計檢定與特徵篩選 
# 1.類別變數(如性別、合約) 2.數值變數(如月費、年資)：用 T 檢定

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats #統計檢定核心 幾乎所有統計假設檢定都在這

data_url="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
print("正在讀取")
df=pd.read_csv(data_url)

#清洗TotalCharges
df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')
#缺失值補0
df['TotalCharges']=df['TotalCharges'].fillna(0)
#將類別標籤轉為二元數值
df['Churn_Target']=df['Churn'].map({'Yes': 1, 'No': 0})

print("資料準備完成")

#卡方檢定 (性別、合約類型、支付方式，跟流失有沒有關係？)
#指定要檢定的類別型特徵 這些資料都是 nominal/categorical
cat_features=['gender', 'Partner', 'Dependents', 'PhoneService',
               'InternetService', 'Contract', 'PaymentMethod']
#<20左對齊 寬度20
print(f"{'變數名稱':<20} | {'P-Value':<12} | {'結論'}")
print("-" * 50)

#用來儲存 與流失顯著相關 的類別變數
significant_cat_features = []

#逐一對每個類別變數進行卡方檢定
for col in cat_features: 
    #建立列聯表 (交叉表) 行:某個類別變數 列:是否流失
    contingency_table=pd.crosstab(df[col], df['Churn'])
    #進行卡方檢定 chi2卡方檢定、p-value、dof自由度、expected期望次數
    chi2, p, dof, expected=stats.chi2_contingency(contingency_table)
    #判斷結果
    significance="顯著" if p < 0.05 else "不顯著"
    #將 與流失有關的類別變數 存起來
    if p < 0.05:
        significant_cat_features.append(col)
    #.2e 科學記號 常用於p-value
    print(f"{col:<20} | {p:.2e}   | {significance}")

#T檢定(流失客戶與留存客戶，他們的「平均月費」或「年資」是否不同？)
#指定數值型變數
num_features=['tenure', 'MonthlyCharges', 'TotalCharges']

#輸出T檢定結果表頭
print("\n" + "="*50 + "\n")
print(f"{'數值變數':<20} | {'P-Value':<12} | {'結論'}")
print("-" *50)

#對每個數值進行T檢定
for col in num_features:
    #分成兩群 流失(yes) 留存(no)
    churn_yes=df[df['Churn'] == 'Yes'][col]
    churn_no=df[df['Churn'] == 'No'][col]

    #進行T檢定
    t_stat, p_val = stats.ttest_ind(churn_yes, churn_no, equal_var=False)

    significance="顯著差異" if p_val < 0.05 else "無差異"
    print(f"{col:<20} | {p_val:.2e}    | {significance}")

#視覺化(畫出最具代表性的：合約(Contract)與年資(tenure))
fig, axes=plt.subplots(1, 2, figsize=(14,5))

sns.countplot(x='Contract', hue='Churn', data=df, ax=axes[0], palette='Set2')
axes[0].set_title('Churn by Contract Type (Chi-Square Significant)')

sns.boxplot(x='Churn', y='tenure', data=df, ax=axes[1], palette='pastel')
axes[1].set_title('Tenure Distribution by Churn (T-Test Significant)')

plt.tight_layout()
plt.show()





