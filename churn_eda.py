#獲取資料與清洗陷阱 

import pandas as pd #資料分析(用來讀取CSV 清洗資料 欄位轉換 統計)
import numpy as np #數值運算基底 做數學做模型
import matplotlib.pyplot as plt #最基礎的繪圖工具
import seaborn as sns #資料分析設計的高階繪圖 自動美化

#一個公開資料集 (IBM提供) 主題:電信客戶是否流失(Churn)
data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

print("正在下載資料集")

try:
    #pandas直接從網路讀取CSV 回傳DataFrame
    df = pd.read_csv(data_url)
    #df.shape [0]列數(樣本數) [1]欄位數(變數數) 檢查資料規格標準寫法
    print(f"資料載入成功，共有{df.shape[0]}筆顧客資料，每筆有{df.shape[1]}個欄位")
except Exception as e:
    print(f"無法下載，錯誤訊息:{e}")
    exit()

#資料品質檢核與清洗
#強制轉為數字，無法轉的變成 NaN 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')

#檢查有多少缺失值 isnull-True/False, sum-True會被當成1
missing_val=df.isnull().sum()
print("\n---缺失值檢查---")
#指顯示有缺失值的欄位
print(missing_val[missing_val > 0])

#填補缺失值 補成0
df['TotalCharges']=df['TotalCharges'].fillna(0)
#把類別墅轉成數值  Yes -> 1(流失), No -> 0(未流失) .map字典 每個值去字典找 找得到就換掉
df['Churn_Target'] = df['Churn'].map({'Yes':1, 'No':0})

#簡易瀏覽資料
print("\n---資料前五筆---")
print(df.head())
#顯示每個欄位型態
print("\n---資料型態檢查---")
print(df.info())

#統計圖表：流失比例
plt.figure(figsize=(6, 4)) #小型統計圖
#seaborn的計數圖 (自動:分組，計數，畫圖)
sns.countplot(x='Churn', data=df, palette='pastel')
#標題 暗示資料是否不平衡
plt.title('Churn Distribution (Imbalanced Data?)')
#Y標籤 :人數
plt.ylabel('Count')

#流失率 1=流失, 0=未流失 .mean()平均值
churn_rate=df['Churn_Target'].mean() * 100
#輸出流失率 :.2f-小數點兩位
print(f"\n整體流失率 (Churn Rate): {churn_rate:.2f}%")

plt.show()



