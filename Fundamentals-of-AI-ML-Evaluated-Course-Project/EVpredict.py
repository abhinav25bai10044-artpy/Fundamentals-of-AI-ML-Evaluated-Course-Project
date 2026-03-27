#=======================================================
# 1: IMPORTING THE NECESSARY MODULES
#=======================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import calendar as c

#=======================================================
# 2: READING THE DATASET (INPUT)
#=======================================================

filename="makeup_sales_dataset_2025.csv"
sales=pd.read_csv(filename)
print(sales.info())
print(sales.describe())
print("Memory Used")
print((sales.memory_usage()))
print("Any Duplicated Values:",sales.duplicated().sum())

#=======================================================
# 3: COLLECTING DATE FROM DATASET
#=======================================================

sales["Date"]=pd.to_datetime(sales["Date"],format="%d-%m-%Y")

#=======================================================
# 4: PLOTTING THE DATASET
#=======================================================

#Units sold vs Price
plt.figure()
plt.scatter(sales["Price_USD"],sales["Units_Sold"])
plt.xlabel("Price USD")
plt.ylabel("Units Sold")
plt.title("Items sold in 2025")
plt.grid()
plt.show()

#2 Units Sold vs Revenue
plt.figure()
plt.scatter(sales["Units_Sold"],sales["Revenue_USD"])
plt.xlabel("Units sold")
plt.ylabel("Revenue USD")
plt.title("Revenue by units in 2025")
plt.grid()
plt.show()

#3 Brand vs Revenue
plt.figure()
sales.groupby("Brand")["Revenue_USD"].sum().plot(kind="bar")
plt.xlabel("Brand")
plt.ylabel("Revenue USD")
plt.title("Revenue by each brand in 2025")
plt.show()
brand_rev=sales.groupby("Brand")["Revenue_USD"].sum()
max_brand=brand_rev.idxmax()
#4 Product type vs Units Sold
plt.figure()
sales.groupby("Product_Type")["Units_Sold"].sum().plot(kind="bar")
plt.xlabel("Product Type")
plt.ylabel("Units Sold")
plt.title("Type of product sold in 2025")
plt.show()
prod_rev=sales.groupby("Product_Type")["Units_Sold"].sum()
max_prod=prod_rev.idxmax()
#5 Country vs Revenue
country_rev=sales.groupby("Country")["Revenue_USD"].sum()
plt.figure()
plt.pie(country_rev,labels=country_rev.index, autopct="%1.1f%%")
plt.title("Revenue by country in 2025")
plt.grid()
plt.show()
max_country=country_rev.idxmax()
#6 Units Sold vs Month
sales["month"]=sales["Date"].dt.month
plt.figure()
sales.groupby("month")["Units_Sold"].sum().plot(marker="o")
plt.title("Total sales by Month in 2025")
plt.xlabel("Month")
plt.ylabel("Units Sold")
plt.grid()
plt.show()
month_rev=sales.groupby("month")["Units_Sold"].sum()
max_month=month_rev.idxmax()
max_val=month_rev.max()
m_name=c.month_name[max_month]

#=======================================================
# 5: TRAINING OUR MODEL (LINEAR REGRESSION)
#=======================================================

x=pd.get_dummies(sales.drop(["Sale_ID","Revenue_USD","Date"],axis=1))
y=sales["Revenue_USD"]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=123)
model=LinearRegression()
model.fit(X_train,Y_train)
predict=model.predict(X_test)

#=======================================================
# 6: FUTURE PREDICTION
#=======================================================

plt.figure()
plt.scatter(Y_test,predict)
plt.title("Actual vs Predicted Revenue")
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.grid()
plt.show()

#=======================================================
# 7: SUMMARISING THE DATASET AND THE MODEL (OUTPUT)
#=======================================================

Root_Mean_Squared_Error=np.sqrt(mean_squared_error(Y_test,predict))
print(
    f"""
Brand with Maximum Units sold:{max_brand}
    
Maximum products sold: {max_prod}
        
Maximum country sold in: {max_country}
        
Maximum sale in {m_name}:{max_val}
        
Number of data points:{len(Y_test)}
        
MSE: {mean_squared_error(Y_test,predict)}
        
Root Mean Squared Error: {Root_Mean_Squared_Error}
        
R^2: {r2_score(Y_test,predict)}"""
      )
if r2_score(Y_test,predict) > 0.8:
    print("Model Performance is Excellent")
elif r2_score(Y_test,predict) > 0.6:
    print("Model Performance is Good")
else:
    print("Model needs improvement")
print(f"""
Model Coefficients: 
{model.coef_}

Intercept: {model.intercept_}"""
      )
