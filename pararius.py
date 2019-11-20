import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_json("pararius.json")

x = dataset[['area']]
y = dataset['rent']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=['Coefficient'])


y_prediction = regressor.predict(x_test)
df = pd.DataFrame({'Actual':y_test,'Predicted':y_prediction})

average_scores = 0
trialNum = 500
for i in range(trialNum):
  regressor = LinearRegression()
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
  regressor.fit(x_train,y_train)
  score = regressor.score(x_test,y_test)
  average_scores += score
  
average_scores = average_scores / trialNum
print("average_score", average_scores)
#plt.plot(x,y,"bo")
#plt.plot(x_test,y_prediction)
#plt.show()
