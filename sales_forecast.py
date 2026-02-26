import pandas as pd
from sklearn.linear_model import LinearRegression

# sample data
data = {
    "Days": [1,2,3,4,5],
    "Sales": [100,200,300,400,500]
}

df = pd.DataFrame(data)

X = df[["Days"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X,y)

prediction = model.predict([[6]])

print("Predicted Sales:", prediction)