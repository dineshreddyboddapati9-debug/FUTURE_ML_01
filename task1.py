import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Month": [1, 2, 3, 4, 5, 6],
    "Sales": [100, 120, 140, 160, 180, 200]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input and output
X = df[["Month"]]
y = df["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future sales (Month 7)
future_month = [[7]]
prediction = model.predict(future_month)

# Print output
print("Predicted sales for Month 7:", prediction[0])

# Save output to file
with open("output.txt", "w") as f:
    f.write("Predicted sales for Month 7: " + str(prediction[0]))

# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.title("Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")

# Save graph image
plt.savefig("output.png")

plt.show()