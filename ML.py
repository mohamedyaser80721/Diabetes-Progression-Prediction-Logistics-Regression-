import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("diabetes.csv")
df = pd.DataFrame(data)

x = df[[
    "Pregnancies",
    "Glucose","BloodPressure","SkinThickness","Insulin",
    "BMI","DiabetesPedigreeFunction","Age"
]]

y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

pipe =Pipeline([
    ("Scalar",StandardScaler()),
    ("model",LogisticRegression())
])

pipe.fit(x_train,y_train)

predict = pipe.predict(x_test)

# predict = pipe.predict([[6,148,72,35,0,33.6,0.627,50]])

acc = accuracy_score(y_test, predict)

print(acc)
