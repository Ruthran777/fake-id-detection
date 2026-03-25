import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Improved dataset
data = {
    "followers": [120, 500, 80, 1000, 300, 50, 2000, 1500, 60, 40],
    "following": [300, 200, 1000, 300, 400, 800, 100, 150, 900, 1200],
    "posts": [5, 50, 2, 120, 20, 1, 300, 200, 3, 2],
    "bio_length": [10, 80, 5, 150, 40, 5, 200, 180, 3, 2],
    "profile_pic": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    "verified": [0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    "fake": [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df.drop("fake", axis=1)
y = df["fake"]

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model retrained successfully")