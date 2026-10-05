from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["target_name"] = df["target"].map({0: "malignant", 1: "benign"})

print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

numeric = df.drop(columns=["target_name"])
print("Mean:\n", numeric.mean())
print("Median:\n", numeric.median())
print("Mode:\n", numeric.mode().iloc[0])
print("Standard deviation:\n", numeric.std())
print("Variance:\n", numeric.var())

print("Correlation with target:\n", numeric.corr()["target"].sort_values())

df["target_name"].value_counts().plot(kind="bar", title="Class Distribution")
plt.tight_layout()
plt.show()

df.groupby("target_name")["mean radius"].mean().plot(kind="bar",
    title="Average Mean Radius by Diagnosis")
plt.tight_layout()
plt.show()

plt.scatter(df["mean radius"], df["mean texture"], alpha=0.55)
plt.xlabel("Mean Radius"); plt.ylabel("Mean Texture")
plt.title("Mean Radius vs Mean Texture")
plt.tight_layout()
plt.show()
