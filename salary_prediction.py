import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("student_placement_clean.csv")
df = df[df.placed == 1].copy()                     # salary exists only for placed students
num = ["cgpa", "avg_test_score", "technical_score", "aptitude_score", "num_projects", "num_internships"]
X = pd.concat([df[num], pd.get_dummies(df["branch"], prefix="branch")], axis=1)
y = df["salary_lpa"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42),
          "Gradient Boosting": GradientBoostingRegressor(random_state=42)}
base = mean_absolute_error(yte, np.full(len(yte), ytr.median()))
print("Baseline MAE (always predict median):", round(base, 2), "LPA")
res = {}
for n, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    res[n] = dict(MAE_LPA=mean_absolute_error(yte, p), R2=r2_score(yte, p))
print(pd.DataFrame(res).T.round(3))

rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.round(3).head(8))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
p = rf.predict(Xte)
ax[0].scatter(yte, p, s=6, alpha=.4, color="#3b6fd4"); lim = [0, 40]
ax[0].plot(lim, lim, "r--"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("Actual salary (LPA)"); ax[0].set_ylabel("Predicted (LPA)"); ax[0].set_title("Random Forest: actual vs predicted")
imp.head(8)[::-1].plot.barh(ax=ax[1], color="#2a9d8f"); ax[1].set_title("What drives salary?")
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/salary_model.png", dpi=120)
