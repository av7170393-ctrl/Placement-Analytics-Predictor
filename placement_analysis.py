import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Phase A: load ----------
df = pd.read_csv("student_placement_dataset.csv")

# ---------- Phase B: fixes ----------
names = {"CSE": "Computer Science & Engg", "IT": "Information Technology",
         "ECE": "Electronics & Communication", "MECH": "Mechanical Engg", "CIVIL": "Civil Engg"}
df["branch"] = df["branch"].map(names)
df["salary_lpa"] = (df["salary"] / 1e5).round(2)          # rupees -> LPA
df.loc[df["placed"] == 0, ["salary", "salary_lpa"]] = np.nan  # 0 was a placeholder
df = df.drop_duplicates()
print("Clean shape:", df.shape, "| missing salary:", df.salary_lpa.isna().sum(), "| unplaced:", (df.placed == 0).sum())
df.to_csv("student_placement_clean.csv", index=False)

# ---------- Phase C: numbers ----------
placed = df[df.placed == 1]
avg_pkg = placed.groupby("branch")["salary_lpa"].mean().sort_values(ascending=False).round(2)
top_co = placed["company"].value_counts().head(10)
rate = (df.groupby("branch")["placed"].mean() * 100).sort_values(ascending=False).round(1)
print("\nAVG PACKAGE (LPA)\n", avg_pkg.to_string())
print("\nTOP COMPANIES\n", top_co.to_string())
print("\nPLACEMENT RATE %\n", rate.to_string())
print("\nOverall placement rate:", round(df.placed.mean() * 100, 1))
print("Median package LPA:", placed.salary_lpa.median())

# ---------- Phase C: dashboard ----------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
sns.barplot(x=avg_pkg.values, y=avg_pkg.index, ax=ax[0], color="#3b6fd4")
ax[0].set_title("Average package by branch (LPA)"); ax[0].set_xlabel("LPA"); ax[0].set_ylabel("")
for i, v in enumerate(avg_pkg.values): ax[0].text(v + 0.1, i, f"{v}", va="center")
sns.barplot(x=top_co.values, y=top_co.index, ax=ax[1], color="#2a9d8f")
ax[1].set_title("Top 10 recruiting companies"); ax[1].set_xlabel("Students placed"); ax[1].set_ylabel("")
sns.barplot(x=rate.values, y=rate.index, ax=ax[2], color="#e76f51")
ax[2].set_title("Placement rate by branch (%)"); ax[2].set_xlabel("%"); ax[2].set_ylabel("")
for i, v in enumerate(rate.values): ax[2].text(v + 0.5, i, f"{v}%", va="center")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/placement_dashboard.png", dpi=120)
