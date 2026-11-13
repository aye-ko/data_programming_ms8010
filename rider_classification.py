# %% [markdown]
# # Divvy Bike Share: Rider Classification Using Machine Learning
# 
# **MSA 8010 – Data Programming**
# 
# ### Project Overview
# 
# This analysis investigates whether historical Divvy bike-sharing trip patterns can be used to classify trips as belonging to casual riders or annual members. Using a cleaned dataset containing approximately 15.8 million trips from 2024 through 2026, the goal is to identify behavioral patterns associated with rider membership type and evaluate how effectively machine-learning models can distinguish between the two groups.
# 
# The analysis begins with data exploration and feature engineering, followed by the development of Logistic Regression and Random Forest classification models. Several experiments are conducted to examine the effects of training sample size, classification thresholds, and model selection on predictive performance.
# 
# Data from 2024 is used for training, 2025 for validation, and 2026 for out-of-time testing. Model performance is assessed using accuracy, precision, recall, F1-score, and confusion matrices, with particular emphasis on correctly identifying casual-rider trips.
# 
# The findings aim to provide insight into the relationship between riding behavior and membership type while identifying potential applications and limitations of machine learning in supporting Divvy's membership-growth strategy.

# %%
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import pyarrow as pa
import sys

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
   accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.ensemble import RandomForestClassifier

# %%
print("Python:", sys.executable)
print("Pandas:", pd.__version__)
print("Pyarrow:", pa.__version__)

# %%
columns = [
    "member_casual",
    "rideable_type",
    "started_at",
    "ended_at"
]

# %%
df = pd.read_parquet(
    "data/divvy_clean_2024_2026.parquet", 
    columns=columns
)
df.shape


# %%
df.info(memory_usage="deep")

# %%
df["member_casual"].value_counts()

# %%
df["member_casual"].value_counts(normalize=True) * 100

# %%
df["year"] = df["started_at"].dt.year
df["year"].value_counts().sort_index()

# %% [markdown]
# # Feature Engineering

# %%
# Ensure timestamps are datetime objects
df["started_at"] = pd.to_datetime(df["started_at"])
df["ended_at"] = pd.to_datetime(df["ended_at"])

# Calculate trip duration in minutes
df["ride_length"] = (
    df["ended_at"] - df["started_at"]
).dt.total_seconds() / 60

# Extract time-related features
df["hour"] = df["started_at"].dt.hour
df["day_of_week"] = df["started_at"].dt.dayofweek
df["month"] = df["started_at"].dt.month

# Identify weekend trips
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Inspect engineered features
df[
    [
        "member_casual",
        "ride_length",
        "hour",
        "day_of_week",
        "month",
        "is_weekend"
    ]
].head(20)

# %%
print("Missing values:")
print(
    df[
        ["member_casual", "rideable_type", "ride_length",
         "hour", "day_of_week", "month"]
    ].isna().sum()
)

print("\nRide length statistics:")
print(df["ride_length"].describe())

print("\nNonpositive durations:")
print((df["ride_length"] <= 0).sum())

print("\nTrips longer than 24 hours:")
print((df["ride_length"] > 1440).sum())


# %%
df['member_casual'].value_counts()

# %%
df.loc[df['member_casual'] == 'casual', [
    'member_casual',
    'ride_length',
    'hour',
    'day_of_week',
    'month',
    'is_weekend'
]].head(10)
       

# %%
ride_comparison = (
    df.groupby("member_casual")["ride_length"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

display(ride_comparison)

# %%
ride_comparison["mean"].plot(
    kind="bar",
    figsize=(8, 5),
    color=["orange", "steelblue"]
)

plt.title("Average Ride Duration by Rider Type (2024–2026)")
plt.xlabel("Rider Type")
plt.ylabel("Average Ride Duration (Minutes)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# # How To Split The Data
# 2024 Training Data |
# 2025 Validation Data |
# 2026 Test Data 

# %%
# Select relevant features
features = [
    "ride_length",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "rideable_type"
]

# Encode the target variable
# casual = 1, member = 0
df["target"] = (
    df["member_casual"] == "casual"
).astype("int8")

# Create separate datasets by year
train_df = df.loc[df["year"] == 2024, features + ["target"]]
val_df = df.loc[df["year"] == 2025, features + ["target"]]
test_df = df.loc[df["year"] == 2026, features + ["target"]]

print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)

# %%
year_distribution = pd.crosstab(
    df["year"],
    df["member_casual"],
    normalize="index"
) * 100

display(year_distribution.round(2))

# %%
# Define sample sizes for our experiment
sample_sizes = [25000, 50000, 100000, 200000]

# Create reproducible nested training samples
train_pool = train_df.sample(
    n=max(sample_sizes),
    random_state=42
)

# Sort randomly so smaller samples are subsets of larger ones
train_pool = train_pool.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

training_samples = {}

for size in sample_sizes:
    training_samples[size] = train_pool.iloc[:size]

    print(
        f"Sample size: {size:,} | "
        f"Casual percentage: "
        f"{training_samples[size]['target'].mean() * 100:.2f}%"
    )

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

numeric_features = [
    "ride_length",
    "hour",
    "day_of_week",
    "month",
    "is_weekend"
]

categorical_features = ["rideable_type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# %%
# Fixed validation sample from 2025
val_sample = val_df.sample(
    n=100000,
    random_state=42
)

X_val = val_sample[features]
y_val = val_sample["target"]

results = []
models = {}

for size, sample in training_samples.items():

    print(f"\nTraining model with {size:,} observations...")

    X_train = sample[features]
    y_train = sample["target"]

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000)
            )
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    results.append({
        "Training Size": size,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1": f1_score(y_val, y_pred)
    })

    models[size] = model

results_df = pd.DataFrame(results)

display(results_df.round(4))

# %%
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["Training Size"],
    results_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["Training Size"],
    results_df["F1"],
    marker="s",
    label="F1-score"
)

plt.xlabel("Number of Training Observations")
plt.ylabel("Validation Score")
plt.title("Effect of Training Size on Logistic Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Increasing the training sample from 25,000 to 200,000 observations produced minimal improvement in Logistic Regression performance. Consequently, the analysis shifted toward examining classification errors, decision thresholds, and alternative algorithms.

# %% [markdown]
# # First Experiment: Confusion Matrix for Casual and Member Riders

# %%
# Select the best model based on F1-score
best_logistic = models[100000]

# Predict using the 2025 validation data
y_pred = best_logistic.predict(X_val)

# Calculate confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Member", "Casual"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression Confusion Matrix - 2025")
plt.tight_layout()
plt.show()

# Detailed classification metrics
print(
    classification_report(
        y_val,
        y_pred,
        target_names=["Member", "Casual"]
    )
)

# %% [markdown]
# This model isn't that effective at predicting casual riders. Let's try lowering the classification threshold to see if we can improve our recall for casual riders.

# %%
# Predicted probability of casual (class 1)
y_proba = best_logistic.predict_proba(X_val)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

threshold_results = []

for threshold in thresholds:

    predictions = (y_proba >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(y_val, predictions, zero_division=0),
        "Recall": recall_score(y_val, predictions, zero_division=0),
        "F1": f1_score(y_val, predictions, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)

display(threshold_df.round(4))

# %% [markdown]
# We are able to increase recall by 53.54% by lowering the classification threshold. The tradeoff is that precision dropped by 19.69%. So, lowering the threshold to 30% will allow us to classify more trips as casual. 

# %% [markdown]
# # Second Experiment: Optimizing the Classification Threshold

# %%
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = np.arange(0.15, 0.51, 0.01)

threshold_results = []

for threshold in thresholds:
    predictions = (y_proba >= threshold).astype(int)

    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

best_row = threshold_df.loc[threshold_df["F1"].idxmax()]

print("Best threshold:", best_row["Threshold"])
print("Best F1-score:", round(best_row["F1"], 4))

display(threshold_df.sort_values("F1", ascending=False).head(10))

# %% [markdown]
# Lowering the threshold to 0.27 proves to be the best approach since recall improved to 0.85! However, this comes at lower precision rate of 0.39. For logistic regression 0.27 is the best threshold. 

# %%
plt.figure(figsize=(9, 5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker=".",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker=".",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker=".",
    label="F1-score"
)

plt.axvline(
    x=0.27,
    color="red",
    linestyle="--",
    label="Best F1 threshold (0.27)"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Logistic Regression: Threshold Optimization")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# This graphs shows that as the threshold increases that recall decreases. The tradeoff here is with precision. As we increase the the threshold classification then precision increases while recall decreases significantly. 

# %% [markdown]
# # Third Experiment: Random Forest

# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Use the same 100,000-record training sample
sample = training_samples[100000]

X_train = sample[features]
y_train = sample["target"]

# Create Random Forest pipeline
rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the same 2025 validation sample
rf_pred = rf_model.predict(X_val)

print("Random Forest Results")
print("Accuracy:", round(accuracy_score(y_val, rf_pred), 4))
print("Precision:", round(precision_score(y_val, rf_pred), 4))
print("Recall:", round(recall_score(y_val, rf_pred), 4))
print("F1:", round(f1_score(y_val, rf_pred), 4))

# %% [markdown]
# The random forest actually performed better at the initial threshold of 50 compared to logistic regression. Let's modify the threshold to see if we can improve our results for random forest. 

# %% [markdown]
# # Fourth Experiment: Modify Threshold For Random Forest

# %%
# Get Random Forest predicted probabilities
rf_proba = rf_model.predict_proba(X_val)[:, 1]

# Evaluate different thresholds
rf_threshold_results = []

for threshold in np.arange(0.15, 0.51, 0.01):

    rf_predictions = (rf_proba >= threshold).astype(int)

    rf_threshold_results.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(y_val, rf_predictions),
        "Precision": precision_score(y_val, rf_predictions, zero_division=0),
        "Recall": recall_score(y_val, rf_predictions, zero_division=0),
        "F1": f1_score(y_val, rf_predictions, zero_division=0)
    })

rf_threshold_df = pd.DataFrame(rf_threshold_results)

# Find best threshold based on F1
best_rf_row = rf_threshold_df.loc[
    rf_threshold_df["F1"].idxmax()
]

display(
    rf_threshold_df
    .sort_values("F1", ascending=False)
    .head(10)
)

print("Best RF threshold:", best_rf_row["Threshold"])
print("Best RF F1:", round(best_rf_row["F1"], 4))

# %% [markdown]
# The random forest model has outperformed the tuned logistic regression model on the 2025 validation data. Moving forward for the 2026 analysis I will use Random Forest at 0.30 threshold. 

# %%
# Select a reproducible 2026 test sample
test_sample = test_df.sample(
    n=100000,
    random_state=42
)

X_test = test_sample[features]
y_test = test_sample["target"]

# Keep our selected 2025 validation threshold fixed
final_threshold = 0.30

# Predict casual rider probabilities
test_proba = rf_model.predict_proba(X_test)[:, 1]

# Convert probabilities to classifications
y_pred_2026 = (test_proba >= final_threshold).astype(int)

# Calculate performance metrics
print("2026 Test Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred_2026), 4))
print("Precision:", round(precision_score(y_test, y_pred_2026), 4))
print("Recall:", round(recall_score(y_test, y_pred_2026), 4))
print("F1:", round(f1_score(y_test, y_pred_2026), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_2026,
        target_names=["Member", "Casual"]
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_2026)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Member", "Casual"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Random Forest: 2026 Test Confusion Matrix")
plt.tight_layout()
plt.show()

# %%
# Actual rider distribution
actual = pd.Series(y_test.to_numpy()).value_counts(
    normalize=True
) * 100

# Predicted rider distribution
predicted = pd.Series(y_pred_2026).value_counts(
    normalize=True
) * 100

# Build comparison table
comparison = pd.DataFrame({
    "Actual": actual,
    "Predicted": predicted
})

comparison.index = comparison.index.map({
    0: "Member",
    1: "Casual"
})

comparison = comparison.reindex(["Member", "Casual"])

display(comparison.round(2))

# Plot comparison
comparison.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("2026 Divvy Trips: Actual vs Predicted Rider Type")
plt.xlabel("Rider Classification")
plt.ylabel("Percentage of Trips")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

# %%
model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Threshold": [0.27, 0.30],
    "Validation F1": [0.5387, 0.5612],
    "Validation Precision": [0.3939, 0.4354],
    "Validation Recall": [0.8519, 0.7893]
})

display(model_comparison)

# %% [markdown]
# # Model Evaluation and Interpretation
# 
# The Random Forest model was selected based on its performance on the 2025 validation dataset, achieving an F1-score of 0.5612 for casual-rider classification at a decision threshold of 0.30.
# 
# When evaluated on 100,000 previously unseen observations from 2026, the model achieved an F1-score of 0.5719, precision of 0.4336, and recall of 0.8398 for casual riders.
# 
# Although the model identified approximately 84% of actual casual trips, its precision remained relatively low. The model predicted 71.14% of trips as casual, whereas only 36.73% were actually casual. This resulted in 40,300 member trips being incorrectly classified as casual.
# 
# These findings demonstrate a tradeoff between identifying casual trips and minimizing false positives. Lowering the classification threshold improved recall but reduced overall accuracy and resulted in substantial overprediction of the casual category.
# 
# The results suggest that riding duration, temporal patterns, and bicycle type provide some information about membership classification, but the selected model is not sufficiently reliable for direct operational deployment.
# 
# Future improvements could include additional behavioral features, alternative algorithms, probability calibration, and evaluation of different classification thresholds based on practical marketing objectives.

