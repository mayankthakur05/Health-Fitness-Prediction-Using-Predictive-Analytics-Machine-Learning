import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Health_Fitness_Expanded_5000_CORRECT.csv")


# Clean column names
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)
print("\n")
print(df.head())

# Check missing values
print(df.isnull().sum())

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n")
print("Missing values handled successfully!")

print("\n")

df["BMI"] = df["Weight (kg)"] / ((df["Height (cm)"] / 100) ** 2)


le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Convert multi-class health into binary health
df["Binary_Health"] = df['How would you rate your overall health condition?'].apply(lambda x: 1 if x >= 2 else 0)


#Modeling

X = df.drop(
    [
        "Binary_Health",
        "How would you rate your overall health condition?"
    ],
    axis=1
)
y = df["Binary_Health"]

feature_columns = X.columns.tolist()

# Clean infinite and NaN values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)


def train_logistic_regression():
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("Accuracy:", acc)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix – Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return acc


def train_knn():
    print("\nTraining KNN...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("Accuracy:", acc)
    sns.heatmap(cm, annot=True, fmt="d",cmap="Blues")
    plt.title("Confusion Matrix – KNN")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return acc


def train_decision_tree():
    print("\nTraining Decision Tree (Readable Version)...")

    # PRUNED Decision Tree
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,        
        min_samples_split=50, 
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
    plt.title("Confusion Matrix – Decision Tree")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    plt.figure(figsize=(16,8))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["Not Healthy", "Healthy"],
        filled=True,
        rounded=True,
        fontsize=12
    )
    plt.title("Decision Tree (Health Prediction)", fontsize=14)
    plt.show()

    return acc


def train_random_forest():
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("Accuracy:", acc)
    sns.heatmap(cm, annot=True, fmt="d",cmap="Blues")
    plt.title("Confusion Matrix – Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return acc

def find_best_model():
    print("\nEvaluating all models to find best performer...")

    accs = {
        "Logistic Regression": train_logistic_regression(),
        "KNN": train_knn(),
        "Decision Tree": train_decision_tree(),
        "Random Forest": train_random_forest()
    }

    # Find best model
    best_model = max(accs, key=accs.get)

    print("\n🏆 Best Performing Model:", best_model)
    print("Best Accuracy:", accs[best_model])

    # ---- MODEL COMPARISON CHART ----
    model_names = list(accs.keys())
    accuracies = list(accs.values())

    plt.figure()
    plt.bar(model_names, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()



def show_health_distribution():
    plt.figure()
    health_map = {
        0: "Very Healthy",
        1: "Healthy",
        2: "Average",
        3: "Unhealthy"
    }

    counts = df['How would you rate your overall health condition?'].value_counts().sort_index()
    counts.index = counts.index.map(health_map)
    counts.plot(kind="bar")
    plt.title("Distribution of Overall Health Condition")
    plt.xlabel("Health Category")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.show()


def show_stress_vs_health():
    stress_col = [c for c in df.columns if "stress" in c.lower()][0]
    plt.figure(figsize=(6,4))
    sns.violinplot(x=df['How would you rate your overall health condition?'], y=df[stress_col])
    plt.title("Stress Level vs Health Condition")
    plt.xlabel("Health Condition")
    plt.ylabel("Stress Level")
    plt.show()


def show_correlation():
     important_cols = [
        "Age",
        "BMI",
        "Stress(1-5)",
        "Average Sleep Hours per Day",
        "Smoking Habits",
        "Alcohol Consumption",
        "Weekly Exercise Duration (hours)",
        "Binary_Health"
    ]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df[important_cols].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Key Feature Correlation with Health", fontsize=14)
    plt.tight_layout()
    plt.show()


def predict_health_menu():
    print("\n--- Predict Your Health & Fitness ---")

    age = int(input("Enter Age (years): "))
    height = float(input("Enter Height (cm): "))
    weight = float(input("Enter Weight (kg): "))
    bmi = weight / ((height / 100) ** 2)

    sleep = float(input("Average Sleep Hours per Day: "))
    stress = float(input("Stress Level (1–10): "))

    alcohol = int(input("Alcohol Consumption (1 = Yes, 0 = No): "))
    smoking = int(input("Smoking (1 = Yes, 0 = No): "))
    exercise = int(input("Regular Exercise (1 = Yes, 0 = No): "))

    # 🔹 Create empty input with all features
    input_data = {col: 0 for col in feature_columns}

    # 🔹 Map user inputs to correct training columns
    input_data["Age"] = age
    input_data["Height (cm)"] = height
    input_data["Weight (kg)"] = weight
    input_data["BMI"] = bmi
    input_data["Average Sleep Hours per Day"] = sleep
    input_data["Stress Level"] = stress
    input_data["Alcohol Consumption"] = alcohol
    input_data["Smoking"] = smoking
    input_data["Regular Exercise"] = exercise

    # 🔹 Create DataFrame with exact training structure
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # 🔹 Scale & predict
    input_scaled = scaler.transform(input_df)
    prob = rf_model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob >= 0.30 else 0

    print(f"\n🧮 BMI: {round(bmi,2)}")
    print(f"Health Probability: {prob*100:.2f}%")

    if prediction == 1:
        print("✅ Prediction: HEALTHY")
    else:
        print("⚠️ Prediction: NOT HEALTHY")


while True:
    print("\n========== MAIN MENU ==========")
    print("1. Show EDA Graphs")
    print("2. Train & Evaluate a Model")
    print("3. Find Best Performing Model")
    print("4. Predict Your Health & Fitness")
    print("5. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        print("\n--- EDA MENU ---")
        print("1. Health Condition Distribution")
        print("2. Stress vs Health Condition")
        print("3. Correlation Heatmap")

        eda_choice = int(input("Choose graph: "))

        if eda_choice == 1:
            show_health_distribution()
        elif eda_choice == 2:
            show_stress_vs_health()
        elif eda_choice == 3:
            show_correlation()
        else:
            print("Invalid choice")

    elif choice == 2:
        print("\n--- MODEL MENU ---")
        print("1. Logistic Regression")
        print("2. KNN")
        print("3. Decision Tree")
        print("4. Random Forest")

        model_choice = int(input("Choose model: "))

        if model_choice == 1:
            train_logistic_regression()
        elif model_choice == 2:
            train_knn()
        elif model_choice == 3:
            train_decision_tree()
        elif model_choice == 4:
            train_random_forest()
        else:
            print("Invalid model choice")

    elif choice == 3:
        find_best_model()

    elif choice == 4:
        predict_health_menu()

    elif choice == 5:
        print("Exiting program. Thank you!")
        break










