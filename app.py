import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support
)

# Streamlit settings
st.set_page_config(layout="wide")
st.title("🎓 Student Performance & Graduation Analysis App")
st.markdown("Analyze student survey data and predict academic performance using KNN.")

# Upload section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # GPA preprocessing
    df["GPA_2023"] = pd.to_numeric(df["Your 2023 academic year average/GPA in % (Ignore if you are 2024 1st year student)"], errors="coerce")

    def classify_performance(score):
        if score < 60:
            return "low"
        elif score < 75:
            return "medium"
        else:
            return "high"

    df["performance level"] = df["GPA_2023"].apply(classify_performance)

    # Encode categorical features
    le = LabelEncoder()
    categorical_cols = [
        "Your Sex?",
        "What faculty does your degree fall under?",
        "Were you on scholarship/bursary in 2023?",
        "How often do you go out partying/socialising during the week? ",
        "On a night out, how many alcoholic drinks do you consume?",
    ]
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Train/test split
    X = df[categorical_cols]
    y = df["performance level"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # KNN classification
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Display metrics
    st.subheader("📊 KNN Model Results")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2%}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    st.subheader("🔷 Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    st.pyplot(fig)

    # Plot correlation heatmap
    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Graduation status
    df["Matric_and_2023_avg"] = df[[
        "Your Matric (grade 12) Average/ GPA (in %)",
        "Your 2023 academic year average/GPA in % (Ignore if you are 2024 1st year student)"
    ]].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    def graduation_status(score): return "pass" if score >= 70 else "fail"
    df["graduation status"] = df["Matric_and_2023_avg"].apply(graduation_status)

    # Pie chart of graduation
    st.subheader("🎓 Graduation Status")
    fig, ax = plt.subplots()
    df["graduation status"].value_counts().plot.pie(autopct='%1.1f%%', colors=["#66b3ff", "#ff9999"], ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

    # Performance bar chart
    st.subheader("📊 Performance Level Distribution")
    fig, ax = plt.subplots()
    df["performance level"].value_counts().plot.bar(color=["#ff9999", "#66b3ff", "#99ff99"], ax=ax)
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)

    # Feature impact on graduation/performance
    st.subheader("📚 Behavioral Factors per Group")
    cols_to_numeric = [
        "Additional amount of studying (in hrs) per week",
        "How often do you go out partying/socialising during the week? ",
        "On a night out, how many alcoholic drinks do you consume?"
    ]
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    st.markdown("**Average behavior by Graduation Status:**")
    st.bar_chart(df.groupby("graduation status")[cols_to_numeric].mean())

    st.markdown("**Average behavior by Performance Level:**")
    st.bar_chart(df.groupby("performance level")[cols_to_numeric].mean())

else:
    st.info("Please upload a dataset to begin.")
