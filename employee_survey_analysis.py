# ================================================================
# Streamlit Employee Survey Dashboard - Updated (Minimal Changes)
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

st.set_page_config(page_title="Homes First Survey Dashboard", layout="wide")
st.title("Homes First Employee Survey Dashboard")

# 1️⃣ Upload dataset
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Dataset shape:", df.shape)

    # -----------------------------
    # Key columns
    # -----------------------------
    role_col = [c for c in df.columns if "role" in c.lower() or "department" in c.lower()][0]
    age_col = [c for c in df.columns if "age" in c.lower()][0]
    gender_col = [c for c in df.columns if "gender" in c.lower()][0]
    recommend_col = [c for c in df.columns if "recommend" in c.lower()][0]
    years_col = [c for c in df.columns if "years" in c.lower() and "employed" in c.lower()][0]
    recognized_col = [c for c in df.columns if "recognized" in c.lower() and "acknowledged" in c.lower()][0]
    growth_col = [c for c in df.columns if "potential for growth" in c.lower()][0]
    impact_col = [c for c in df.columns if "positive impact" in c.lower()][0]
    training_pref_col = [c for c in df.columns if "live virtual training" in c.lower()][0]
    fulfillment_col = [c for c in df.columns if "fulfilling and rewarding" in c.lower()][0]
    disability_col = [c for c in df.columns if "disability" in c.lower()][0]

    # -----------------------------
    # Convert key columns to str
    # -----------------------------
    for col in [recommend_col, recognized_col, growth_col, impact_col, training_pref_col, fulfillment_col, disability_col]:
        df[col] = df[col].astype(str)

    # -----------------------------
    # Filters
    # -----------------------------
    st.sidebar.header("Filter Employees")
    role_filter = st.sidebar.multiselect("Role/Department", df[role_col].unique(), default=df[role_col].unique())
    age_filter = st.sidebar.multiselect("Age Group", df[age_col].unique(), default=df[age_col].unique())
    gender_filter = st.sidebar.multiselect("Gender", df[gender_col].unique(), default=df[gender_col].unique())
    df_filtered = df[df[role_col].isin(role_filter) & df[age_col].isin(age_filter) & df[gender_col].isin(gender_filter)]
    st.write(f"Filtered dataset: {df_filtered.shape[0]} respondents")

    # -----------------------------
    # KPIs
    # -----------------------------
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    total = df_filtered.shape[0]

    positive_rec = df_filtered[df_filtered[recommend_col].str.lower().str.contains("likely|yes", na=False)].shape[0]
    col1.metric("Recommend Homes First", f"{positive_rec}/{total} ({positive_rec/total*100:.1f}%)")

    recognized = df_filtered[df_filtered[recognized_col].str.lower().str.contains("yes|somewhat", na=False)].shape[0]
    col2.metric("Feel Recognized", f"{recognized}/{total} ({recognized/total*100:.1f}%)")

    growth = df_filtered[df_filtered[growth_col].str.lower().str.contains("yes", na=False)].shape[0]
    col3.metric("See Potential for Growth", f"{growth}/{total} ({growth/total*100:.1f}%)")

    impact = df_filtered[df_filtered[impact_col].str.lower().str.contains("positive impact", na=False)].shape[0]
    col4.metric("Feel Positive Impact", f"{impact}/{total} ({impact/total*100:.1f}%)")

    # -----------------------------
    # Helper for counts inside bars
    # -----------------------------
    def add_counts(ax, vertical=True):
        for p in ax.patches:
            if vertical:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}', 
                                (p.get_x() + p.get_width() / 2., height/2),
                                ha='center', va='center', color='white', fontsize=10)
            else:
                width = p.get_width()
                if width > 0:
                    ax.annotate(f'{int(width)}',
                                (width/2, p.get_y() + p.get_height()/2),
                                ha='center', va='center', color='white', fontsize=10)

    # -----------------------------
    # Demographics
    # -----------------------------
    st.header("Demographics")
    col1, col2, col3 = st.columns(3)

    # Role
    fig, ax = plt.subplots()
    order = df_filtered[role_col].value_counts().index
    sns.countplot(y=df_filtered[role_col], order=order, palette='viridis', ax=ax)
    ax.set_title("Respondents by Role/Department")
    col1.pyplot(fig)

    # Age
    fig, ax = plt.subplots()
    order = df_filtered[age_col].value_counts().index
    sns.countplot(x=df_filtered[age_col], order=order, palette='magma', ax=ax)
    ax.set_title("Respondents by Age Group")
    plt.xticks(rotation=0)
    add_counts(ax)
    col2.pyplot(fig)

    # Gender
    fig, ax = plt.subplots()
    order = df_filtered[gender_col].value_counts().index
    sns.countplot(x=df_filtered[gender_col], order=order, palette='coolwarm', ax=ax)
    ax.set_title("Respondents by Gender")
    plt.xticks(rotation=0)
    add_counts(ax)
    col3.pyplot(fig)

    # -----------------------------
    # Job Fulfillment
    # -----------------------------
    st.header("Job Fulfillment")
    fig, ax = plt.subplots()
    order = df_filtered[fulfillment_col].value_counts().index
    sns.countplot(x=df_filtered[fulfillment_col], order=order, palette='plasma', ax=ax)
    ax.set_title("Job Fulfillment")
    plt.xticks(rotation=0)
    add_counts(ax)
    st.pyplot(fig)

    # -----------------------------
    # Training Preferences
    # -----------------------------
    st.header("Training Preferences")
    fig, ax = plt.subplots()
    order = df_filtered[training_pref_col].value_counts().index
    sns.countplot(x=df_filtered[training_pref_col], order=order, palette='Set2', ax=ax)
    ax.set_title("Training Mode Preference")
    plt.xticks(rotation=0)
    add_counts(ax)
    st.pyplot(fig)

    # -----------------------------
    # Disability Analysis (Stacked Percentage)
    # -----------------------------
    st.header("Disability Analysis")

    def plot_stacked_pct(df, category, hue, palette):
        cross = pd.crosstab(df[category], df[hue], normalize='index')*100
        cross.plot(kind='bar', stacked=True, color=palette, figsize=(10,6))
        plt.ylabel("Percentage")
        plt.xticks(rotation=0)
        plt.legend(title=hue, bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

    st.subheader("Disability by Age Group")
    plot_stacked_pct(df_filtered, age_col, disability_col, palette=sns.color_palette("Set1"))

    st.subheader("Disability by Gender")
    plot_stacked_pct(df_filtered, gender_col, disability_col, palette=sns.color_palette("Set2"))

    # -----------------------------
    # Cross Analysis
    # -----------------------------
    st.header("Cross Analysis")
    st.subheader("Recommendation by Role")
    fig, ax = plt.subplots()
    sns.countplot(x=df_filtered[recommend_col], hue=df_filtered[role_col], data=df_filtered, palette='Set2', ax=ax)
    ax.set_title("Recommendation by Role/Department")
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    st.pyplot(fig)

    st.subheader("Recommendation by Age Group")
    fig, ax = plt.subplots()
    sns.countplot(x=df_filtered[recommend_col], hue=df_filtered[age_col], data=df_filtered, palette='Set3', ax=ax)
    ax.set_title("Recommendation by Age Group")
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    st.pyplot(fig)
