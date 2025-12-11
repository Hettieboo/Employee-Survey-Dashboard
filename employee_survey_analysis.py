import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

st.set_page_config(page_title="Homes First Survey Analysis", layout="wide")
st.title("Homes First Employee Survey Dashboard")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Dataset shape:", df.shape)
else:
    st.stop()

# -----------------------------
# Key columns
# -----------------------------
role_col = [c for c in df.columns if "role" in c.lower() or "department" in c.lower()][0]
age_col = [c for c in df.columns if "age" in c.lower()][0]
gender_col = [c for c in df.columns if "gender" in c.lower()][0]
recommend_col = [c for c in df.columns if "recommend" in c.lower()][0]
fulfillment_col = [c for c in df.columns if "fulfilling" in c.lower()][0]
training_pref_col = [c for c in df.columns if "live virtual training" in c.lower()][0]
disability_col = [c for c in df.columns if "disability" in c.lower() and "identify" in c.lower()][0]

# -----------------------------
# Filters
# -----------------------------
with st.sidebar:
    st.header("Filter Employees")
    role_filter = st.multiselect("Role/Department", df[role_col].dropna().unique(), default=df[role_col].dropna().unique())
    age_filter = st.multiselect("Age Group", df[age_col].dropna().unique(), default=df[age_col].dropna().unique())
    gender_filter = st.multiselect("Gender", df[gender_col].dropna().unique(), default=df[gender_col].dropna().unique())

df_filtered = df[df[role_col].isin(role_filter) & df[age_col].isin(age_filter) & df[gender_col].isin(gender_filter)]
st.write(f"Filtered dataset: {df_filtered.shape[0]} respondents")

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Respondents", f"{df_filtered.shape[0]}")
col2.metric("Avg Years Employed", f"{df_filtered['How many years have you been employed by Homes First?'].dropna().mean():.1f}")

# Job Fulfillment average: convert to numeric code safely
fulf_codes = pd.Categorical(df_filtered[fulfillment_col]).codes
col3.metric("Avg Job Fulfillment", f"{fulf_codes.mean():.1f}")

# Recommend Homes First: numeric out of 10
avg_recommend = df_filtered[recommend_col].mean()
median_recommend = df_filtered[recommend_col].median()
col4.metric("Avg Recommend Homes First", f"{avg_recommend:.1f}/10")
st.caption(f"Median Recommendation Score: {median_recommend:.0f}/10")

# -----------------------------
# Helper function for single countplot labels
# -----------------------------
def add_labels(ax, labels=None):
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        # Count on top
        ax.text(p.get_x() + p.get_width()/2., height + 0.1, int(height), ha="center", va="bottom", fontsize=12, color="black")
        # Answer label inside
        if labels is not None and i < len(labels):
            ax.text(p.get_x() + p.get_width()/2., height/2, labels[i], ha="center", va="center", fontsize=12, color="white", wrap=True)

# -----------------------------
# Job Fulfillment Chart
# -----------------------------
fig, ax = plt.subplots(figsize=(14,10))
order = df_filtered[fulfillment_col].value_counts().index
sns.countplot(
    x=df_filtered[fulfillment_col],
    order=order,
    palette=sns.color_palette("plasma", len(order)).as_hex(),
    ax=ax
)
ax.set_title("How fulfilling and rewarding do you find your work?", fontsize=16, pad=60)
ax.set_xticklabels([])  # remove bottom labels
add_labels(ax, labels=order)
plt.subplots_adjust(top=0.88)
st.pyplot(fig)

# -----------------------------
# Training Preferences Chart
# -----------------------------
fig, ax = plt.subplots(figsize=(14,10))
order = df_filtered[training_pref_col].value_counts().index
sns.countplot(
    x=df_filtered[training_pref_col],
    order=order,
    palette=sns.color_palette("muted", len(order)).as_hex(),
    ax=ax
)
ax.set_title("How do you feel about live virtual training (Zoom/Teams) vs in-person?", fontsize=16, pad=60)
ax.set_xticklabels([])
add_labels(ax, labels=order)
plt.subplots_adjust(top=0.88)
st.pyplot(fig)

# -----------------------------
# Disabilities by Age Chart
# -----------------------------
df_filtered[disability_col] = df_filtered[disability_col].fillna("No Disability")
fig, ax = plt.subplots(figsize=(14,8))
order = df_filtered[age_col].value_counts().index
sns.countplot(
    x=df_filtered[age_col],
    hue=df_filtered[disability_col],
    palette=sns.color_palette("dark", len(df_filtered[disability_col].unique())).as_hex(),
    order=order,
    ax=ax
)
ax.set_title("Do you identify as an individual living with a disability? - by Age Group", fontsize=16, pad=60)
ax.set_xticklabels([])
plt.ylabel("Number of Employees")
plt.xlabel("Age Group")
plt.legend(title="Disability Status", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.subplots_adjust(top=0.88)
st.pyplot(fig)

# -----------------------------
# Disabilities by Gender Chart
# -----------------------------
fig, ax = plt.subplots(figsize=(14,8))
order = df_filtered[gender_col].value_counts().index
sns.countplot(
    x=df_filtered[gender_col],
    hue=df_filtered[disability_col],
    palette=sns.color_palette("deep", len(df_filtered[disability_col].unique())).as_hex(),
    order=order,
    ax=ax
)
ax.set_title("Do you identify as an individual living with a disability? - by Gender", fontsize=16, pad=60)
ax.set_xticklabels([])
plt.ylabel("Number of Employees")
plt.xlabel("Gender")
plt.legend(title="Disability Status", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.subplots_adjust(top=0.88)
st.pyplot(fig)
