import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8,5)

st.set_page_config(page_title="Homes First Survey Analysis", layout="wide")
st.title("Homes First Employee Survey Dashboard")
st.markdown("Interactive survey insights with highlights and key visualizations.")

# -----------------------------
# Upload the Excel file
# -----------------------------
uploaded_file = st.file_uploader("Upload your survey Excel file", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # -----------------------------
    # Map Columns
    # -----------------------------
    role_col = "Select the role/department that best describes your current position at Homes First."
    age_col = "Please select the option that includes your current age."
    gender_col = "Which of the following best describes your gender identity?"
    disability_col = "Do you identify as an individual living with a disability/disabilities and if so, what type of disability/disabilities do you have? (Select all that apply.)"
    race_col = "Which racial or ethnic identity/identities best reflect you. (Select all that apply.)"
    comment_cols = [c for c in df.columns if "comment" in c.lower()]
    recommend_col = "How likely are you to recommend Homes First as a good place to work?"

    # -----------------------------
    # Define Themes
    # -----------------------------
    THEMES = {
        "Workload": ["understaffed", "workload", "busy", "overwhelming", "paperwork", "time", "tasks"],
        "Support": ["support", "help", "assist", "guidance", "resources", "tools"],
        "Team": ["team", "colleagues", "coworkers", "collaboration", "together", "staff"],
        "Training": ["training", "development", "learning", "skills", "education", "growth"],
        "Clients": ["client", "resident", "people", "helping", "care", "community"],
        "Management": ["management", "leadership", "supervisor", "manager", "direction"],
        "Recognition": ["recognition", "appreciation", "valued", "acknowledged", "feedback"],
        "Work-Life Balance": ["balance", "flexibility", "schedule", "hours", "time off"]
    }

    # -----------------------------
    # Calculate Theme Mentions
    # -----------------------------
    theme_counts = {theme: 0 for theme in THEMES}
    for col in comment_cols:
        for text in df[col].dropna().astype(str):
            t = text.lower()
            for theme, keywords in THEMES.items():
                if any(kw in t for kw in keywords):
                    theme_counts[theme] += 1
    theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Mentions']).sort_values('Mentions', ascending=False)

    # -----------------------------
    # Highlights Metrics
    # -----------------------------
    st.subheader("Highlights")
    col1, col2 = st.columns(2)
    col1.metric("Total Responses", df.shape[0])
    col2.metric("Areas of Concern (Theme Mentions)", sum(theme_counts.values()))

    # -----------------------------
    # Demographics Visuals
    # -----------------------------
    st.subheader("Role / Department Distribution")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3*len(df[role_col].unique()))))
    sns.countplot(y=df[role_col], order=df[role_col].value_counts().index, palette='viridis', ax=ax)
    ax.set_xlabel("Number of Respondents")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Age Distribution")
    age_counts = df[age_col].value_counts()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("Gender Distribution")
    gender_counts = df[gender_col].value_counts()
    fig, ax = plt.subplots(figsize=(6,6))
    wedges, texts, autotexts = ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("Disability Status")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=df[disability_col], order=df[disability_col].value_counts().index, palette='coolwarm', ax=ax)
    ax.set_ylabel("Number of Respondents")
    ax.set_xlabel("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Race / Ethnic Identity")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5*len(df[race_col].unique()))))
    sns.countplot(y=df[race_col], order=df[race_col].value_counts().index, palette='viridis', ax=ax)
    ax.set_xlabel("Number of Respondents")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------------
    # Key Themes Bar Chart (Counts Inside)
    # -----------------------------
    st.subheader("Key Themes in Survey Responses (Top 10)")
    top_themes = theme_df.head(10)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5*len(top_themes))))
    sns.barplot(x='Mentions', y='Theme', data=top_themes, palette='RdYlGn', ax=ax)
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.5, p.get_y() + p.get_height()/2, int(width), va='center', fontsize=10)
    ax.set_xlabel("Mentions")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------------
    # Top 10 Words from Comments
    # -----------------------------
    st.subheader("Top 10 Frequent Words in Comments")
    STOP_WORDS = {'and','the','to','of','a','i','my','in','for','on','it','is','with','as','we','be'}
    all_text = " ".join(df[col].dropna().astype(str).sum() for col in comment_cols)
    words = [w.lower().strip('.,!?;:()[]{}') for w in all_text.split()
             if w.lower() not in STOP_WORDS and len(w)>3]
    word_freq = Counter(words)
    word_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values('Frequency', ascending=False)
    top_words = word_df.head(10)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5*len(top_words))))
    sns.barplot(x='Frequency', y='Word', data=top_words, palette='coolwarm', ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------------
    # Cross-analysis: Top Themes by Gender
    # -----------------------------
    st.subheader("Top Themes by Gender")
    theme_gender = {theme: {} for theme in THEMES}
    for col in comment_cols:
        for text, gender in zip(df[col].dropna().astype(str), df[gender_col].dropna()):
            t = text.lower()
            for theme, keywords in THEMES.items():
                if any(kw in t for kw in keywords):
                    if gender not in theme_gender[theme]:
                        theme_gender[theme][gender] = 0
                    theme_gender[theme][gender] += 1

    plot_df = pd.DataFrame(theme_gender).fillna(0).T
    plot_df = plot_df[top_themes['Theme']]  # top 10 themes only
    plot_df.plot(kind='bar', stacked=True, figsize=(10,6), colormap='tab20')
    plt.ylabel("Number of Mentions")
    plt.title("Top Themes by Gender")
    plt.tight_layout()
    st.pyplot(plt.gcf())

else:
    st.info("Please upload your Homes First survey Excel file to start the analysis.")
