
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Job Impact Analyzer",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.metric-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 AI Job Impact Analyzer - 2030</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Job Automation Risk Analysis</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

    return df


df = load_data()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🔎 Job Analysis")

st.sidebar.write(
    "Select a job to analyze its AI exposure and automation risk."
)

job_list = sorted(df["Job_Title"].dropna().unique())

selected_job = st.sidebar.selectbox(
    "Select Job Title",
    job_list
)


# --------------------------------------------------
# FILTER SELECTED JOB
# --------------------------------------------------

job_data = df[df["Job_Title"] == selected_job]


# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------

total_jobs = len(df)

high_risk = len(
    df[df["Risk_Category"].str.lower() == "high"]
)

medium_risk = len(
    df[df["Risk_Category"].str.lower() == "medium"]
)

low_risk = len(
    df[df["Risk_Category"].str.lower() == "low"]
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Jobs",
        total_jobs
    )

with col2:

    st.metric(
        "High Risk",
        high_risk
    )

with col3:

    st.metric(
        "Medium Risk",
        medium_risk
    )

with col4:

    st.metric(
        "Low Risk",
        low_risk
    )


st.divider()


# --------------------------------------------------
# SELECTED JOB INFORMATION
# --------------------------------------------------

st.header("📌 Selected Job Analysis")

if len(job_data) > 0:

    row = job_data.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:

        salary = row["Average_Salary"]

        st.metric(
            "Average Salary",
            f"${salary:,.0f}"
        )

    with col2:

        experience = row["Years_Experience"]

        st.metric(
            "Years of Experience",
            f"{experience:.1f}"
        )

    with col3:

        risk = row["Risk_Category"]

        st.metric(
            "Risk Category",
            risk
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        ai_exposure = row["AI_Exposure_Index"]

        st.metric(
            "AI Exposure Index",
            f"{ai_exposure:.2f}"
        )

    with col2:

        automation = row["Automation_Probability_2030"]

        st.metric(
            "Automation Probability",
            f"{automation:.2%}"
        )

    with col3:

        growth = row["Tech_Growth_Factor"]

        st.metric(
            "Technology Growth Factor",
            f"{growth:.2f}"
        )


st.divider()


# --------------------------------------------------
# RISK DISTRIBUTION
# --------------------------------------------------

st.header("📊 Automation Risk Distribution")

risk_counts = df["Risk_Category"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))

risk_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Job Automation Risk Distribution")

ax.set_xlabel("Risk Category")

ax.set_ylabel("Number of Jobs")

plt.xticks(rotation=0)

st.pyplot(fig)


# --------------------------------------------------
# AUTOMATION PROBABILITY BY RISK
# --------------------------------------------------

st.header("📈 Automation Probability by Risk Category")

risk_probability = (
    df.groupby("Risk_Category")["Automation_Probability_2030"]
    .mean()
    .sort_values()
)

fig, ax = plt.subplots(figsize=(8, 5))

risk_probability.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Average Automation Probability by Risk Category"
)

ax.set_xlabel("Risk Category")

ax.set_ylabel("Average Automation Probability")

plt.xticks(rotation=0)

st.pyplot(fig)


# --------------------------------------------------
# AI EXPOSURE ANALYSIS
# --------------------------------------------------

st.header("🤖 AI Exposure Analysis")

ai_exposure_data = (
    df.groupby("Risk_Category")["AI_Exposure_Index"]
    .mean()
    .sort_values()
)

fig, ax = plt.subplots(figsize=(8, 5))

ai_exposure_data.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Average AI Exposure by Risk Category"
)

ax.set_xlabel("Risk Category")

ax.set_ylabel("AI Exposure Index")

plt.xticks(rotation=0)

st.pyplot(fig)


# --------------------------------------------------
# SALARY VS AUTOMATION
# --------------------------------------------------

st.header("💰 Salary vs Automation Probability")

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    df["Average_Salary"],
    df["Automation_Probability_2030"],
    alpha=0.5
)

ax.set_title(
    "Salary vs Automation Probability"
)

ax.set_xlabel("Average Salary")

ax.set_ylabel("Automation Probability")

st.pyplot(fig)


# --------------------------------------------------
# JOB TABLE
# --------------------------------------------------

st.header("📋 Selected Job Details")

if len(job_data) > 0:

    display_columns = [
        "Job_Title",
        "Average_Salary",
        "Years_Experience",
        "Education_Level",
        "AI_Exposure_Index",
        "Tech_Growth_Factor",
        "Automation_Probability_2030",
        "Risk_Category"
    ]

    available_columns = [
        col for col in display_columns
        if col in job_data.columns
    ]

    st.dataframe(
        job_data[available_columns],
        use_container_width=True
    )


# --------------------------------------------------
# SKILLS ANALYSIS
# --------------------------------------------------

st.header("🛠️ Job Skill Profile")

skill_columns = [
    col for col in df.columns
    if col.lower().startswith("skill")
]

if skill_columns:

    skill_values = job_data[skill_columns].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    skill_values.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        f"Skill Profile - {selected_job}"
    )

    ax.set_xlabel("Skills")

    ax.set_ylabel("Skill Level")

    plt.xticks(rotation=45)

    st.pyplot(fig)


# --------------------------------------------------
# SIMPLE RISK INTERPRETATION
# --------------------------------------------------

st.header("💡 Risk Interpretation")

if len(job_data) > 0:

    risk = str(
        job_data["Risk_Category"].iloc[0]
    ).lower()

    automation = job_data[
        "Automation_Probability_2030"
    ].iloc[0]

    if risk == "high":

        st.warning(
            f"""
            **Risk Category: High**

            The dataset classifies **{selected_job}**
            as High automation risk.

            Automation probability in the dataset:
            **{automation:.2%}**
            """
        )

    elif risk == "medium":

        st.info(
            f"""
            **Risk Category: Medium**

            The dataset classifies **{selected_job}**
            as Medium automation risk.

            Automation probability in the dataset:
            **{automation:.2%}**
            """
        )

    else:

        st.success(
            f"""
            **Risk Category: Low**

            The dataset classifies **{selected_job}**
            as Low automation risk.

            Automation probability in the dataset:
            **{automation:.2%}**
            """
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI Job Impact Analyzer | Python + Pandas + Machine Learning + Data Visualization"
)

