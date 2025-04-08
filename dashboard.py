import streamlit as st
import requests
import pandas as pd
import gcsfs
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import base64

# Constants
API_URL = "https://employee-attrition-778867587659.europe-west1.run.app/"
GCS_BUCKET = "employee_attr"
GCS_RISK_SCORES = "risk_score_df.csv"
GCS_FEATURE_IMPORTANCE = "feature_importance_df.csv"

def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def load_css():
    # Set page config
    st.set_page_config(
        page_title="Employee Risk Dashboard",
        layout="wide",
        page_icon="📊"
    )


    # 1. Paths to files
    css_file = Path(__file__).parent / "assets" / "styles.css"
    img_file = Path(__file__).parent / "assets" / "bg.jpg"

    # 2. Convert image to Base64
    img_b64 = img_to_base64(img_file)
    bg_image = f'url("data:image/jpg;base64,{img_b64}")'

    # 3. Read CSS and replace the variable
    with open(css_file) as f:
        css = f.read().replace("var(--bg-image)", bg_image)

    # 4. Inject the modified CSS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# @st.cache_data(ttl=3600)  # Cache for 1 hour
def load_gcs_data(bucket_name: str, file_path: str) -> pd.DataFrame:

    try:
        # Initialize GCSFS with credentials from Streamlit Secrets
        fs = gcsfs.GCSFileSystem(token=dict(st.secrets["gcp_service_account"]))
        with fs.open(f"{bucket_name}/{file_path}") as f:
            return pd.read_csv(f, skip_blank_lines=True, on_bad_lines="skip", encoding="utf-8")
    except Exception as e:
        st.error(f"Failed to read {file_path} from GCS: {e}")
        return pd.DataFrame()  # Return empty DataFrame as fallback


def plot_survival_curves(survival_df: pd.DataFrame):
        # Create an interactive survival curve plot using plotly
    fig = px.line(
        survival_df,
        x='Time',
        y='SurvivalProbability',
        color='EmployeeNumber',
        labels={'Time': 'Time (Years)', 'SurvivalProbability': 'Survival Probability'},
        title='Survival Curves for Top 10 High-Risk Employees',
        hover_name='EmployeeNumber'  # Show EmployeeNumber on hover
    )

    # Customize the plot
    fig.update_traces(mode='lines+markers', marker=dict(size=5))
    fig.update_layout(
        xaxis_title='Time (Years)',
        yaxis_title='Tenure Probability',
        legend_title='Employee Number',
        xaxis_range=[0, 10],
        height=500,
    )
    return fig

def risk_gradient_color(val, max_risk, min_risk):
    """Generate orange gradient from max to min risk"""
    normalized = (val - min_risk) / (max_risk - min_risk)
    # Convert to 0-255 range (more orange for higher risk)
    r = 255
    g = max(50, int(255 * (1 - normalized * 0.8)))  # Keep some orange tint
    b = 0
    return f'background-color: rgb({r},{g},{b})'

# Main App
def main():
    # Load CSS
    load_css()
    st.title("Employee Attrition Risk Analysis")
    st.markdown("""
    This dashboard analyzes employee attrition risk using predictive models.
    Data is loaded directly from Google Cloud Storage.
    """)

    # Load data
    try:
        st.empty()
        st.empty()
        risk_scores_df = load_gcs_data(GCS_BUCKET, GCS_RISK_SCORES)
        feature_importance_df = load_gcs_data(GCS_BUCKET, GCS_FEATURE_IMPORTANCE)
        col1, col2 = st.columns([3, 4])
        with col2:
            # Display data preview
            st.empty()
            st.empty()
            st.empty()
            st.empty()
            with st.expander("Preview Active Employees at Risk", expanded=True):
                # Reorder columns with predictedRisk first
                columns_order = ['PredictedRisk'] + [col for col in risk_scores_df.columns if col != 'PredictedRisk']

                # Display dataframe without index
                st.dataframe(
                    risk_scores_df[columns_order].head(10),
                    hide_index=True,
            )
        with col1:
            st.subheader("Main factors")

            # Set modern style
            sns.set_style("whitegrid")

            # Filter features by importance threshold (0.001)
            threshold = 0.001
            filtered_df = feature_importance_df[feature_importance_df['Importance'] > threshold].sort_values(by='Importance', ascending=False)
            # Create custom color gradient from light pink to your red (#FF4B4B)
            colors = [
                (1.0, 0.9, 0.0),  # Yellow
                (1.0, 0.294, 0.0)  # #FF4B4B converted to 0-1 RGB
            ]
            cmap = LinearSegmentedColormap.from_list("custom_red_gradient", colors)

            # Normalize importance values for coloring
            norm = plt.Normalize(filtered_df['Importance'].min(), filtered_df['Importance'].max())
            bar_colors = cmap(norm(filtered_df['Importance']))


            # Create plot with modern styling
            plt.figure(figsize=(6, 8))  # Slightly taller for better proportions
            ax = sns.barplot(
                x='Importance',
                y='Feature',  # Horizontal bars often work better for feature importance
                data=filtered_df,
                edgecolor='.1',  # Subtle edge color
                palette=bar_colors,
                linewidth=0.1
            )

            # Customize plot appearance
            # plt.title("Feature Importance in Employee Attrition Prediction",
            #         fontsize=16, pad=20, fontweight='bold')
            plt.xlabel("Importance", fontsize=12, labelpad=10)
            plt.ylabel("Feature", fontsize=12, labelpad=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # Remove spines for cleaner look
            sns.despine(left=True, bottom=True)

            # Add value annotations if there's space
            max_len = filtered_df['Feature'].str.len().max()
            if len(filtered_df) < 15 and max_len < 20:  # Only annotate if not too crowded
                for p in ax.patches:
                    width = p.get_width()
                    ax.text(width + 0.001, p.get_y() + p.get_height()/2.,
                            f'{width:.3f}',
                            ha='left', va='center', fontsize=9)
            st.pyplot(plt.gcf())
            plt.clf()  # Clear figure to avoid memory issues

        col1, col2 = st.columns([1, 4])
        with col1:
            num_samples = st.slider(
                "Number of high-risk employees to analyze",
                min_value=5,
                max_value=25,
                value=10,
                step=5
            )


        if st.button("Show Survival Curves"):
            with st.spinner("Analysing data..."):
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    top_n_high_risk =risk_scores_df.head(num_samples)
                    # Prepare API request payload
                    payload = {
                        "hr_data": top_n_high_risk.drop(columns=['PredictedRisk']).to_dict(orient="list"),
                    }

                    # Make API request
                    response = requests.post(API_URL+"getSurvivalCurves", json=payload)
                    response.raise_for_status()

                    # Process response
                    results = response.json()
                    survival_df = pd.DataFrame(results["survival_df"])

                    st.subheader(f"Top {num_samples} High-Risk Employees")
                    max_risk = top_n_high_risk['PredictedRisk'].max()
                    min_risk = top_n_high_risk['PredictedRisk'].min()

                    # Create compact display
                    compact_cols = ['PredictedRisk','EmployeeNumber', 'Age', 'Department', 'JobRole',
                                'JobLevel', 'YearsAtCompany']
                    compact_view = top_n_high_risk[compact_cols].copy()
                    compact_view.insert(0, 'Rank', range(1, len(compact_view)+1))
                    # Apply gradient styling
                    styled_df = compact_view.style.apply(
                        lambda x: [risk_gradient_color(v, max_risk, min_risk) for v in x],
                        subset=['PredictedRisk'],
                        axis=0
                    ).format({'PredictedRisk': "{:.2f}"})

                    df_height = 35 * len(top_n_high_risk) + 35
                    st.dataframe(styled_df, height=df_height, hide_index=True, use_container_width=True)

                with col2:
                    # Display survival curves

                    st.subheader("Survival Analysis")
                    st.plotly_chart(
                        plot_survival_curves(survival_df),
                        use_container_width=True
                    )
                with col3:
                    st.subheader("Key Metrics")
                    st.metric("Highest Risk", f"{top_n_high_risk['PredictedRisk'].max():.2f}")
                    st.metric("Average Tenure", f"{top_n_high_risk['YearsAtCompany'].mean():.1f} years")

                    # Get top 2 risky departments
                    dept_counts = top_n_high_risk['Department'].value_counts().head(2)
                    top_depts = pd.DataFrame({
                        'Department': dept_counts.index,
                        'At Risk Employees': dept_counts.values
                    })

                    st.write("**Departments at Risk**")
                    st.dataframe(
                        top_depts,
                        hide_index=True,
                        use_container_width=True,
                        height = 35 * len(top_depts) + 38)

                                # Find highest JobLevel at risk
                                # max_level = top_n_high_risk['JobLevel'].max()
                                # high_level_risks = top_n_high_risk[top_n_high_risk['JobLevel'] == max_level]

                                # if not high_level_risks.empty:
                                #     highest = high_level_risks.iloc[0]

                                #     # Create mini-table
                                #     high_risk_employee = pd.DataFrame([['Risk Score', 'Employee', 'Department'],
                                #         [
                                #             f"{highest['PredictedRisk']:.2f}",
                                #             highest['EmployeeNumber'],
                                #             highest['Department']
                                #         ]])

                                #     st.write(f"**Highest Level at Risk: (L{max_level})**")
                                #     st.dataframe(
                                #         high_risk_employee,
                                #         hide_index=True,
                                #         use_container_width=True,
                                #         height=150  # Fixed compact size
                                #     )

                    st.download_button(
                            label="Download Results",
                            data=top_n_high_risk.to_csv(index=False),
                            file_name=f"high_risk_employees.csv",
                            mime="text/csv"
                        )



    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()



# import mlflow.pyfunc

# # Load model from MLflow
# MLFLOW_TRACKING_URI = "http://your-mlflow-server.com"  # Update with your MLflow server
# MODEL_NAME = "employee_attrition"
# MODEL_STAGE = "Production"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# # Streamlit UI
# st.title("Employee Attrition Risk Dashboard")

# # Fetch employee data
# api_url = "http://localhost:8000/employees"
# response = requests.get(api_url)
# if response.status_code == 200:
#     employees = pd.DataFrame(response.json())
# else:
#     st.error("Failed to fetch employee data")
#     employees = pd.DataFrame()

# # Predict attrition risk
# if not employees.empty:
#     employees["attrition_risk"] = model.predict(employees.drop(columns=["EmployeeID"]))
#     high_risk = employees.sort_values(by="attrition_risk", ascending=False)

#     # Display top at-risk employees
#     num_top = st.slider("Select number of high-risk employees to view", 1, 20, 5)
#     st.write(high_risk.head(num_top))

#     # Placeholder for survival curves & risk factor insights
#     st.subheader("Survival Curves for High-Risk Employees")
#     st.write("(Visualization placeholder)")

#     st.subheader("Risk Factors & HR Recommendations")
#     st.write("(Analysis placeholder)")

# else:
#     st.warning("No employee data available.")
