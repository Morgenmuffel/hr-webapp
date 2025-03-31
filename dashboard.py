import streamlit as st
import requests
import pandas as pd
import gcsfs
import plotly.express as px
from datetime import datetime
# Constants
API_URL = "http://localhost:8000/employee_attrition"
GCS_BUCKET = "employee_attr"
GCS_RISK_SCORES = "risk_score_df.csv"

# Set page config
st.set_page_config(
    page_title="Employee Risk Dashboard",
    layout="wide",
    page_icon="📊"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_gcs_data(bucket_name: str, file_path: str) -> pd.DataFrame:
    """Load CSV from GCS"""
    fs = gcsfs.GCSFileSystem()
    with fs.open(f"{bucket_name}/{file_path}") as f:
        return pd.read_csv(f)

def format_risk_table(df: pd.DataFrame)     :
    """Format the high-risk employee table"""
    return df.style.format({
        'risk_score': '{:.2f}',
        'age': '{:.0f}',
        'tenure': '{:.1f} years'
    }).background_gradient(
        subset=['risk_score'],
        cmap='Reds'
    ).set_properties(
        subset=None, **{'text-align': 'left'}
    )

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
        yaxis_title='Survival Probability',
        legend_title='Employee Number',
        xaxis_range=[0, 10]
    )
    return fig

# Main App
def main():
    st.title("Employee Attrition Risk Analysis")
    st.markdown("""
    This dashboard analyzes employee attrition risk using predictive models.
    Data is loaded directly from Google Cloud Storage.
    """)

    # Load data
    try:
        risk_scores_df = load_gcs_data(GCS_BUCKET, GCS_RISK_SCORES)

        # Display data preview
        with st.expander("Preview Source Data"):
            st.dataframe(risk_scores_df.head())

        # Make API request when button clicked
        if st.button("Run Risk Prediction"):
            with st.spinner("Analyzing employee risks..."):
                try:
                    # Prepare API request payload
                    payload = {
                        "risk_scores_df": risk_scores_df.to_dict(orient="records"),
                        "num_samples": 10
                    }

                    # Make API request
                    response = requests.post(API_URL, json=payload)
                    response.raise_for_status()

                    # Process response
                    results = response.json()
                    top_n_high_risk = pd.DataFrame(results["top_n_high_risk"])
                    survival_df = pd.DataFrame(results["survival_df"])

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader(f"Top {num_samples} High-Risk Employees")
                        st.dataframe(
                            top_n_high_risk.style.format({
                                'risk_score': '{:.3f}',
                                'age': '{:.0f}'
                            }).background_gradient(
                                subset=['risk_score'],
                                cmap='Reds'
                            ),
                            height=600,
                            use_container_width=True
                        )

                    with col2:
                        st.subheader("Survival Analysis")
                        st.plotly_chart(
                            plot_survival_curves(survival_df),
                            use_container_width=True
                        )

                    # Download buttons
                    st.download_button(
                        label="Download High-Risk Employees Data",
                        data=top_n_high_risk.to_csv(index=False),
                        file_name="high_risk_employees.csv",
                        mime="text/csv"
                    )

                except requests.exceptions.RequestException as e:
                    st.error(f"API Error: {str(e)}")
                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")

    except Exception as e:
        st.error(f"Error loading data from GCS: {str(e)}")

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
