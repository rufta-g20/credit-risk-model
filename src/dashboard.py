import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Bati Bank Credit Tool", layout="wide")

def get_prediction(payload):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

st.title("🏦 Bati Bank: Strategic Credit Decisioning")
st.markdown("Compare customer profiles and simulate 'What-If' scenarios.")

# Create two columns for two different scenarios
col_a, col_b = st.columns(2)

scenarios = {}

for label, col in [("Scenario A", col_a), ("Scenario B", col_b)]:
    with col:
        st.header(label)
        # Unique keys for streamlit widgets are required
        f_count = st.slider(f"Frequency (Transactions) - {label}", 1, 500, 50, key=f"f_{label}")
        m_total = st.number_input(f"Total Debit Amount - {label}", value=1000.0, key=f"m_{label}")
        r_days = st.slider(f"Recency (Days) - {label}", 0, 365, 30, key=f"r_{label}")
        
        # Fixed values for simplicity in comparison
        payload = {
            "R_Min_Days": float(r_days),
            "F_Count": float(f_count),
            "M_Debit_Total": float(m_total),
            "M_Debit_Mean": float(m_total / f_count),
            "M_Debit_Std": 100.0,
            "Hour_Mode": 12,
            "Channel_Mode": "ChannelId_1",
            "Product_Mode": "airtime"
        }
        
        if st.button(f"Analyze {label}"):
            scenarios[label] = get_prediction(payload)

st.divider()

# Results Section
if scenarios:
    res_col1, res_col2 = st.columns(2)
    
    for i, (label, res) in enumerate(scenarios.items()):
        target_col = res_col1 if i == 0 else res_col2
        with target_col:
            if "error" in res:
                st.error(f"Error: {res['error']}")
            else:
                # Display Metrics
                color = "inverse" if res["risk_label"] == "High Risk" else "normal"
                st.metric(f"{label} Probability", f"{res['risk_probability']*100:.1f}%", delta=res["risk_label"], delta_color=color)
                
                # Display SHAP explanations as a horizontal bar chart
                if "justification" in res:
                    st.subheader(f"Why is {label} {res['risk_label']}?")
                    shap_df = pd.DataFrame(list(res["justification"].items()), columns=['Feature', 'Impact'])
                    # Plotly is cleaner for dashboards
                    fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', 
                                 color='Impact', color_continuous_scale='RdYlGn_r')
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)