import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Bati Bank Credit Tool", layout="wide")

def get_prediction(payload):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

st.title("🏦 Bati Bank: Strategic Credit Decisioning")

col_a, col_b = st.columns(2)

# Use Session State to keep results visible
if 'results' not in st.session_state:
    st.session_state.results = {}

for label, col in [("Scenario A", col_a), ("Scenario B", col_b)]:
    with col:
        with st.form(key=f"form_{label}"):
            st.header(label)
            f_count = st.slider("Frequency", 1, 500, 50, key=f"f_val_{label}")
            m_total = st.number_input("Total Debit", value=1000.0, key=f"m_val_{label}")
            r_days = st.slider("Recency (Days)", 0, 365, 30, key=f"r_val_{label}")
            
            submit = st.form_submit_button(f"Analyze {label}")
            
            if submit:
                payload = {
                    "R_Min_Days": float(r_days),
                    "F_Count": float(f_count),
                    "M_Debit_Total": float(m_total),
                    "M_Debit_Mean": float(m_total / f_count) if f_count > 0 else 0.0,
                    "M_Debit_Std": 100.0,
                    "Hour_Mode": 12,
                    "Channel_Mode": "ChannelId_1",
                    "Product_Mode": "airtime"
                }
                st.session_state.results[label] = get_prediction(payload)

st.divider()

# Results Section
res_cols = st.columns(2)
for i, label in enumerate(["Scenario A", "Scenario B"]):
    if label in st.session_state.results:
        res = st.session_state.results[label]
        with res_cols[i]:
            if "error" in res:
                st.error(res["error"])
            else:
                color = "inverse" if res["risk_label"] == "High Risk" else "normal"
                st.metric(f"{label} Probability", f"{res['risk_probability']*100:.1f}%", 
                          delta=res["risk_label"], delta_color=color)
                
                if "justification" in res and res["justification"]:
                    st.subheader("Decision Logic")
                    shap_df = pd.DataFrame(list(res["justification"].items()), columns=['Feature', 'Impact'])
                    shap_df = shap_df.sort_values(by='Impact')
                    
                    fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h',
                                 color='Impact', color_continuous_scale='RdYlGn_r',
                                 title="Feature Influence")
                    # Fixed the deprecated width parameter here
                    st.plotly_chart(fig, use_container_width=True, on_select="ignore")