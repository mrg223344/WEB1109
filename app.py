# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import shap

st.set_page_config(page_title="Rebleeding Risk Prediction", page_icon="🩸", layout="centered")

# ==============================
# 加载模型 + 动态创建 SHAP 解释器
# ==============================
@st.cache_resource
def load_artifacts():
    try:
        model = CatBoostClassifier()
        model.load_model("catboost_model.cbm")
        feature_names = joblib.load("feature_names.pkl")
        explainer = shap.TreeExplainer(model)  # 动态创建，最稳定
        return model, explainer, feature_names
    except Exception as e:
        st.error(f"❌ 加载失败: {e}")
        st.stop()

model, explainer, feature_names = load_artifacts()

# ==============================
# 预测函数
# ==============================
def predict_risk(method, location, creatinine, bun, pt, aptt, rockall, aims65):
    # 构造输入（顺序不重要，但列名必须匹配）
    input_dict = {
        "Method": int(method),
        "Location": int(location),
        "Creatinine": float(creatinine),
        "BUN": float(bun),
        "PT": float(pt),
        "APTT": float(aptt),
        "Rockall": int(rockall),
        "AIMS65": int(aims65)
    }
    input_df = pd.DataFrame([input_dict])

    # 确保列顺序与训练一致（防止 CatBoost 警告）
    input_df = input_df[feature_names]

    # 预测
    prob = model.predict_proba(input_df)[0][1]  # DN group probability
    prob = np.clip(prob, 0.0, 1.0)

    # SHAP
    shap_vals = explainer.shap_values(input_df)[0]
    return prob, shap_vals, input_df.iloc[0]

# ==============================
# 用户界面
# ==============================
st.title("🩸 Rebleeding Risk Prediction for Peptic Ulcer Bleeding")
st.markdown("""
> Predicts risk of rebleeding (**DN group**) using clinical features.
""")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox("Treatment Method", options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Clip", 2: "Electrocoagulation", 3: "Spray", 4: "Injection", 5: "Combined"}[x])
        location = st.selectbox("Lesion Location", options=list(range(1, 10)),
            format_func=lambda x: {
                1: "Cardia", 2: "Gastric Body", 3: "Fundus", 4: "Angle",
                5: "Antrum", 6: "Pylorus", 7: "Duodenal Bulb",
                8: "Descending Duodenum", 9: "Anastomotic Stoma"
            }[x])
        creatinine = st.number_input("Creatinine (μmol/L)", min_value=0.0, max_value=2000.0, value=80.0, step=0.1)
        bun = st.number_input("BUN (mmol/L)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    with col2:
        pt = st.number_input("PT (seconds)", min_value=0.0, max_value=60.0, value=13.0, step=0.1)
        aptt = st.number_input("APTT (seconds)", min_value=0.0, max_value=200.0, value=30.0, step=0.1)
        rockall = st.number_input("Rockall Score", min_value=0, max_value=12, value=9, step=1)
        aims65 = st.number_input("AIMS65 Score", min_value=0, max_value=5, value=3, step=1)

    submitted = st.form_submit_button("📊 Calculate Rebleeding Risk", type="primary")

# ==============================
# 结果展示
# ==============================
if submitted:
    with st.spinner("Calculating..."):
        try:
            prob, shap_vals, sample = predict_risk(method, location, creatinine, bun, pt, aptt, rockall, aims65)
        except Exception as e:
            st.error(f"⚠️ 预测出错: {e}")
            st.stop()

    # 风险等级
    if prob < 0.4:
        risk, color, icon = "Low", "green", "✅"
    elif prob < 0.7:
        risk, color, icon = "Moderate", "orange", "⚠️"
    else:
        risk, color, icon = "High", "red", "🚨"

    st.markdown(f"### 🔍 Result: {icon} <span style='color:{color}; font-weight:bold'>{risk} Risk</span>", unsafe_allow_html=True)
    st.write(f"**Rebleeding Probability:** {prob * 100:.1f}%")
    st.progress(float(prob))

    # SHAP 解释
    st.markdown("### 📊 SHAP Feature Contribution")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals,
            base_values=explainer.expected_value,
            data=sample.values,
            feature_names=feature_names
        ),
        max_display=8,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)

st.caption("For research and decision support only. Not a substitute for clinical judgment.")