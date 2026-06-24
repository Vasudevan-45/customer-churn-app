import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import os
warnings.filterwarnings("ignore")

# Always find the CSV relative to this app.py file, regardless of where CMD is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #64748b; font-size: 1rem; margin-top: 0; }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.2rem 1.5rem;
        text-align: center; margin-bottom: 1rem;
    }
    .metric-card .value {
        font-size: 2rem; font-weight: 700; color: #6366f1;
    }
    .metric-card .label {
        font-size: 0.82rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 0.05em; margin-top: 0.2rem;
    }
    .pred-box {
        padding: 1.5rem 2rem; border-radius: 14px;
        text-align: center; font-size: 1.5rem; font-weight: 700;
        margin-top: 1rem;
    }
    .pred-churn   { background: #fee2e2; color: #b91c1c; border: 2px solid #fca5a5; }
    .pred-stay    { background: #dcfce7; color: #15803d; border: 2px solid #86efac; }
    .section-header {
        font-size: 1.25rem; font-weight: 700; color: #1e293b;
        border-left: 4px solid #6366f1; padding-left: 0.75rem;
        margin: 1.5rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load & Preprocess ────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    csv_path = os.path.join(BASE_DIR, "customer_churn", "Churn_Modelling.csv")
    df = pd.read_csv(csv_path)

    # Drop irrelevant columns
    df_model = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    # Label encode Gender; One-Hot encode Geography
    le = LabelEncoder()
    df_model["Gender"] = le.fit_transform(df_model["Gender"])          # Female=0, Male=1
    df_model = pd.get_dummies(df_model, columns=["Geography"], drop_first=False)

    feature_cols = [c for c in df_model.columns if c != "Exited"]
    X = df_model[feature_cols]
    y = df_model["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "cm":       confusion_matrix(y_test, preds),
        }
        trained[name] = model

    # Feature importance from Random Forest
    rf = trained["Random Forest"]
    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return df, X_train, X_test, y_train, y_test, trained, results, feat_imp, feature_cols


df, X_train, X_test, y_train, y_test, trained, results, feat_imp, feature_cols = load_and_train()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML-powered churn analysis · Logistic Regression · Decision Tree · Random Forest</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Top KPIs ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
best_model_name = max(results, key=lambda k: results[k]["accuracy"])
best_acc = results[best_model_name]["accuracy"]

with c1:
    st.markdown(f'<div class="metric-card"><div class="value">{len(df):,}</div><div class="label">Total Customers</div></div>', unsafe_allow_html=True)
with c2:
    churn_rate = df["Exited"].mean() * 100
    st.markdown(f'<div class="metric-card"><div class="value">{churn_rate:.1f}%</div><div class="label">Churn Rate</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="value">{best_acc*100:.1f}%</div><div class="label">Best Accuracy</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="value">{best_model_name.split()[0]}</div><div class="label">Best Model</div></div>', unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Dataset", "📈 Model Performance", "📊 Visualizations", "🤖 Predict"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dataset
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=320)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Dataset Info</div>', unsafe_allow_html=True)
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type":   df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
        churn_counts = df["Exited"].value_counts().reset_index()
        churn_counts.columns = ["Exited", "Count"]
        churn_counts["Label"] = churn_counts["Exited"].map({0: "Retained", 1: "Churned"})
        fig = px.pie(churn_counts, values="Count", names="Label",
                     color_discrete_sequence=["#6366f1", "#f43f5e"],
                     hole=0.45)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Accuracy Comparison</div>', unsafe_allow_html=True)

    acc_df = pd.DataFrame([
        {"Model": k, "Accuracy": v["accuracy"] * 100}
        for k, v in results.items()
    ]).sort_values("Accuracy", ascending=True)

    fig_acc = px.bar(
        acc_df, x="Accuracy", y="Model", orientation="h",
        color="Accuracy", color_continuous_scale=["#a5b4fc", "#6366f1", "#4f46e5"],
        text=acc_df["Accuracy"].map("{:.2f}%".format),
        range_x=[80, 100],
    )
    fig_acc.update_traces(textposition="outside")
    fig_acc.update_layout(
        height=300, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis_title="Accuracy (%)", yaxis_title="",
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, (name, res) in enumerate(results.items()):
        cm = res["cm"]
        with cols[idx]:
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Retained", "Churned"], y=["Retained", "Churned"],
                color_continuous_scale=["#ede9fe", "#6366f1"],
                title=name,
            )
            fig_cm.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_cm, use_container_width=True)

            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall    = tp / (tp + fn) if (tp + fn) else 0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Accuracy  | {res['accuracy']*100:.2f}% |
            | Precision | {precision*100:.2f}% |
            | Recall    | {recall*100:.2f}% |
            | F1 Score  | {f1*100:.2f}% |
            """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)

    fi_df = feat_imp.reset_index()
    fi_df.columns = ["Feature", "Importance"]
    fi_df = fi_df.sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#c7d2fe", "#6366f1"],
        text=fi_df["Importance"].map("{:.3f}".format),
    )
    fig_fi.update_traces(textposition="outside")
    fig_fi.update_layout(
        height=420, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=10, r=70, t=10, b=10),
        xaxis_title="Importance Score", yaxis_title="",
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Age vs Churn</div>', unsafe_allow_html=True)
        fig_age = px.histogram(
            df, x="Age", color=df["Exited"].map({0: "Retained", 1: "Churned"}),
            barmode="overlay", nbins=30,
            color_discrete_map={"Retained": "#6366f1", "Churned": "#f43f5e"},
            labels={"color": "Status"},
        )
        fig_age.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Balance vs Churn</div>', unsafe_allow_html=True)
        fig_bal = px.box(
            df, x=df["Exited"].map({0: "Retained", 1: "Churned"}),
            y="Balance", color=df["Exited"].map({0: "Retained", 1: "Churned"}),
            color_discrete_map={"Retained": "#6366f1", "Churned": "#f43f5e"},
            labels={"x": "Status", "color": "Status"},
        )
        fig_bal.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig_bal, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Churn by Geography</div>', unsafe_allow_html=True)
        geo = df.groupby(["Geography", "Exited"]).size().reset_index(name="Count")
        geo["Status"] = geo["Exited"].map({0: "Retained", 1: "Churned"})
        fig_geo = px.bar(geo, x="Geography", y="Count", color="Status", barmode="group",
                         color_discrete_map={"Retained": "#6366f1", "Churned": "#f43f5e"})
        fig_geo.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_geo, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Churn by Gender</div>', unsafe_allow_html=True)
        gen = df.groupby(["Gender", "Exited"]).size().reset_index(name="Count")
        gen["Status"] = gen["Exited"].map({0: "Retained", 1: "Churned"})
        fig_gen = px.bar(gen, x="Gender", y="Count", color="Status", barmode="group",
                         color_discrete_map={"Retained": "#6366f1", "Churned": "#f43f5e"})
        fig_gen.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_gen, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Predict
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.write("Enter customer details below and choose a model to predict churn probability.")

    selected_model = st.selectbox("🤖 Select Model", list(trained.keys()), index=2)

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score    = st.slider("Credit Score",    300, 850, 650)
        age             = st.slider("Age",              18,  92,  35)
        tenure          = st.slider("Tenure (years)",   0,  10,   5)
        num_products    = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)

    with col2:
        geography       = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender          = st.selectbox("Gender", ["Female", "Male"])
        has_cr_card     = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        is_active       = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)

    with col3:
        balance         = st.number_input("Balance ($)", 0.0, 300000.0, 50000.0, step=1000.0)
        estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 75000.0, step=1000.0)

    if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):
        # Build input row matching training features
        input_dict = {
            "CreditScore":      credit_score,
            "Gender":           1 if gender == "Male" else 0,
            "Age":              age,
            "Tenure":           tenure,
            "Balance":          balance,
            "NumOfProducts":    num_products,
            "HasCrCard":        1 if has_cr_card == "Yes" else 0,
            "IsActiveMember":   1 if is_active == "Yes" else 0,
            "EstimatedSalary":  estimated_salary,
            "Geography_France":  1 if geography == "France"  else 0,
            "Geography_Germany": 1 if geography == "Germany" else 0,
            "Geography_Spain":   1 if geography == "Spain"   else 0,
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]

        model  = trained[selected_model]
        pred   = model.predict(input_df)[0]
        proba  = model.predict_proba(input_df)[0]

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred == 1:
                st.markdown('<div class="pred-box pred-churn">⚠️ Customer likely to CHURN</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pred-box pred-stay">✅ Customer likely to STAY</div>', unsafe_allow_html=True)

            st.metric("Churn Probability",    f"{proba[1]*100:.1f}%")
            st.metric("Retention Probability", f"{proba[0]*100:.1f}%")

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                title={"text": "Churn Risk %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#6366f1"},
                    "steps": [
                        {"range": [0,  40], "color": "#dcfce7"},
                        {"range": [40, 70], "color": "#fef9c3"},
                        {"range": [70,100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#b91c1c", "width": 3},
                        "thickness": 0.75,
                        "value": 70,
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=20, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(f"Prediction made using **{selected_model}** · Model accuracy: **{results[selected_model]['accuracy']*100:.2f}%**")
