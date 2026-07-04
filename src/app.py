from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from config import DEFAULT_MODEL_PATH
from utils import load_artifact
from preprocess import engineer_features
from shap_utils import generate_shap_waterfall


@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_resource
def load_model(path: Path | str):
    return load_artifact(path)


def style_risk_column(val):
    """Aplicați gradient roșu pe coloana de procent de risc de fraudă."""
    if isinstance(val, str) and "%" in val:
        try:
            risk_val = float(val.replace("%", "").strip())
            # Mapează riscul 0-100 la intensitate de culoare
            intensity = risk_val / 100
            return f"background-color: rgba(255, {int(100 - intensity * 100)}, {int(100 - intensity * 100)}, {intensity * 0.8})"
        except:
            return ""
    return ""


def main():
    st.set_page_config(page_title="Tabloul de Control Detecție Fraudă", layout="wide")

    # ========== LOAD MODEL FIRST ==========
    artifact = None
    try:
        artifact = load_model(DEFAULT_MODEL_PATH)
    except Exception as e:
        st.error(f"Eroare la încărcarea modelului: {e}")
        return
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("## ⚙️ Controale Sistem")
        threshold = st.slider(
            "Prag Detecție Fraudă",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.01,
            help="Ajustați pragul de probabilitate pentru detectarea fraudei"
        )
        st.divider()
        
        # Display Model Performance Metrics
        st.markdown("### 📈 Performanța Modelului")
        metrics = artifact.get("metrics", {})
        
        if metrics:
            st.metric(
                "Acuratețe",
                f"{metrics.get('accuracy', 0) * 100:.2f}%"
            )
            st.metric(
                "Precizie",
                f"{metrics.get('precision', 0) * 100:.2f}%"
            )
            st.metric(
                "Recall",
                f"{metrics.get('recall', 0) * 100:.2f}%"
            )
            st.metric(
                "ROC AUC",
                f"{metrics.get('roc_auc', 0) * 100:.2f}%"
            )
        else:
            st.info("Metricile nu sunt disponibile. Antrenează modelul mai întâi.")
        
        st.divider()
        st.markdown("**Status Model:** ✅ Operațional")
        st.markdown("**Ultima actualizare:** 21/06/2026")

    # ========== MAIN TITLE ==========
    st.title("🏦 Tabloul de Control - Detecție Fraudă")
    st.markdown("---")

    # ========== TABS ==========
    tabs = st.tabs(["📊 Prezentare Generală", "📁 Analiză în Lot", "🔍 Tranzacție Unică"])

    # ========== TAB 1: Dashboard Overview ==========
    with tabs[0]:
        st.header("Prezentare Generală a Tabloului de Control")
        st.markdown("Bine ați venit la Tabloul de Control de Detecție a Fraudei. Utilizați fila **Analiză în Lot** pentru a încărca fișiere de tranzacții, "
                    "sau fila **Tranzacție Unică** pentru a evalua tranzacții individuale.")
        
        col1, col2 = st.columns(2)
        col1.metric("🔐 Tip Model", "XGBoost Classifier")
        col2.metric("⚡ Status", "Activ")
        
        st.markdown("### Cum se utilizează")
        st.markdown("""
        1. **Analiză în Lot**: Încărcați un fișier CSV cu date de tranzacții. Sistemul va prezice probabilitatea de fraudă pentru fiecare tranzacție.
        2. **Tranzacție Unică**: Introduceți detaliile unei tranzacții pentru a obține o evaluare instantanee a riscului de fraudă.
        3. Utilizați cursorul **Prag Detecție Fraudă** (⚙️ bară laterală) pentru a ajusta sensibilitatea predicțiilor.
        """)

    # ========== TAB 2: Batch Analysis ==========
    with tabs[1]:
        st.header("Analiză în Lot")
        
        uploaded = st.file_uploader("Încărcați fișier CSV", type=["csv"])

        if uploaded is not None:
            df_original = load_csv(uploaded)
            df_proc = engineer_features(df_original.copy())

            # Prevent data leakage / shape mismatch
            if "Class" in df_proc.columns:
                df_proc = df_proc.drop(columns=["Class"])

            feature_columns = artifact.get("feature_columns", [])
            scaler = artifact.get("scaler")
            model = artifact.get("model")

            # Ensure feature columns exist
            missing = [c for c in feature_columns if c not in df_proc.columns]
            for c in missing:
                df_proc[c] = 0.0

            X = df_proc[feature_columns]
            X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_columns)

            proba = model.predict_proba(X_scaled)[:, 1]
            df_proc["fraud_proba"] = proba
            df_proc["predicted"] = (df_proc["fraud_proba"] >= threshold).astype(int)

            # Store original Amount and Hour for display
            original_amount = df_original["Amount"].values if "Amount" in df_original.columns else np.zeros(len(df_proc))
            original_hour = ((df_original["Time"] // 3600) % 24).astype(int).values if "Time" in df_original.columns else np.zeros(len(df_proc), dtype=int)

            # ========== METRICS ==========
            total = len(df_proc)
            frauds = int(df_proc["predicted"].sum())
            safe = total - frauds

            col1, col2, col3 = st.columns(3)
            col1.metric("Tranzacții Totale", total, delta=None)
            col2.metric("Tranzacții Legitime", safe, delta=None)
            col3.metric("⚠️ Fraude Suspectate", frauds, delta=None)

            st.divider()

            # ========== PLOTLY CHARTS ==========
            chart_col1, chart_col2 = st.columns(2)

            # Chart 1: Average Amount Comparison
            with chart_col1:
                try:
                    amount_means = df_proc.assign(Amount=original_amount).groupby("predicted")["Amount"].mean()
                    amount_means = amount_means.reindex([0, 1]).fillna(0.0)
                    
                    chart_data = pd.DataFrame({
                        "Status": ["Legitim", "Fraudulent"],
                        "Suma Medie (EUR)": [amount_means.loc[0], amount_means.loc[1]]
                    })
                    
                    fig_bar = px.bar(
                        chart_data,
                        x="Status",
                        y="Suma Medie (EUR)",
                        color="Status",
                        color_discrete_map={"Legitim": "#2ecc71", "Fraudulent": "#e74c3c"},
                        text_auto=".2f",
                        title="Suma Medie a Tranzacțiilor",
                        labels={"Suma Medie (EUR)": "Suma (EUR)"}
                    )
                    fig_bar.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.error(f"Eroare la generarea graficului sumelor: {e}")

            # Chart 2: Transaction Distribution (Pie/Donut)
            with chart_col2:
                try:
                    dist_data = pd.DataFrame({
                        "Status": ["Legitim", "Fraudulent"],
                        "Număr": [safe, frauds]
                    })
                    
                    fig_pie = px.pie(
                        dist_data,
                        names="Status",
                        values="Număr",
                        color="Status",
                        color_discrete_map={"Legitim": "#2ecc71", "Fraudulent": "#e74c3c"},
                        hole=0.4,
                        title="Distribuția Tranzacțiilor"
                    )
                    fig_pie.update_layout(height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as e:
                    st.error(f"Eroare la generarea graficului distribuției: {e}")

            st.divider()

            # ========== FRAUD REGISTER TABLE ==========
            st.subheader("🚨 Registrul Fraudelor")
            
            fraud_df = df_proc[df_proc["predicted"] == 1].copy()
            
            if len(fraud_df) > 0:
                # Create display dataframe with only relevant columns
                # Adjust ID by +2 so displayed ID matches user's file numbering
                display_df = pd.DataFrame({
                    "ID Tranzacție": fraud_df.index.to_numpy() + 2,
                    "Suma (EUR)": original_amount[fraud_df.index],
                    "Oră": original_hour[fraud_df.index],
                    "Risc Fraudă (%)": (fraud_df["fraud_proba"].values * 100).astype(int).astype(str) + "%"
                })

                # Apply styling using map instead of deprecated applymap
                styled_df = display_df.style.map(style_risk_column)
                st.dataframe(styled_df, use_container_width=True, height=400)

                # Download button
                csv_bytes = fraud_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Descărcați Raportul de Fraudă (CSV)",
                    data=csv_bytes,
                    file_name="raport_frauda.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ Nu au fost detectate tranzacții frauduloase!")

            st.divider()

            # ========== FEATURE IMPORTANCES ==========
            st.subheader("📊 Importanța Caracteristicilor (XGBoost)")
            try:
                importances = model.feature_importances_
                fi = pd.DataFrame({"Caracteristică": feature_columns, "Importanță": importances})
                fi = fi.sort_values("Importanță", ascending=False).head(15).reset_index(drop=True)
                
                fig_imp = px.bar(
                    fi,
                    x="Importanță",
                    y="Caracteristică",
                    orientation="h",
                    title="Top 15 Caracteristici Cel Mai Importante",
                    color="Importanță",
                    color_continuous_scale="Viridis"
                )
                fig_imp.update_layout(height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception:
                st.info("Modelul nu expune importanța caracteristicilor.")

    # ========== TAB 3: Single Transaction ==========
    with tabs[2]:
        st.header("Predicție Tranzacție Unică")
        st.markdown("Evaluați o singură tranzacție introducând detaliile mai jos.")

        if artifact is None:
            st.warning("Modelul nu este disponibil.")
        else:
            feature_columns = artifact.get("feature_columns", [])
            scaler = artifact.get("scaler")
            model = artifact.get("model")

            mode = st.radio(
                "Alegeți modul de predicție",
                ["Prediciție simplă", "Predicție avansată cu analiză SHAP"],
                index=0,
                horizontal=True,
            )

            if mode == "Prediciție simplă":
                with st.form("single_tx_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        amount = st.number_input("Suma (EUR)", min_value=0.0, value=10.0, format="%.2f")
                    
                    with col2:
                        time_seconds = st.number_input("Timp (secunde)", min_value=0, max_value=86400, value=0)

                    st.markdown("**Notă:** Alte caracteristici (V1-V28) sunt inițializate cu zero.")

                    submitted = st.form_submit_button("🔍 Preziceți Riscul de Fraudă", use_container_width=True)

                if submitted:
                    row = {c: 0.0 for c in feature_columns}
                    if "Amount_log" in row:
                        row["Amount_log"] = np.log1p(amount)
                    if "Hour" in row:
                        row["Hour"] = int((time_seconds // 3600) % 24)

                    X_row = pd.DataFrame([row], columns=feature_columns)
                    X_scaled = pd.DataFrame(scaler.transform(X_row), columns=feature_columns)

                    proba = model.predict_proba(X_scaled)[0, 1]
                    pred = int(proba >= threshold)

                    st.divider()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Probabilitate Fraudă", f"{proba*100:.2f}%")
                    
                    with col2:
                        if pred == 1:
                            st.error(f"⚠️ **Predicție: FRAUDULENT**")
                        else:
                            st.success(f"✅ **Predicție: LEGITIM**")

                    st.markdown(f"**Prag Utilizat:** {threshold*100:.0f}%")

            else:  # Advanced SHAP Analysis
                with st.form("shap_tx_form"):
                    st.markdown(
                        "Introduceți un vector complet de tranzacție cu 30 de valori separate prin virgulă: "
                        "Time, V1, V2, ..., V28, Amount"
                    )
                    transaction_text = st.text_area(
                        "Vector de tranzacție",
                        value="",
                        height=200,
                        placeholder="0,0.1,-0.2,...,0.00,100.0"
                    )

                    submitted = st.form_submit_button("🔍 Analiză SHAP Avansată", use_container_width=True)

                if submitted:
                    error_message = None
                    try:
                        values = [float(x.strip()) for x in transaction_text.split(",") if x.strip() != ""]
                        if len(values) != 30:
                            raise ValueError(f"Vectorul trebuie să conțină exact 30 de valori; ați furnizat {len(values)}.")

                        columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
                        X_row = pd.DataFrame([values], columns=columns)
                        X_proc = engineer_features(X_row.copy())

                        missing = [c for c in feature_columns if c not in X_proc.columns]
                        for c in missing:
                            X_proc[c] = 0.0

                        X_proc = X_proc[feature_columns]
                        X_scaled = pd.DataFrame(scaler.transform(X_proc), columns=feature_columns)

                        proba = model.predict_proba(X_scaled)[0, 1]
                        pred = int(proba >= threshold)

                        st.divider()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Probabilitate Fraudă", f"{proba*100:.2f}%")
                        with col2:
                            if pred == 1:
                                st.error(f"⚠️ **Predicție: FRAUDULENT**")
                            else:
                                st.success(f"✅ **Predicție: LEGITIM**")

                        st.markdown(f"**Prag Utilizat:** {threshold*100:.0f}%")

                        try:
                            fig = generate_shap_waterfall(model, X_scaled)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Eroare la generarea vizualizării SHAP: {e}")

                    except Exception as e:
                        error_message = str(e)

                    if error_message:
                        st.error(error_message)


if __name__ == "__main__":
    main()