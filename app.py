import streamlit as st
import pandas as pd
from io import BytesIO
import os

st.set_page_config(page_title="Hors-Site | Explorer BDD Antoine (Cloud)", layout="wide")

@st.cache_data(show_spinner=False)
def load_and_transform(file_bytes: bytes):
    raw = pd.read_excel(BytesIO(file_bytes), sheet_name="BDD Antoine", header=None)
    project_cols = raw.iloc[:, 4:]
    labels = raw.iloc[:, 1]
    df = project_cols.T.reset_index(drop=True)
    df.columns = labels
    df.columns = df.columns.astype(str).str.strip()

    wanted = [
        "OPÉRATION",
        "DATE ATTRIBUTION",
        "TYPOLOGIE",
        "SYSTÈME HORS SITE",
        "NB LOGEMENTS",
        "Industriel",
        "SHAB",
        "Sacc (SDP pour les vieux projets)",
        "Prix conception",
        "Prix travaux (compris VRD)",
        "Prix VRD",
        "Prix global",
        "Prix hors-site seul",
        "Prix global / m² SHAB",
        "Prix C/R hors VRD / m² SHAB",
    ]
    cols_exist = [c for c in wanted if c in df.columns]
    df = df[cols_exist].copy()

    if "DATE ATTRIBUTION" in df.columns:
        df["Année"] = df["DATE ATTRIBUTION"].astype(str).str.extract(r"(\d{4})")

    # ✅ to_num robuste
    def to_num(s):
        s = pd.Series([str(x) if x is not None else "" for x in s])
        s = (
            s.str.replace("\u202f", "", regex=False)
             .str.replace("\xa0", "", regex=False)
             .str.replace(" ", "", regex=False)
             .str.replace(",", ".", regex=False)
             .str.replace("€", "", regex=False)
             .str.replace("m²", "", regex=False)
             .str.replace("%", "", regex=False)
             .str.strip()
        )
        return pd.to_numeric(s, errors="coerce")

    for col in ["SHAB","Sacc (SDP pour les vieux projets)","Prix conception",
                "Prix travaux (compris VRD)","Prix VRD","Prix global",
                "Prix hors-site seul","Prix global / m² SHAB",
                "Prix C/R hors VRD / m² SHAB"]:
        if col in df.columns:
            df[col] = to_num(df[col])

    if "OPÉRATION" in df.columns:
        df["OPÉRATION"] = df["OPÉRATION"].astype(str).str.strip()

    return df

# ✅ Fix: nettoyer les colonnes non sérialisables
def sanitize_for_streamlit(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict, set))).any():
            df[col] = df[col].astype(str)
    return df

def format_money(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):,.0f} €".replace(",", " ")

def format_unit(x, unit=""):
    if pd.isna(x):
        return "—"
    val = float(x)
    if unit == "€/m²":
        return f"{val:,.0f} €/m²".replace(",", " ")
    elif unit == "m²":
        return f"{val:,.0f} m²".replace(",", " ")
    return f"{val:,.0f}".replace(",", " ")

st.title("🔎 Explorer les projets — BDD Antoine (Cloud)")

default_file = "HSC_Matrice prix Pilotes_2025.xlsx"
df = None

with st.sidebar:
    st.header("📂 Source des données")
    uploaded = st.file_uploader("Importer le fichier Excel", type=["xlsx"])

    if uploaded is not None:
        df = load_and_transform(uploaded.read())
    elif os.path.exists(default_file):
        st.success(f"Fichier trouvé : {default_file} (chargé automatiquement)")
        with open(default_file, "rb") as f:
            df = load_and_transform(f.read())
    else:
        st.warning("⚠️ Merci de charger un fichier Excel pour commencer.")
        st.stop()

    st.header("🧭 Arborescence")
    years = sorted([y for y in df.get("Année", pd.Series(dtype=str)).dropna().unique()])
    year = st.selectbox("1) Sélectionner l'année", ["Toutes"] + years)
    df_year = df if year == "Toutes" else df[df["Année"] == year]

    projets = sorted([p for p in df_year.get("OPÉRATION", pd.Series(dtype=str)).dropna().unique()])
    projet = st.selectbox("2) Sélectionner le projet", ["Tous"] + projets)
    df_proj = df_year if projet == "Tous" else df_year[df_year["OPÉRATION"] == projet]

st.subheader("📊 Chiffres clés")
if len(df_proj) == 0:
    st.warning("Aucun résultat avec ces critères.")
else:
    if projet != "Tous" and len(df_proj) >= 1:
        row = df_proj.iloc[0]

        # Nouveaux indicateurs demandés
        c1, c2, c3, c4, c5 = st.columns(5)
        if not pd.isna(row.get("Prix travaux (compris VRD)")) and not pd.isna(row.get("VRD")) and not pd.isna(row.get("SHAB")):
            with c1: st.metric("Travaux hors VRD / m² SHAB",
                               format_unit((row.get("Prix travaux (compris VRD)") - row.get("Prix VRD")) / row.get("SHAB"), "€/m²"))
        with c2: st.metric("Prix global / m² SHAB", format_unit(row.get("Prix global / m² SHAB"), "€/m²"))
        if not pd.isna(row.get("Sacc (SDP pour les vieux projets)")):
            with c3: st.metric("Travaux hors VRD / m² Sacc",
                               format_unit((row.get("Prix travaux (compris VRD)") - row.get("Prix VRD")) / row.get("Sacc (SDP pour les vieux projets)"), "€/m²"))
        with c4: st.metric("SHAB", format_unit(row.get("SHAB"), "m²"))
        if not pd.isna(row.get("Prix conception")) and not pd.isna(row.get("Prix global")):
            with c5: st.metric("Taux honoraire", f"{row.get('Prix conception') / row.get('Prix global') * 100:.1f} %")

    else:
        st.info("Sélectionnez un projet précis pour voir les chiffres clés.")

    st.divider()
    st.subheader("🔬 Résultats détaillés")

    # 🔧 Fix: supprimer colonnes dupliquées + objets non sérialisables
    df_proj = df_proj.loc[:, ~df_proj.columns.duplicated()].copy()
    df_proj = sanitize_for_streamlit(df_proj)

    st.dataframe(df_proj, use_container_width=True)

    csv = df_proj.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Exporter en CSV", data=csv, file_name="resultats.csv", mime="text/csv")

    out_xlsx = BytesIO()
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_proj.to_excel(writer, index=False, sheet_name="Résultats")
    st.download_button("📊 Exporter en Excel", data=out_xlsx.getvalue(),
                       file_name="resultats.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("💡 Conseil : placez le fichier Excel dans le repo avec le nom exact `HSC_Matrice prix Pilotes_2025.xlsx` pour qu'il soit chargé automatiquement.")


