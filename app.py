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

def format_val(x, unit=""):
    if pd.isna(x):
        return "—"
    if unit == "€/m²":
        return f"{x:,.0f} €/m²".replace(",", " ")
    elif unit == "m²":
        return f"{x:,.0f} m²".replace(",", " ")
    elif unit == "%":
        return f"{x:.1f} %"
    return f"{x:,.0f}".replace(",", " ")

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
    # === Mode Projet précis ===
    if projet != "Tous":
        st.markdown(f"### 📌 Projet : **{projet}**")

        # Calcul des indicateurs par groupement
        indicateurs = []
        for _, row in df_proj.iterrows():
            travaux_hors_vrd_m2_shab = None
            travaux_hors_vrd_m2_sacc = None
            taux_hono = None
            if not pd.isna(row.get("Prix travaux (compris VRD)")) and not pd.isna(row.get("Prix VRD")) and not pd.isna(row.get("SHAB")):
                travaux_hors_vrd_m2_shab = (row["Prix travaux (compris VRD)"] - row["Prix VRD"]) / row["SHAB"]
            if not pd.isna(row.get("Sacc (SDP pour les vieux projets)")) and not pd.isna(row.get("Prix VRD")):
                travaux_hors_vrd_m2_sacc = (row["Prix travaux (compris VRD)"] - row["Prix VRD"]) / row["Sacc (SDP pour les vieux projets)"]
            if not pd.isna(row.get("Prix conception")) and not pd.isna(row.get("Prix global")):
                taux_hono = (row["Prix conception"] / row["Prix global"]) * 100

            indicateurs.append({
                "Industriel": row.get("Industriel", "—"),
                "Travaux hors VRD / m² SHAB": travaux_hors_vrd_m2_shab,
                "Prix global / m² SHAB": row.get("Prix global / m² SHAB"),
                "Travaux hors VRD / m² Sacc": travaux_hors_vrd_m2_sacc,
                "SHAB": row.get("SHAB"),
                "Taux honoraire": taux_hono,
            })

        df_indic = pd.DataFrame(indicateurs)

        # Trouver le minimum pour chaque colonne numérique
        min_cols = {}
        for col in ["Travaux hors VRD / m² SHAB", "Prix global / m² SHAB", "Travaux hors VRD / m² Sacc"]:
            if col in df_indic.columns:
                min_cols[col] = df_indic[col].min()

        # Appliquer la mise en évidence du minimum
        def highlight_min(val, col):
            if pd.isna(val):
                return format_val(val)
            if col in min_cols and val == min_cols[col]:
                return f"✅ {format_val(val, '€/m²')}"
            return format_val(val, '€/m²') if "m²" in col else format_val(val)

        # Transformer en tableau affichable
        styled = df_indic.copy()
        for col in styled.columns:
            if col in min_cols:
                styled[col] = styled[col].apply(lambda v: highlight_min(v, col))
            elif col in ["SHAB"]:
                styled[col] = styled[col].apply(lambda v: format_val(v, "m²"))
            elif col in ["Taux honoraire"]:
                styled[col] = styled[col].apply(lambda v: format_val(v, "%"))
            else:
                styled[col] = styled[col].apply(lambda v: str(v) if not pd.isna(v) else "—")

        # Ajouter ligne moyenne projet
        moy = df_indic.mean(numeric_only=True)
        moy_row = {
            "Industriel": "📊 Moyenne projet",
            "Travaux hors VRD / m² SHAB": format_val(moy.get("Travaux hors VRD / m² SHAB"), "€/m²"),
            "Prix global / m² SHAB": format_val(moy.get("Prix global / m² SHAB"), "€/m²"),
            "Travaux hors VRD / m² Sacc": format_val(moy.get("Travaux hors VRD / m² Sacc"), "€/m²"),
            "SHAB": format_val(moy.get("SHAB"), "m²"),
            "Taux honoraire": format_val(moy.get("Taux honoraire"), "%"),
        }
        styled = pd.concat([styled, pd.DataFrame([moy_row])], ignore_index=True)

        st.markdown("#### Comparatif par groupement")
        st.markdown(styled.to_html(index=False, escape=False), unsafe_allow_html=True)

    # === Mode Année entière ===
    elif year != "Toutes":
        st.markdown(f"### 📌 Moyenne pour l'année {year}")
        moy = df_proj.mean(numeric_only=True)
        indicateurs_annee = {
            "Travaux hors VRD / m² SHAB": (df_proj["Prix travaux (compris VRD)"] - df_proj["Prix VRD"]).sum() / df_proj["SHAB"].sum() if "Prix travaux (compris VRD)" in df_proj and "Prix VRD" in df_proj and "SHAB" in df_proj else None,
            "Prix global / m² SHAB": moy.get("Prix global / m² SHAB"),
            "Travaux hors VRD / m² Sacc": (df_proj["Prix travaux (compris VRD)"] - df_proj["Prix VRD"]).sum() / df_proj["Sacc (SDP pour les vieux projets)"].sum() if "Sacc (SDP pour les vieux projets)" in df_proj else None,
            "SHAB": moy.get("SHAB"),
            "Taux honoraire": (df_proj["Prix conception"].sum() / df_proj["Prix global"].sum() * 100) if "Prix conception" in df_proj and "Prix global" in df_proj else None,
        }
        df_annee = pd.DataFrame([indicateurs_annee])
        for col in df_annee.columns:
            unit = "€/m²" if "m²" in col else "%" if "Taux" in col else "m²"
            df_annee[col] = df_annee[col].apply(lambda v: format_val(v, unit))
        st.markdown(df_annee.to_html(index=False, escape=False), unsafe_allow_html=True)

st.caption("💡 Conseil : placez le fichier Excel dans le repo avec le nom exact `HSC_Matrice prix Pilotes_2025.xlsx` pour qu'il soit chargé automatiquement.")


