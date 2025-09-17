
import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Hors-Site | Explorer BDD Antoine", layout="wide")

@st.cache_data(show_spinner=False)
def load_and_transform(file_bytes: bytes):
    # Charger Excel
    xls = pd.ExcelFile(BytesIO(file_bytes))
    # On lit la feuille "BDD Antoine" sans header pour garder toutes les lignes
    raw = pd.read_excel(BytesIO(file_bytes), sheet_name="BDD Antoine", header=None)
    # Colonnes projet = à partir de la colonne 4
    project_cols = raw.iloc[:, 4:]
    # Étiquettes = colonne 1
    labels = raw.iloc[:, 1]
    # Transposition : une ligne par projet
    df = project_cols.T.reset_index(drop=True)
    df.columns = labels
    # Nettoyage des colonnes
    df.columns = df.columns.astype(str).str.strip()

    # Colonnes que l'on tente de récupérer si elles existent
    wanted = [
        "OPÉRATION",
        "DATE ATTRIBUTION",
        "TYPOLOGIE",
        "SYSTÈME HORS SITE",
        "NB LOGEMENTS",
        "Groupement",
        "Phase",
        "Industriel",
        "SHAB",
        "Sacc (SDP pour les vieux projets)",
        "Prix conception",
        "Prix travaux (compris VRD)",
        "Prix VRD",
        "Prix VRD / m² de terrain",
        "Prix global",
        "Prix hors-site seul",
        "Compacité",
        "Taux d'industrialisation (hors VRD)",
        "Taux d'honoraires",
        "Taux VRD / prix travaux",
        "Prix global / m² SHAB",
        "Prix C/R hors VRD / m² SHAB",
    ]
    cols_exist = [c for c in wanted if c in df.columns]
    df = df[cols_exist].copy()

    # Année depuis "DATE ATTRIBUTION" si dispo
    if "DATE ATTRIBUTION" in df.columns:
        df["Année"] = df["DATE ATTRIBUTION"].astype(str).str.extract(r"(\d{4})")

    # Harmoniser types numériques
    def to_num(s):
        return pd.to_numeric(
            pd.Series(s)
              .astype(str)
              .str.replace("\u202f", "", regex=False)  # espace fine
              .str.replace("\xa0", "", regex=False)    # nbsp
              .str.replace(" ", "", regex=False)
              .str.replace(",", ".", regex=False)
              .str.replace("€", "", regex=False)
              .str.replace("m²", "", regex=False)
              .str.replace("%", "", regex=False)
              .str.strip(),
            errors="coerce"
        )

    for col in ["SHAB",
                "Sacc (SDP pour les vieux projets)",
                "Prix conception",
                "Prix travaux (compris VRD)",
                "Prix VRD",
                "Prix VRD / m² de terrain",
                "Prix global",
                "Prix hors-site seul",
                "Prix global / m² SHAB",
                "Prix C/R hors VRD / m² SHAB"]:
        if col in df.columns:
            df[col] = to_num(df[col])

    # Nettoyage nom projet
    if "OPÉRATION" in df.columns:
        df["OPÉRATION"] = df["OPÉRATION"].astype(str).str.strip()

    return df

def format_money(x):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):,.0f} €".replace(",", " ")
    except Exception:
        return str(x)

def format_unit(x, unit=""):
    if pd.isna(x):
        return "—"
    try:
        val = float(x)
        if unit == "€/m²":
            return f"{val:,.0f} €/m²".replace(",", " ")
        elif unit == "m²":
            return f"{val:,.0f} m²".replace(",", " ")
        else:
            return f"{val:,.0f}".replace(",", " ")
    except Exception:
        return str(x)

st.title("🔎 Explorer les projets — BDD Antoine")

with st.sidebar:
    st.header("📂 Source des données")
    uploaded = st.file_uploader("Importer le fichier Excel (HSC_Matrice prix Pilotes_2025.xlsx)", type=["xlsx"])
    if uploaded is None:
        st.info("Aucun fichier importé. Chargez le fichier Excel pour commencer.")
        st.stop()

    df = load_and_transform(uploaded.read())

    # Filtres principaux (arborescence)
    st.header("🧭 Arborescence")
    years = sorted([y for y in df.get("Année", pd.Series(dtype=str)).dropna().unique()])
    year = st.selectbox("1) Sélectionner l'année", ["Toutes"] + years)
    df_year = df if year == "Toutes" else df[df["Année"] == year]

    projets = sorted([p for p in df_year.get("OPÉRATION", pd.Series(dtype=str)).dropna().unique()])
    projet = st.selectbox("2) Sélectionner le projet", ["Tous"] + projets)
    df_proj = df_year if projet == "Tous" else df_year[df_year["OPÉRATION"] == projet]

    with st.expander("🔎 Filtres avancés (optionnels)"):
        typologies = sorted([t for t in df.get("TYPOLOGIE", pd.Series(dtype=str)).dropna().unique()])
        typology = st.multiselect("Typologie", typologies)
        systemes = sorted([s for s in df.get("SYSTÈME HORS SITE", pd.Series(dtype=str)).dropna().unique()])
        systeme = st.multiselect("Système hors-site", systemes)
        industriel = st.multiselect("Industriel", sorted([i for i in df.get("Industriel", pd.Series(dtype=str)).dropna().unique()]))
        mot_clef = st.text_input("Mot-clé dans le nom du projet")

        temp = df_proj.copy()
        if typology:
            temp = temp[temp["TYPOLOGIE"].isin(typology)]
        if systeme:
            temp = temp[temp["SYSTÈME HORS SITE"].isin(systeme)]
        if industriel and "Industriel" in temp.columns:
            temp = temp[temp["Industriel"].isin(industriel)]
        if mot_clef and "OPÉRATION" in temp.columns:
            temp = temp[temp["OPÉRATION"].str.contains(mot_clef, case=False, na=False)]
        df_proj = temp

st.subheader("📊 Chiffres clés")
if len(df_proj) == 0:
    st.warning("Aucun résultat avec ces critères.")
else:
    # Si un seul projet sélectionné, afficher ses KPIs
    if projet != "Tous" and len(df_proj) >= 1:
        row = df_proj.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("SHAB", format_unit(row.get("SHAB"), "m²"))
        with c2:
            st.metric("Sacc (SDP)", format_unit(row.get("Sacc (SDP pour les vieux projets)"), "m²"))
        with c3:
            st.metric("Prix hors-site seul", format_money(row.get("Prix hors-site seul")))
        with c4:
            st.metric("Prix travaux (compris VRD)", format_money(row.get("Prix travaux (compris VRD)")))

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Prix VRD", format_money(row.get("Prix VRD")))
        with c6:
            st.metric("Prix global", format_money(row.get("Prix global")))
        with c7:
            st.metric("Prix global / m² SHAB", format_unit(row.get("Prix global / m² SHAB"), "€/m²"))
        with c8:
            st.metric("Prix C/R hors VRD / m² SHAB", format_unit(row.get("Prix C/R hors VRD / m² SHAB"), "€/m²"))
    else:
        st.info("Sélectionnez un projet précis pour afficher les KPIs individuels ci-dessus.")

    st.divider()
    st.subheader("🔬 Recherche détaillée")
    show_cols = [c for c in [
        "Année",
        "OPÉRATION",
        "TYPOLOGIE",
        "SYSTÈME HORS SITE",
        "NB LOGEMENTS",
        "Industriel",
        "SHAB",
        "Sacc (SDP pour les vieux projets)",
        "Prix hors-site seul",
        "Prix travaux (compris VRD)",
        "Prix VRD",
        "Prix global",
        "Prix global / m² SHAB",
        "Prix C/R hors VRD / m² SHAB",
    ] if c in df_proj.columns]

    grid = df_proj[show_cols].copy()

    # Formatage lisible pour export/affichage
    if "SHAB" in grid.columns:
        grid["SHAB"] = grid["SHAB"].apply(lambda v: format_unit(v, "m²"))
    if "Sacc (SDP pour les vieux projets)" in grid.columns:
        grid["Sacc (SDP pour les vieux projets)"] = grid["Sacc (SDP pour les vieux projets)"].apply(lambda v: format_unit(v, "m²"))
    for money_col in ["Prix hors-site seul", "Prix travaux (compris VRD)", "Prix VRD", "Prix global"]:
        if money_col in grid.columns:
            grid[money_col] = grid[money_col].apply(format_money)
    for unit_col in ["Prix global / m² SHAB", "Prix C/R hors VRD / m² SHAB"]:
        if unit_col in grid.columns:
            grid[unit_col] = grid[unit_col].apply(lambda v: format_unit(v, "€/m²"))

    st.dataframe(grid, use_container_width=True)

    # Export CSV
    csv = df_proj.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Exporter les résultats (CSV)", data=csv, file_name="resultats_filtrés.csv", mime="text/csv")

st.caption("💡 Astuce : chargez votre fichier Excel, choisissez d'abord l'année, puis le projet — les chiffres clés s'affichent. Utilisez les filtres avancés pour affiner par typologie, système, industriel, etc.")
