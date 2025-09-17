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

    # ✅ Version robuste de to_num
    def to_num(s):
        s = pd.Series(s, dtype="string")  # force tout en texte
        return pd.to_numeric(
            s.str.replace("\u202f", "", regex=False)
             .str.replace("\xa0", "", regex=False)
             .str.replace(" ", "", regex=False)
             .str.replace(",", ".", regex=False)
             .str.replace("€", "", regex=False)
             .str.replace("m²", "", regex=False)
             .str.replace("%", "", regex=False)
             .str.strip(),
            errors="coerce"
        )

    for col in ["SHAB","Sacc (SDP pour les vieux projets)","Prix conception",
                "Prix travaux (compris VRD)","Prix VRD","Prix global",
                "Prix hors-site seul","Prix global / m² SHAB",
                "Prix C/R hors VRD / m² SHAB"]:
        if col in df.columns:
            df[col] = to_num(df[col])

    if "OPÉRATION" in df.columns:
        df["OPÉRATION"] = df["OPÉRATION"].astype(str).str.strip()

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

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("SHAB", format_unit(row.get("SHAB"), "m²"))
        with c2: st.metric("Sacc (SDP)", format_unit(row.get("Sacc (SDP pour les vieux projets)"), "m²"))
        with c3: st.metric("Prix hors-site seul", format_money(row.get("Prix hors-site seul")))
        with c4: st.metric("Prix global", format_money(row.get("Prix global")))

        c5, c6, c7, c8 = st.columns(4)
        with c5: st.metric("Prix travaux (VRD inclus)", format_money(row.get("Prix travaux (compris VRD)")))
        with c6: st.metric("Prix VRD", format_money(row.get("Prix VRD")))
        with c7: st.metric("Prix global / m² SHAB", format_unit(row.get("Prix global / m² SHAB"), "€/m²"))
        with c8: st.metric("Prix C/R hors VRD / m² SHAB", format_unit(row.get("Prix C/R hors VRD / m² SHAB"), "€/m²"))

        c9, c10 = st.columns(2)
        if not pd.isna(row.get("Prix global")) and not pd.isna(row.get("Prix VRD")):
            val1 = row.get("Prix global") - row.get("Prix VRD")
            with c9: st.metric("Prix C/R hors VRD total (Méthode 1)", format_money(val1))
        if not pd.isna(row.get("Prix C/R hors VRD / m² SHAB")) and not pd.isna(row.get("SHAB")):
            val2 = row.get("Prix C/R hors VRD / m² SHAB") * row.get("SHAB")
            with c10: st.metric("Prix C/R hors VRD total (Méthode 2)", format_money(val2))
    else:
        st.info("Sélectionnez un projet précis pour voir les chiffres clés.")

    st.divider()
    st.subheader("🔬 Résultats détaillés")
    st.dataframe(df_proj, use_container_width=True)

    csv = df_proj.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Exporter en CSV", data=csv, file_name="resultats.csv", mime="text/csv")

    out_xlsx = BytesIO()
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_proj.to_excel(writer, index=False, sheet_name="Résultats")
    st.download_button("📊 Exporter en Excel", data=out_xlsx.getvalue(),
                       file_name="resultats.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("💡 Conseil : placez le fichier Excel dans le repo avec le nom exact `HSC_Matrice prix Pilotes_2025.xlsx` pour qu'il soit chargé automatiquement.")
