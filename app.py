import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
        "OPÉRATION", "DATE ATTRIBUTION", "TYPOLOGIE", "SYSTÈME HORS SITE",
        "NB LOGEMENTS", "Industriel", "SHAB", "Sacc (SDP pour les vieux projets)",
        "Prix conception", "Prix travaux (compris VRD)", "Prix VRD",
        "Prix global", "Prix hors-site seul"
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
                "Prix hors-site seul"]:
        if col in df.columns:
            df[col] = to_num(df[col])

    if "OPÉRATION" in df.columns:
        df["OPÉRATION"] = df["OPÉRATION"].astype(str).str.strip()

    return df

def format_val(x, unit=""):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return "—"
    if pd.isna(x):
        return "—"
    x = float(x)
    if unit == "€/m²":
        return f"{x:,.0f} €/m²".replace(",", " ")
    elif unit == "m²":
        return f"{x:,.0f} m²".replace(",", " ")
    elif unit == "%":
        return f"{x:.1f} %"
    else:
        return f"{x:,.2f}"

def safe_div(a, b):
    try:
        if pd.notna(a) and pd.notna(b) and float(b) != 0:
            return float(a) / float(b)
    except Exception:
        return None
    return None

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

    st.header("📈 Options graphiques")
    show_graph = st.checkbox("Afficher l'évolution des prix moyens par année", value=False)

st.subheader("📊 Chiffres clés")

if len(df_proj) == 0:
    st.warning("Aucun résultat avec ces critères.")
else:
    # === Mode Projet précis ===
    if projet != "Tous":
        st.markdown(f"### 📌 Projet : **{projet}**")

        indicateurs = []
        for idx, row in df_proj.iterrows():
            shab = row.get("SHAB")
            sdp = row.get("Sacc (SDP pour les vieux projets)")
            prix_travaux = row.get("Prix travaux (compris VRD)")
            prix_vrd = row.get("Prix VRD")
            prix_global = row.get("Prix global")
            prix_conception = row.get("Prix conception")
            prix_hors_site = row.get("Prix hors-site seul")

            travaux_hors_vrd = None
            if pd.notna(prix_travaux) and pd.notna(prix_vrd):
                travaux_hors_vrd = prix_travaux - prix_vrd

            industriel = row.get("Industriel")
            if pd.isna(industriel) or str(industriel).strip() == "":
                industriel = f"Groupement {len(indicateurs)+1}"

            indicateurs.append({
                "Industriel": str(industriel),
                "Prix global / m² SHAB": safe_div(prix_global, shab),
                "Prix C/R hors VRD / m² SHAB": safe_div(prix_conception + travaux_hors_vrd, shab),
                "Prix travaux hors VRD / m² SHAB": safe_div(travaux_hors_vrd, shab),
                "Prix hors-site seul / m² SHAB": safe_div(prix_hors_site, shab),
                "SDP / SHOB": safe_div(sdp, shab),
                "Prix global / m² SDP": safe_div(prix_global, sdp),
                "Prix C/R hors VRD / m² SDP": safe_div(prix_conception + travaux_hors_vrd, sdp),
                "Prix travaux hors VRD / m² SDP": safe_div(travaux_hors_vrd, sdp),
            })

        df_indic = pd.DataFrame(indicateurs)

        # Min par colonne
        min_cols = {
            col: df_indic[col].min()
            for col in df_indic.columns if col not in ["Industriel"]
        }

        def highlight_min(val, col):
            if pd.isna(val):
                return "—"
            # ✅ afficher toujours la valeur, avec ✅ si min
            if col in min_cols and val == min_cols[col]:
                return f"✅ {format_val(val, '€/m²')}"
            if "€/m²" in col:
                return format_val(val, "€/m²")
            elif col == "SDP / SHOB":
                return f"{val:.2f}" if pd.notna(val) else "—"
            return format_val(val)

        styled = df_indic.copy()
        for col in styled.columns:
            if col in min_cols:
                styled[col] = styled[col].apply(lambda v: highlight_min(v, col))
            elif col == "Industriel":
                styled[col] = styled[col].apply(lambda v: str(v))
            elif col == "SDP / SHOB":
                styled[col] = styled[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
            else:
                styled[col] = styled[col].apply(lambda v: format_val(v, "€/m²"))

        # Moyenne projet
        moy = df_indic.mean(numeric_only=True)
        moyenne = {"Industriel": "📊 Moyenne projet"}
        for col in df_indic.columns:
            if col == "Industriel":
                continue
            if col == "SDP / SHOB":
                moyenne[col] = f"{moy.get(col):.2f}" if pd.notna(moy.get(col)) else "—"
            else:
                moyenne[col] = format_val(moy.get(col), "€/m²")

        styled = pd.concat([styled, pd.DataFrame([moyenne])], ignore_index=True)

        st.markdown("#### Comparatif par groupement (avec moyennes)")
        st.markdown(styled.to_html(index=False, escape=False), unsafe_allow_html=True)

        # 📥 Export Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            styled.to_excel(writer, index=False, sheet_name="Comparatif")
        excel_data = output.getvalue()

        st.download_button(
            label="📥 Télécharger le comparatif en Excel",
            data=excel_data,
            file_name=f"comparatif_{projet}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # === Mode Année ===
    elif year != "Toutes":
        st.markdown(f"### 📌 Moyennes pour l'année {year}")

        indicateurs_annee = {
            "Prix global / m² SHAB": safe_div(df_proj["Prix global"].sum(), df_proj["SHAB"].sum()),
            "Prix C/R hors VRD / m² SHAB": safe_div(df_proj["Prix conception"].sum() + (df_proj["Prix travaux (compris VRD)"].sum() - df_proj["Prix VRD"].sum()), df_proj["SHAB"].sum()),
            "Prix travaux hors VRD / m² SHAB": safe_div(df_proj["Prix travaux (compris VRD)"].sum() - df_proj["Prix VRD"].sum(), df_proj["SHAB"].sum()),
            "Prix hors-site seul / m² SHAB": safe_div(df_proj["Prix hors-site seul"].sum(), df_proj["SHAB"].sum()),
            "SDP / SHOB": safe_div(df_proj["Sacc (SDP pour les vieux projets)"].sum(), df_proj["SHAB"].sum()),
            "Prix global / m² SDP": safe_div(df_proj["Prix global"].sum(), df_proj["Sacc (SDP pour les vieux projets)"].sum()),
            "Prix C/R hors VRD / m² SDP": safe_div(df_proj["Prix conception"].sum() + (df_proj["Prix travaux (compris VRD)"].sum() - df_proj["Prix VRD"].sum()), df_proj["Sacc (SDP pour les vieux projets)"].sum()),
            "Prix travaux hors VRD / m² SDP": safe_div(df_proj["Prix travaux (compris VRD)"].sum() - df_proj["Prix VRD"].sum(), df_proj["Sacc (SDP pour les vieux projets)"].sum()),
        }

        df_annee = pd.DataFrame([indicateurs_annee])
        for col in df_annee.columns:
            if "Prix" in col:
                df_annee[col] = df_annee[col].apply(lambda v: format_val(v, "€/m²"))
            elif "SDP" in col:
                df_annee[col] = df_annee[col].apply(lambda v: format_val(v))
        st.markdown(df_annee.to_html(index=False, escape=False), unsafe_allow_html=True)

# === Graphiques évolution ===
if show_graph and "Année" in df:
    st.subheader("📈 Évolution des prix moyens par année")

    indicateurs_graph = st.sidebar.multiselect(
        "Choisir les indicateurs à afficher",
        ["Prix global / m² SHAB", "Prix travaux hors VRD / m² SHAB", "Prix travaux hors VRD / m² SDP"],
        default=["Prix global / m² SHAB"]
    )

    for indic in indicateurs_graph:
        df_temp = df.copy()
        if indic == "Prix travaux hors VRD / m² SHAB" and all(c in df.columns for c in ["Prix travaux (compris VRD)", "Prix VRD", "SHAB"]):
            df_temp[indic] = (df_temp["Prix travaux (compris VRD)"] - df_temp["Prix VRD"]) / df_temp["SHAB"]
        elif indic == "Prix travaux hors VRD / m² SDP" and all(c in df.columns for c in ["Prix travaux (compris VRD)", "Prix VRD", "Sacc (SDP pour les vieux projets)"]):
            df_temp[indic] = (df_temp["Prix travaux (compris VRD)"] - df_temp["Prix VRD"]) / df_temp["Sacc (SDP pour les vieux projets)"]

        if indic in df_temp.columns:
            df_graph = df_temp.groupby("Année")[indic].mean().reset_index()
            if len(df_graph) > 0:
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(df_graph["Année"], df_graph[indic], marker="o")
                ax.set_title(f"{indic} (moyenne annuelle)")
                ax.set_xlabel("Année")
                ax.set_ylabel(indic)
                st.pyplot(fig)

st.caption("💡 Conseil : placez le fichier Excel dans le repo avec le nom exact `HSC_Matrice prix Pilotes_2025.xlsx` pour qu'il soit chargé automatiquement.")


