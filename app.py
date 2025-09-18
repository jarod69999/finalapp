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
        "Prix global", "Prix hors-site seul", "Prix global / m² SHAB",
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
        for _, row in df_proj.iterrows():
            travaux_hors_vrd_m2_shab = None
            travaux_hors_vrd_m2_sacc = None
            taux_hono = None
            if all(pd.notna([row["Prix travaux (compris VRD)"], row["Prix VRD"], row["SHAB"]])):
                travaux_hors_vrd_m2_shab = (row["Prix travaux (compris VRD)"] - row["Prix VRD"]) / row["SHAB"]
            if all(pd.notna([row["Sacc (SDP pour les vieux projets)"], row["Prix VRD"], row["Prix travaux (compris VRD)"]])):
                travaux_hors_vrd_m2_sacc = (row["Prix travaux (compris VRD)"] - row["Prix VRD"]) / row["Sacc (SDP pour les vieux projets)"]
            if all(pd.notna([row["Prix conception"], row["Prix global"]])):
                taux_hono = (row["Prix conception"] / row["Prix global"]) * 100

            indicateurs.append({
                "Industriel": row["Industriel"] if "Industriel" in row else "—",
                "Travaux hors VRD / m² SHAB": travaux_hors_vrd_m2_shab,
                "Prix global / m² SHAB": row["Prix global / m² SHAB"] if "Prix global / m² SHAB" in row else None,
                "Travaux hors VRD / m² Sacc": travaux_hors_vrd_m2_sacc,
                "SHAB": row["SHAB"] if "SHAB" in row else None,
                "Taux honoraire": taux_hono,
            })

        df_indic = pd.DataFrame(indicateurs)

        # Min par colonne
        min_cols = {col: df_indic[col].min() for col in ["Travaux hors VRD / m² SHAB","Prix global / m² SHAB","Travaux hors VRD / m² Sacc"] if col in df_indic}

        def highlight_min(val, col):
            # 🔧 sécurité : si ce n’est pas un scalaire, on sort
            if isinstance(val, (pd.Series, pd.DataFrame)):
                return "—"
            if pd.isna(val):
                return "—"
            if col in min_cols and val == min_cols[col]:
                return f"✅ {format_val(val, '€/m²')}"
            return format_val(val, "€/m²") if "m²" in col else format_val(val)

        styled = df_indic.copy()
        for col in styled.columns:
            if col in min_cols:
                styled[col] = styled[col].apply(lambda v: highlight_min(v, col))
            elif col == "SHAB":
                styled[col] = styled[col].apply(lambda v: format_val(v, "m²"))
            elif col == "Taux honoraire":
                styled[col] = styled[col].apply(lambda v: format_val(v, "%"))
            else:
                styled[col] = styled[col].apply(lambda v: str(v) if pd.notna(v) else "—")

        # Moyenne projet
        moy = df_indic.mean(numeric_only=True)
        styled = pd.concat([styled, pd.DataFrame([{
            "Industriel": "📊 Moyenne projet",
            "Travaux hors VRD / m² SHAB": format_val(moy.get("Travaux hors VRD / m² SHAB"), "€/m²"),
            "Prix global / m² SHAB": format_val(moy.get("Prix global / m² SHAB"), "€/m²"),
            "Travaux hors VRD / m² Sacc": format_val(moy.get("Travaux hors VRD / m² Sacc"), "€/m²"),
            "SHAB": format_val(moy.get("SHAB"), "m²"),
            "Taux honoraire": format_val(moy.get("Taux honoraire"), "%"),
        }])], ignore_index=True)

        st.markdown("#### Comparatif par groupement")
        st.markdown(styled.to_html(index=False, escape=False), unsafe_allow_html=True)

    # === Mode Année ===
    elif year != "Toutes":
        st.markdown(f"### 📌 Moyenne pour l'année {year}")
        indicateurs_annee = {
            "Travaux hors VRD / m² SHAB": (df_proj["Prix travaux (compris VRD)"] - df_proj["Prix VRD"]).sum() / df_proj["SHAB"].sum() if "Prix travaux (compris VRD)" in df_proj and "Prix VRD" in df_proj and "SHAB" in df_proj else None,
            "Prix global / m² SHAB": df_proj["Prix global / m² SHAB"].mean() if "Prix global / m² SHAB" in df_proj else None,
            "Travaux hors VRD / m² Sacc": (df_proj["Prix travaux (compris VRD)"] - df_proj["Prix VRD"]).sum() / df_proj["Sacc (SDP pour les vieux projets)"].sum() if "Sacc (SDP pour les vieux projets)" in df_proj else None,
            "SHAB": df_proj["SHAB"].mean() if "SHAB" in df_proj else None,
            "Taux honoraire": (df_proj["Prix conception"].sum() / df_proj["Prix global"].sum() * 100) if "Prix conception" in df_proj and "Prix global" in df_proj else None,
        }
        df_annee = pd.DataFrame([indicateurs_annee])
        for col in df_annee.columns:
            unit = "€/m²" if "m²" in col else "%" if "Taux" in col else "m²"
            df_annee[col] = df_annee[col].apply(lambda v: format_val(v, unit))
        st.markdown(df_annee.to_html(index=False, escape=False), unsafe_allow_html=True)

# === Graphiques évolution ===
if show_graph and "Année" in df:
    st.subheader("📈 Évolution des prix moyens par année")

    indicateurs_graph = st.sidebar.multiselect(
        "Choisir les indicateurs à afficher",
        ["Prix global / m² SHAB", "Travaux hors VRD / m² SHAB", "Travaux hors VRD / m² Sacc"],
        default=["Prix global / m² SHAB"]
    )

    for indic in indicateurs_graph:
        df_temp = df.copy()
        if indic == "Travaux hors VRD / m² SHAB" and all(c in df.columns for c in ["Prix travaux (compris VRD)", "Prix VRD", "SHAB"]):
            df_temp[indic] = (df_temp["Prix travaux (compris VRD)"] - df_temp["Prix VRD"]) / df_temp["SHAB"]
        elif indic == "Travaux hors VRD / m² Sacc" and all(c in df.columns for c in ["Prix travaux (compris VRD)", "Prix VRD", "Sacc (SDP pour les vieux projets)"]):
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

