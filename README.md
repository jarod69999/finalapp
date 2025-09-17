
# Hors-Site | Explorer BDD Antoine (Cloud Ready)

## Contenu
- `app.py` → code Streamlit
- `requirements.txt` → dépendances
- `README.md` → ce guide
- (optionnel) `HSC_Matrice prix Pilotes_2025.xlsx` → si présent, l'app le charge automatiquement

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement sur Streamlit Cloud
1. Décompressez ce dossier et poussez-le sur un repo GitHub public ou privé.
2. Allez sur [https://streamlit.io/cloud](https://streamlit.io/cloud) et connectez votre repo.
3. Définissez `app.py` comme script principal.
4. Le build prend 2–3 minutes → vous obtenez un lien public `https://<nom>.streamlit.app`.

⚠️ Si vous voulez un fichier Excel par défaut, placez `HSC_Matrice prix Pilotes_2025.xlsx` dans le repo.
