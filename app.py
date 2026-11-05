"""
app.py - Petite interface pour tester l'endpoint Azure ML (sujet F : abandon étudiant)

Lancement (dans le dossier qui contient sample-request.json) :
    pip install streamlit requests pandas
    streamlit run app.py

L'URL et la clé de l'endpoint se saisissent dans la barre latérale :
la clé n'est jamais écrite dans le code.
"""
import json

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Prédiction d'abandon étudiant", page_icon="🎓")
st.title("🎓 Prédiction d'abandon étudiant")
st.caption("Interface de test de l'endpoint Azure ML (modèle students-model)")

# --- Connexion à l'API ---
with st.sidebar:
    st.header("Connexion à l'API")
    url = st.text_input("URL de l'endpoint (scoring_uri)")
    cle = st.text_input("Clé d'accès", type="password")
    st.caption("URL : endpoint.scoring_uri dans le notebook. "
               "Clé : ml_client.online_endpoints.get_keys(nom_endpoint).primary_key")

# --- Données à envoyer ---
st.subheader("Données à envoyer")
fichier = st.file_uploader('Fichier JSON au format {"data": [ {...}, ... ]}', type="json")

if fichier is not None:
    texte = fichier.read().decode("utf-8")
else:
    try:
        with open("sample-request.json", encoding="utf-8") as f:
            texte = f.read()
    except FileNotFoundError:
        texte = '{"data": []}'

texte = st.text_area("Contenu JSON (modifiable avant l'envoi)", texte, height=300)

# --- Envoi ---
if st.button("Envoyer à l'API", type="primary"):
    if not url or not cle:
        st.error("Renseigne l'URL et la clé dans la barre latérale.")
    else:
        try:
            payload = json.loads(texte)
        except json.JSONDecodeError as e:
            st.error(f"JSON invalide : {e}")
        else:
            try:
                r = requests.post(
                    url,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {cle}"},
                    timeout=60,
                )
            except requests.RequestException as e:
                st.error(f"Impossible de joindre l'API : {e}")
            else:
                st.write("Code HTTP :", r.status_code)
                try:
                    rep = r.json()
                    if isinstance(rep, str):
                        rep = json.loads(rep)
                except ValueError:
                    st.error("Réponse non lisible :")
                    st.code(r.text)
                else:
                    if isinstance(rep, dict) and "erreur" in rep:
                        st.error(f"L'API a renvoyé une erreur : {rep['erreur']}")
                    elif isinstance(rep, dict) and "predictions" in rep:
                        probas = rep.get("probabilites_abandon",
                                         rep.get("probabilites_defaut", []))
                        tableau = pd.DataFrame({
                            "Étudiant": range(1, len(rep["predictions"]) + 1),
                            "Prédiction": ["Abandon" if p == 1 else "Pas d'abandon"
                                           for p in rep["predictions"]],
                            "Probabilité d'abandon": probas,
                        })
                        st.success("Prédictions reçues")
                        st.dataframe(tableau, hide_index=True)
                    else:
                        st.json(rep)