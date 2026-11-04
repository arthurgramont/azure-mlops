# score.py : script de scoring du modèle students-model
import os
import json
import pandas as pd
import mlflow.sklearn

model = None


def init():
    """Chargement du modèle au démarrage du conteneur."""
    global model
    racine = os.getenv("AZUREML_MODEL_DIR")

    chemin_modele = None
    for dossier, _, fichiers in os.walk(racine):
        if "MLmodel" in fichiers:
            chemin_modele = dossier
            break

    model = mlflow.sklearn.load_model(chemin_modele)
    print(f"[init] Modèle chargé depuis : {chemin_modele}")


def run(raw_data):
    """Appelée à chaque requête. raw_data est le texte JSON envoyé à l'API."""
    try:
        donnees = json.loads(raw_data)["data"]
        X = pd.DataFrame(donnees).astype(float)

        colonnes = getattr(model, "feature_names_in_", None)
        if colonnes is not None:
            manquantes = [c for c in colonnes if c not in X.columns]
            if manquantes:
                print(f"[run] ERREUR : colonnes manquantes {manquantes}")
                return {"erreur": f"{len(manquantes)} colonne(s) manquante(s) : {manquantes}"}
            X = X[list(colonnes)]

        predictions = model.predict(X).tolist()
        probabilites = model.predict_proba(X)[:, 1].round(3).tolist()

        print(f"[run] {len(X)} ligne(s) reçue(s), prédictions : {predictions}")
        return {"predictions": predictions, "probabilites_abandon": probabilites}

    except Exception as erreur:
        print(f"[run] ERREUR : {erreur}")
        return {"erreur": str(erreur)}
