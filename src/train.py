"""
train.py - Entraînement du modèle de prédiction d'abandon des étudiants (sujet F)
MLflow (paramètres, métriques, modèle) + validation et enregistrement conditionnel.
"""
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "abandon"


def parse_args():
    """1. Lecture des paramètres passés au script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Chemin du fichier CSV")
    parser.add_argument("--class_weight", type=str, default="none",
                        choices=["none", "balanced"],
                        help="Pondération des classes")
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="Part des données gardée pour le test (0.3 = 30 %%)")
    parser.add_argument("--min_recall", type=float, default=0.5,
                        help="Recall minimum pour enregistrer le modèle")
    parser.add_argument("--model_name", type=str, default="students-model",
                        help="Nom du modèle dans le registre")
    return parser.parse_args()


def load_data(path):
    """2. Lecture (séparateur point-virgule) et préparation du sujet F."""
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    df[TARGET] = (df["Target"] == "Dropout").astype(int)
    df = df.drop(columns=["Target"])
    return df


def train_model(X_train, y_train, class_weight):
    """3. Entraînement : standardisation + régression logistique."""
    cw = None if class_weight == "none" else class_weight
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight=cw, max_iter=1000),
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """4. Évaluation sur le jeu de test."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }


def main():
    args = parse_args()
    print(f"Paramètres : class_weight={args.class_weight}, "
          f"test_size={args.test_size}")

    df = load_data(args.data)
    print(f"Données : {df.shape[0]} lignes, {df.shape[1]} colonnes, "
          f"taux d'abandon = {df[TARGET].mean():.1%}")
    X = df.drop(columns=[TARGET]).astype(float)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y)

    with mlflow.start_run():
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("test_size", args.test_size)

        model = train_model(X_train, y_train, args.class_weight)
        metrics = evaluate(model, X_test, y_test)
        for name, value in metrics.items():
            print(name, ":", round(value, 4))

        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
        print("Modèle enregistré dans le run MLflow (dossier model/)")

        if metrics["recall"] >= args.min_recall:
            run_id = mlflow.active_run().info.run_id
            mlflow.register_model(f"runs:/{run_id}/model", args.model_name)
            mlflow.set_tag("validation", "acceptée")
            print(f"Modèle validé (recall >= {args.min_recall}) : "
                  f"nouvelle version de {args.model_name}")
        else:
            mlflow.set_tag("validation", "refusée")
            print(f"Modèle refusé : recall {metrics['recall']:.3f} "
                  f"< {args.min_recall}")


if __name__ == "__main__":
    main()
