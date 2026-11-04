"""
prep.py - Étape 1 du pipeline : préparation des données (sujet F)
Lit le fichier (séparateur point-virgule), supprime les lignes incomplètes
et les doublons, puis écrit un fichier propre. La cible est créée dans train.py.
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV d'entrée")
    parser.add_argument("--output", type=str, required=True, help="Dossier de sortie")
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep=";")
    df.columns = df.columns.str.strip()   # nettoie les noms de colonnes
    print("Lignes en entrée :", len(df))

    df = df.dropna()
    df = df.drop_duplicates()
    print("Lignes en sortie :", len(df))

    os.makedirs(args.output, exist_ok=True)
    df.to_csv(os.path.join(args.output, "clean.csv"), sep=";", index=False)
    print("Fichier écrit :", os.path.join(args.output, "clean.csv"))


if __name__ == "__main__":
    main()
