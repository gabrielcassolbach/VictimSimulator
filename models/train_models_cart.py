#!/usr/bin/env python
"""
----
$ python train_severity_models.py           # usa caminhos padrão
$ python train_severity_models.py --csv data/meus_dados.csv --out models/
$ python train_severity_models.py -h        # ajuda
"""
from __future__ import annotations
from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_and_save(
    csv_path: str | Path,
    out_dir: str | Path = "models",
    features: list[str] | None = None,
    seed: int = 42,
) -> None:
    csv_path = Path(csv_path)
    out_dir  = Path(out_dir)

    if features is None:
        features = ["pSist", "pDiast", "qPA", "pulse", "resp_freq"]

    # 1) Carrega dados
    df = pd.read_csv(csv_path)

    X = df[features].values
    y_class = df["severity_class"].values
    y_value = df["severity_value"].values

    # 2) Pré-processamento (todas numéricas)
    preproc = ColumnTransformer(
        transformers=[("num", StandardScaler(), list(range(X.shape[1])))],
        remainder="drop",
    )
    """
    -> CLASSIFIER 1:
    Shallower forest (reduce variance, speed up)
    """
    clf = Pipeline([
        ("pre", preproc),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,            # limit tree depth
            class_weight="balanced",
            random_state=seed
        )),
    ])
   
    """ 
    -> REGRESSOR 3
    """
    reg = Pipeline(
        steps=[
            ("pre", preproc),
            ("model", RandomForestRegressor(
                n_estimators=200,
                random_state=seed)),
        ]
    )

    # 4) Treina
    clf.fit(X, y_class)
    reg.fit(X, y_value)

    # 5) Salva
    clf_path = "severity_clf.joblib"
    reg_path = "severity_reg.joblib"
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    print(f"✅ Modelos gravados em: {clf_path}, {reg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina e salva os modelos de severidade.")
    parser.add_argument("--csv",  "-c", default="training_vitals.csv",
                        help="Caminho do CSV de treino (default: %(default)s)")
    parser.add_argument("--out",  "-o", default="",
                        help="Diretório de saída (default: %(default)s)")
    args = parser.parse_args()

    train_and_save(args.csv, args.out)


    """
    -> CLASSIFIER 2:
    Deeper, more randomized splits (more variance, possibly higher bias)
    clf = Pipeline([
        ("pre", preproc),
        ("model", RandomForestClassifier(
            n_estimators=300,        # more trees
            max_features="sqrt",     # fewer features per split
            class_weight="balanced",
            random_state=seed
        )),
    ])
     """

    """
    -> CLASSIFIER 3:
    clf = Pipeline([
        ("pre", preproc),
        ("model", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,      # each leaf needs at least 5 samples
            class_weight="balanced_subsample",
            random_state=seed
        )),
    ])
    """

    """ 
    -> REGRESSOR 1
    Shallower trees for speed and less variance
    reg = Pipeline([
        ("pre", preproc),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=8,             # shallower trees
            random_state=seed
        )),
    ])
    """ 

    """ 
    -> REGRESSOR 2
   More features per split for lower bias
    reg = Pipeline([
        ("pre", preproc),
        ("model", RandomForestRegressor(
            n_estimators=400,        # more trees
            max_features="log2",     # fewer features per split
            random_state=seed
        )),
    ])
    """ 