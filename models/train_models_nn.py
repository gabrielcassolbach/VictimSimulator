from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split

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
    y_class = df["severity_class"].astype(int).values
    y_value = df["severity_value"].values

    # 2) Split para classif. e reg.
    X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
        X, y_class, test_size=0.2, random_state=seed, stratify=y_class
    )
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
        X, y_value, test_size=0.2, random_state=seed
    )

    # 3) Pré‑processamento (todos numéricos)
    preproc = ColumnTransformer(
        [("num", StandardScaler(), list(range(X.shape[1])))],
        remainder="drop",
    )
    # Classifier 1
    clf = Pipeline([
        ("pre", preproc),
        ("model", MLPClassifier(
            hidden_layer_sizes=(50, 25),  # duas camadas ocultas
            activation="relu",
            solver="adam",
            alpha=1e-4,                   # L2 regularization
            max_iter=300,
            random_state=seed
        )),
    ])
    
    reg = Pipeline([
        ("pre", preproc),
        ("model", MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=300,
            random_state=seed
        )),
    ])

    # 5) Treina
    clf.fit(X_train_clf, y_train_clf)
    reg.fit(X_train_reg, y_train_reg)

    # 6) Avalia (opcional)
    print("Classifier acc (hold‑out):", clf.score(X_val_clf, y_val_clf))
    print("Regressor R² (hold‑out):", reg.score(X_val_reg, y_val_reg))

    # 7) Salva com joblib
    joblib.dump(clf, out_dir / "severity_clf_mlp.joblib")
    joblib.dump(reg, out_dir / "severity_reg_mlp.joblib")
    print(f"✅ treinamento concluido")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina MLPs e salva com joblib")
    parser.add_argument("--csv", "-c", default="training_vitals.csv")
    parser.add_argument("--out", "-o", default="")
    args = parser.parse_args()
    train_and_save(args.csv, args.out)


"""
-> MODELO 1
    # 4) Pipelines com MLP
    clf = Pipeline([
        ("pre", preproc),
        ("model", MLPClassifier(
            hidden_layer_sizes=(50, 25),  # duas camadas ocultas
            activation="relu",
            solver="adam",
            alpha=1e-4,                   # L2 regularization
            max_iter=300,
            random_state=seed
        )),
    ])
    
    reg = Pipeline([
        ("pre", preproc),
        ("model", MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=300,
            random_state=seed
        )),
    ])

    # Classifier 2
    clf = Pipeline([
        ("pre", preproc),
        ("model", MLPClassifier(
            hidden_layer_sizes=(100, 100, 50),  # 3 camadas
            activation="tanh",                  # outra função de ativação
            solver="sgd",                       # SGD em vez de Adam
            learning_rate="invscaling",
            learning_rate_init=1e-3,
            alpha=1e-4,
            max_iter=500,
            random_state=seed
        )),
    ])

    # Config 3: rede profunda leve
    reg = Pipeline([
        ("pre", preproc),
        ("model", MLPRegressor(
            hidden_layer_sizes=(100, 100, 50),
            activation="tanh",
            solver="sgd",
            learning_rate="invscaling",
            learning_rate_init=1e-3,
            alpha=1e-4,
            max_iter=500,
            random_state=seed
        )),
    ])
    # Classifier 3
    clf = Pipeline([
        ("pre", preproc),
        ("model", MLPClassifier(
            hidden_layer_sizes=(50,),   # 1 camada com 50 neurônios
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=300,
            random_state=seed
        )),
    ])

    # Config 1: rede rasa
    reg = Pipeline([
        ("pre", preproc),
        ("model", MLPRegressor(
            hidden_layer_sizes=(50,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=300,
            random_state=seed
        )),
    ])

"""