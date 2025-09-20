from __future__ import annotations

"""
Transfer Learning para séries temporais (LSTM) SEM hyperparameter tuning.
- Fase A: congela o modelo base e treina apenas a nova cabeça Dense(1)
- Fase B (opcional): fine-tuning curto liberando somente a última LSTM, com LR menor
- Usa seus arrays salvos por ticker (X_train_<T>.npy, y_train_<T>.npy, X_test_<T>.npy, y_test_<T>.npy, scaler_<T>.pkl)
- Gera métricas (R2, RMSE, MAE, MAPE) e 3 gráficos (série, dispersão, resíduos)

Exemplos:
  # SBSP3 -> ELET3, SEM fine-tuning
  python transfer_tl.py --base Modelos/SBSP3/Modelo_SBSP3.keras --ticker ELET3 --no-ft

  # BBDC4 -> BBAS3, COM fine-tuning
  python transfer_tl.py --base Modelos/BBDC4/Modelo_BBDC4.keras --ticker BBAS3 --ft
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# ==========================
# Configs (sem hyperparameter tuning)
# ==========================
EPOCHS_HEAD = 1000   # teto alto; EarlyStopping interrompe antes
EPOCHS_FT   = 300    # teto alto para FT curto
VAL_SPLIT   = 0.15
LR_HEAD     = 1e-3   # LR “normal” para treinar a cabeça
LR_FT       = 1e-4   # LR menor no FT (padrão do tutorial)
METRICS     = [keras.metrics.RootMeanSquaredError(name="rmse"), "mae"]


# ==========================
# Utilitários
# ==========================
def load_ticker_arrays(ticker: str, root: str = "Dados/Treinamento"):
    p = Path(root) / ticker
    Xtr = np.load(p / f"X_train_{ticker}.npy")
    ytr = np.load(p / f"y_train_{ticker}.npy")
    Xte = np.load(p / f"X_test_{ticker}.npy")
    yte = np.load(p / f"y_test_{ticker}.npy")
    scaler = joblib.load(p / f"scaler_{ticker}.pkl")
    return Xtr, ytr, Xte, yte, scaler


def build_transfer_model(base: keras.Model) -> keras.Model:
    # Seu modelo termina em ... -> Dense(10) -> Dense(1).
    # Pegamos a saída da penúltima camada (Dense(10)) e criamos nova Dense(1).
    feat = base.layers[-2].output
    out  = layers.Dense(1, name="regressor_transfer")(feat)
    return keras.Model(inputs=base.input, outputs=out, name=f"{base.name}_TL")


def last_lstm_layer(model: keras.Model):
    for lyr in reversed(model.layers):
        if isinstance(lyr, layers.LSTM):
            return lyr
    return None


def compile_with(model: keras.Model, lr: float):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=METRICS)


def inverse_transform_target(scaler, y_scaled: np.ndarray, n_features: int) -> np.ndarray:
    # Desnormaliza o alvo assumindo alvo na 1ª coluna do scaler (como no seu notebook).
    expand = np.zeros((len(y_scaled), n_features))
    expand[:, 0] = y_scaled.reshape(-1)
    return scaler.inverse_transform(expand)[:, 0]


def maybe_inverse_y_test(scaler, y_test: np.ndarray, n_features: int) -> np.ndarray:
    # Se y_test já estiver no domínio real, apenas retorna.
    try:
        return inverse_transform_target(scaler, y_test.reshape(-1, 1), n_features)
    except Exception:
        return y_test


def callbacks_for(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.keras"),
            monitor="val_rmse",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_rmse",
            mode="min",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_rmse",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            mode="min",
        ),
    ]


# ==========================
# Core: Transfer + (opcional) Fine-tuning
# ==========================
def transfer_and_finetune(
    base_model_path: str,
    ticker_target: str,
    dados_root: str = "Dados/Treinamento",
    out_root: str = "ModelosTransfer",
    do_finetune: bool = True,
) -> Dict:
    # 1) Carrega base (seu modelo principal)
    base = keras.models.load_model(base_model_path, compile=False)

    # 2) Congela a base e cria nova cabeça
    base.trainable = False
    model = build_transfer_model(base)

    # 3) Dados do ticker alvo
    Xtr, ytr, Xte, yte, scaler = load_ticker_arrays(ticker_target, dados_root)
    n_features = getattr(scaler, "n_features_in_", None) or Xtr.shape[-1]

    # 4) Treinar apenas a cabeça (feature extraction)
    out_dir = Path(out_root) / ticker_target
    cbs = callbacks_for(out_dir)
    compile_with(model, LR_HEAD)
    histA = model.fit(
        Xtr, ytr,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS_HEAD,   # teto alto; EarlyStopping para antes
        verbose=2             # sem batch_size -> default (32)
    )

    # 5) FT curto (opcional): libera somente a última LSTM
    histB = None
    if do_finetune:
        lstm = last_lstm_layer(model)
        if lstm is not None:
            lstm.trainable = True
        compile_with(model, LR_FT)
        histB = model.fit(
            Xtr, ytr,
            validation_split=VAL_SPLIT,
            epochs=EPOCHS_FT,
            verbose=2
        )

    # 6) Avaliação + previsões desnormalizadas
    test_values = model.evaluate(Xte, yte, verbose=0)
    test_metrics = dict(zip(model.metrics_names, map(float, test_values)))

    y_pred_s = model.predict(Xte, verbose=0).reshape(-1, 1)
    y_pred   = inverse_transform_target(scaler, y_pred_s, n_features)
    y_true   = maybe_inverse_y_test(scaler, yte, n_features)

    # 7) Persistência
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / f"{ticker_target}_final.keras")
    np.save(out_dir / "y_pred_real.npy", y_pred)
    np.save(out_dir / "y_test_real.npy", y_true)

    report = {
        "ticker": ticker_target,
        "out_dir": str(out_dir),
        "phaseA_val_rmse_last": float(histA.history.get("val_rmse", [None])[-1]) if histA else None,
        "phaseB_val_rmse_last": float(histB.history.get("val_rmse", [None])[-1]) if histB else None,
        "test_metrics": test_metrics,
        "config": {
            "epochs_head_teto": EPOCHS_HEAD,
            "epochs_ft_teto": EPOCHS_FT if do_finetune else 0,
            "val_split": VAL_SPLIT,
            "lr_head": LR_HEAD,
            "lr_ft": LR_FT if do_finetune else None,
            "batch_size": "default(32)",
            "do_finetune": do_finetune,
        },
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report