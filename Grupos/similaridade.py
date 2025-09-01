import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import re

# VARIÁVEIS FIXAS
# Padronização das ações
TICKERS = ["ABEV3", "B3SA3", "BBAS3", "BBDC4", "ELET3", "ITUB4", "PETR3", "PETR4", "SBSP3", "VALE3"]

# Path de onde ficam os arquivos de excel
BASE_DIR = Path("../Dados/Excel/Treinamento")

# Padronização dos nomes das features
FEATURES_STD = ["Close", "Volume"]

# Estatísticas para a "assinatura" de cada ativo
AGG_FUNCS = ["mean", "std", "median", "skew"]

# K permitido (mantém no máximo 3 grupos) - vai escolher entre 2 ou 3 grupos
K_RANGE = [2, 3]

# LIMPAR CARACTERES
def _clean(s: str) -> str:
    s0 = re.sub(r"\s+", " ", str(s).strip().lower())
    s0 = (s0
          .replace("á","a").replace("à","a").replace("â","a").replace("ã","a")
          .replace("é","e").replace("ê","e")
          .replace("í","i")
          .replace("ó","o").replace("ô","o").replace("õ","o")
          .replace("ú","u")
          .replace("ç","c"))
    return s0

# DEIXAR AS COLUNAS DO EXCEL COM O MESMO NOME DAS FEATURES
def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeia colunas no formato Close_*, Volume_* e equivalentes para os nomes padronizados."""
    rename_map = {}
    for col in df.columns:
        c = _clean(col)

        if c.startswith("close_"):
            tail = c.replace("close_", "")
            if tail in ("dolar", "usdbrl"):
                rename_map[col] = "Dolar"
                continue
            if tail in ("petroleo", "brent", "oil"):
                rename_map[col] = "Petroleo"
                continue
            if tail in ("ibovespa", "ibov"):
                rename_map[col] = "Ibov"
                continue
            if "minerio" in tail or "iron" in tail:
                rename_map[col] = "MinerioFerro"
                continue
            rename_map[col] = "Close"
            continue

        if c.startswith("volume_"):
            rename_map[col] = "Volume"
            continue

        if "minerio" in c or "iron" in c:
            rename_map[col] = "MinerioFerro"
        elif "ibov" in c:
            rename_map[col] = "Ibov"
        elif "brent" in c or "petroleo" in c or "oil" in c:
            rename_map[col] = "Petroleo"
        elif c in ("adjclose", "close", "fechamento", "preco", "saida"):
            rename_map[col] = "Close"
        elif "volume" in c:
            rename_map[col] = "Volume"
        elif "dolar" in c or "usdbrl" in c or "usdb" in c:
            rename_map[col] = "Dolar"

    return df.rename(columns=rename_map)

# CARREGAMENTO E ASSINATURAS
def carregar_csv_ticker(ticker: str, features_std: List[str]) -> pd.DataFrame:
    cand1 = BASE_DIR / f"df_completo_{ticker}.csv"
    cand2 = BASE_DIR / ticker / f"{ticker}.csv"
    cand3 = BASE_DIR / f"{ticker}.csv"

    for path in (cand1, cand2, cand3):
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(f"CSV não encontrado para {ticker}: {cand1} | {cand2} | {cand3}")

    if "Date" not in df.columns:
        for c in ("date", "Data", "data"):
            if c in df.columns:
                df = df.rename(columns={c: "Date"})
                break
    if "Date" not in df.columns:
        raise ValueError(f"O arquivo {path} não tem coluna 'Date'/'Data'.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df = padronizar_colunas(df)
    feat_cols = [c for c in features_std if c in df.columns]

    if not feat_cols:
        cols_log = BASE_DIR / f"__columns_{ticker}.txt"
        cols_log.write_text("\n".join(map(str, df.columns)), encoding="utf-8")
        raise ValueError(f"Nenhuma feature válida encontrada no CSV do {ticker}. "
                         f"Veja as colunas detectadas em: {cols_log}")

    df = df[["Date"] + feat_cols].copy()
    df = df.dropna(subset=feat_cols, how="all")
    return df

def assinatura_por_ticker(df: pd.DataFrame, features_std: List[str], agg_funcs: List[str]) -> pd.Series:
    feat_cols = [c for c in features_std if c in df.columns and c != "Date"]
    agg = df[feat_cols].agg(agg_funcs)
    flat = {}
    for func in agg.index:
        for col in agg.columns:
            flat[f"{col}_{func}"] = agg.loc[func, col]
    return pd.Series(flat)

def montar_matriz_assinaturas(tickers: List[str], features_std: List[str], agg_funcs: List[str]) -> Tuple[pd.DataFrame, Dict]:
    assinaturas = []
    meta = {}
    for tk in tickers:
        df = carregar_csv_ticker(tk, features_std)
        linhas_validas = len(df)
        try:
            vec = assinatura_por_ticker(df, features_std, agg_funcs)
            vec.name = tk
            assinaturas.append(vec)
            meta[tk] = {"linhas_validas": int(linhas_validas)}
        except Exception as e:
            meta[tk] = {"linhas_validas": int(linhas_validas), "erro": str(e)}
            print(f"[AVISO] Ignorando {tk}: {e}")

    if not assinaturas:
        raise RuntimeError("Nenhuma assinatura pôde ser construída. Verifique os CSVs e as FEATURES.")

    M = pd.DataFrame(assinaturas)
    M = M.replace([np.inf, -np.inf], np.nan)
    M = M.apply(lambda col: col.fillna(col.median()), axis=0)
    return M, meta

# CLUSTERIZAÇÃO
def escolher_k_e_clusterizar(M: pd.DataFrame, k_range: List[int], random_state: int = 42) -> Tuple[pd.Series, Dict]:
    scaler = StandardScaler()
    X = scaler.fit_transform(M.values)

    melhor_k = None
    melhor_score = -1.0
    relatorio = {}
    for k in k_range:
        if k <= 1 or k > len(M):
            continue
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        relatorio[k] = {"silhouette": float(score)}
        if score > melhor_score:
            melhor_score = score
            melhor_k = k

    if melhor_k is None:
        melhor_k = 2
        km = KMeans(n_clusters=melhor_k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        relatorio["fallback"] = True
    else:
        km = KMeans(n_clusters=melhor_k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)

    grupos = pd.Series(labels, index=M.index, name="cluster")
    return grupos, {"melhor_k": melhor_k, "scores": relatorio}

# REPRESENTANTES
def representantes_por_cluster(M: pd.DataFrame, grupos: pd.Series):
    scaler = StandardScaler()
    X = scaler.fit_transform(M.values)
    dfX = pd.DataFrame(X, index=M.index, columns=M.columns)

    reps = {}
    for c in sorted(grupos.unique()):
        membros = grupos[grupos == c].index
        Xc = dfX.loc[membros].values
        centroide = Xc.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(Xc - centroide, axis=1)
        rep = membros[np.argmin(dists)]
        reps[c] = rep
    return reps

# SAÍDA
def salvar_resultados(grupos: pd.Series, relatorio: Dict, M: pd.DataFrame) -> None:
    grupos.to_frame().reset_index(names="ticker").to_csv("grupos_kmeans.csv", index=False)
    with open("grupos_kmeans.json", "w", encoding="utf-8") as f:
        json.dump(
            {"melhor_k": relatorio.get("melhor_k"),
             "scores": relatorio.get("scores"),
             "grupos": grupos.to_dict()},
            f, ensure_ascii=False, indent=2
        )
    M.to_csv("assinaturas_agregadas.csv", index=True)
    print("\nArquivos salvos: grupos_kmeans.csv, grupos_kmeans.json, assinaturas_agregadas.csv")

# MAIN
def main():
    print(">> Montando matriz de assinaturas por ticker...")
    M, meta = montar_matriz_assinaturas(TICKERS, FEATURES_STD, AGG_FUNCS)

    print(f"\n>> Escolhendo K por silhouette em {K_RANGE}")
    grupos, rel = escolher_k_e_clusterizar(M, K_RANGE)

    print(f"\nMelhor K escolhido: {rel['melhor_k']}")
    for k, info in rel["scores"].items():
        print(f"K={k} -> silhouette={info['silhouette']:.4f}")

    print("\n== Grupos formados ==")
    for g, membros in grupos.groupby(grupos).groups.items():
        print(f"Cluster {g}: {', '.join(list(membros))}")

    reps = representantes_por_cluster(M, grupos)
    print("\nRepresentantes por cluster (para tunar hiperparâmetros):")
    for c, t in reps.items():
        print(f"- Cluster {c}: {t}")
    pd.Series(reps, name="representante").to_csv("representantes_por_cluster.csv", header=True)
    print("Arquivo salvo: representantes_por_cluster.csv")

    salvar_resultados(grupos, rel, M)
    
if __name__ == "__main__":
    main()