#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare, fisher_exact

ROOT = Path(__file__).parent
BOOKS_DIR = ROOT / "books"
SITE_DATA_PATH = ROOT / "site-data.json"

METHODOLOGY_GITHUB_URL = "https://github.com/votre-compte/votre-depot"
VOTE_OPTIONS = ["Le Seigneur des anneaux"]

BOOK_METADATA = {
    "la-ballade-de-pern": {
        "title": "La ballade de Pern",
        "description": "Roman(s) actuellement disponibles pour l'exploration des graphiques de genre.",
    },
    "le-seigneur-des-anneaux": {
        "title": "Le Seigneur des anneaux",
        "description": "Exemple de futur corpus. Ajoutez ses CSV dans le dossier correspondant pour l'activer.",
    },
}

REQUIRED_FILES = {
    "genre_persos": "genre persos.csv",
    "morts": "morts.csv",
    "resume": "resume_genre_personnages.csv",
    "ttr": "ttr.csv",
}


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text


def pct(value: float) -> float:
    return round(float(value) * 100, 1)


def fmt_p(p: float) -> str:
    return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")


def stars_from_p(p: float | None) -> str:
    if p is None:
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_chart_payload(theme_id: str, title: str, subtitle: str, bars: list[dict], note: str, p_value: float | None = None, stat_label: str | None = None) -> dict:
    return {
        "id": theme_id,
        "titre": title,
        "sousTitre": subtitle,
        "type": "horizontal-bar",
        "bars": bars,
        "note": note,
        "significance": {
            "pValue": None if p_value is None else round(float(p_value), 6),
            "label": stat_label,
            "significant": None if p_value is None else bool(p_value < 0.05),
            "stars": stars_from_p(p_value),
        },
    }


def build_table_repartition(path: Path) -> tuple[dict, dict]:
    df = read_csv(path).copy()
    total = df["Total"].sum()
    df["Pourcentage"] = df["Total"] / total
    observed = df["Total"].tolist()
    expected = [total / len(observed)] * len(observed)
    stat, p = chisquare(observed, f_exp=expected)
    bars = [
        {
            "label": str(r["Genre"]),
            "count": int(r["Total"]),
            "percentage": pct(r["Pourcentage"]),
            "value": pct(r["Pourcentage"]),
        }
        for _, r in df.iterrows()
    ]
    note = (
        "Note. Les pourcentages sont calculés sur l'ensemble des personnages recensés. "
        f"Test du χ² contre une répartition égale entre genres observés : χ² = {stat:.2f}, {fmt_p(p)}."
    )
    features = {f"repartition_{slugify(str(r['Genre']))}": pct(r["Pourcentage"]) for _, r in df.iterrows()}
    return build_chart_payload(
        "genre_persos",
        "Répartition des personnages par genre",
        "Effectifs et pourcentages de personnages par genre.",
        bars,
        note,
        p,
        f"χ² = {stat:.2f}",
    ), features


def build_table_morts(path: Path) -> tuple[dict, dict]:
    df = read_csv(path).copy()
    df["Survie"] = df["Total"] - df["Mort"]
    df["Taux_de_mortalite"] = df["Mort"] / df["Total"]
    contingency = df[["Mort", "Survie"]].to_numpy()
    _, p_fisher = fisher_exact(contingency)
    chi2, p_chi, _, _ = chi2_contingency(contingency)
    bars = [
        {
            "label": str(r["Genre"]),
            "count": int(r["Mort"]),
            "total": int(r["Total"]),
            "percentage": pct(r["Taux_de_mortalite"]),
            "value": pct(r["Taux_de_mortalite"]),
        }
        for _, r in df.iterrows()
    ]
    note = (
        "Note. Le taux de mortalité correspond à morts / total dans chaque groupe. "
        f"Test exact de Fisher sur le tableau mort/survie × genre : {fmt_p(p_fisher)}. "
        f"À titre indicatif, χ² = {chi2:.2f}, {fmt_p(p_chi)}."
    )
    features = {f"mortalite_{slugify(str(r['Genre']))}": pct(r["Taux_de_mortalite"]) for _, r in df.iterrows()}
    return build_chart_payload(
        "morts",
        "Mortalité des personnages selon le genre",
        "Taux de mortalité par genre, avec effectifs de morts.",
        bars,
        note,
        p_fisher,
        f"Fisher; χ² = {chi2:.2f}",
    ), features


def build_table_resume(path: Path) -> tuple[dict, dict]:
    df = read_csv(path).copy()
    total = df["n"].sum()
    df["Pourcentage"] = df["n"] / total
    observed = df["n"].tolist()
    expected = [total / len(observed)] * len(observed)
    stat, p = chisquare(observed, f_exp=expected)
    labels = {
        "homme_seul": "Homme seul",
        "femme_seule": "Femme seule",
        "homme_avec_homme": "Homme avec homme",
        "femme_avec_femme": "Femme avec femme",
        "homme_avec_femme": "Homme avec femme",
    }
    bars = [
        {
            "label": labels.get(r["categorie"], r["categorie"]),
            "count": int(r["n"]),
            "percentage": pct(r["Pourcentage"]),
            "value": pct(r["Pourcentage"]),
        }
        for _, r in df.iterrows()
    ]
    note = (
        "Note. Les pourcentages sont calculés sur l'ensemble des configurations observées. "
        f"Test du χ² contre une répartition égale entre catégories : χ² = {stat:.2f}, {fmt_p(p)}."
    )
    features = {f"config_{slugify(labels.get(r['categorie'], r['categorie']))}": pct(r["Pourcentage"]) for _, r in df.iterrows()}
    return build_chart_payload(
        "resume",
        "Configurations de présence des personnages",
        "Répartition des scènes ou configurations selon le genre des personnages présents.",
        bars,
        note,
        p,
        f"χ² = {stat:.2f}",
    ), features


def build_table_ttr(path: Path) -> tuple[dict, dict]:
    df = read_csv(path).copy()
    df["TTR_pct"] = df["TTR"] * 100
    bars = [
        {
            "label": str(r["gender"]).title(),
            "count": int(r["types"]),
            "total": int(r["tokens"]),
            "percentage": round(float(r["TTR_pct"]), 1),
            "value": round(float(r["TTR_pct"]), 1),
        }
        for _, r in df.iterrows()
    ]
    note = (
        "Note. Le TTR (type-token ratio) est présenté en pourcentage. "
        "Aucun test de significativité n'est calculé automatiquement pour ce graphique ; "
        "un protocole de rééchantillonnage ou une modélisation dédiée serait préférable."
    )
    features = {f"ttr_{slugify(str(r['gender']))}": round(float(r["TTR_pct"]), 1) for _, r in df.iterrows()}
    return build_chart_payload(
        "ttr",
        "Diversité lexicale associée aux personnages selon le genre",
        "TTR par genre, avec rappel des types et des tokens.",
        bars,
        note,
        None,
        "Pas de test automatique",
    ), features


def get_book_title(folder_name: str) -> str:
    return BOOK_METADATA.get(folder_name, {}).get("title", folder_name.replace("-", " ").title())


def get_book_description(folder_name: str) -> str:
    return BOOK_METADATA.get(folder_name, {}).get("description", "Corpus disponible sur le site.")


def build_book(book_dir: Path) -> dict | None:
    if not book_dir.is_dir():
        return None
    file_map = {key: book_dir / filename for key, filename in REQUIRED_FILES.items()}
    missing = [str(path.name) for path in file_map.values() if not path.exists()]
    if missing:
        return None
    title = get_book_title(book_dir.name)

    themes = []
    features = {}
    for builder, key in [
        (build_table_repartition, "genre_persos"),
        (build_table_morts, "morts"),
        (build_table_resume, "resume"),
        (build_table_ttr, "ttr"),
    ]:
        chart, chart_features = builder(file_map[key])
        themes.append(chart)
        features.update(chart_features)

    return {
        "nom": title,
        "slug": slugify(book_dir.name),
        "folder": book_dir.name,
        "description": get_book_description(book_dir.name),
        "themes": themes,
        "features": features,
    }


def compute_pca(books: list[dict]) -> dict:
    if not books:
        return {"available": False, "reason": "Aucun livre disponible."}
    feature_names = sorted({k for book in books for k in book.get("features", {}).keys()})
    if len(feature_names) < 2 or len(books) < 2:
        return {
            "available": False,
            "reason": "Au moins deux livres disposant de plusieurs indicateurs sont nécessaires pour calculer une ACP comparative.",
            "featureNames": feature_names,
        }

    X = []
    for book in books:
        row = [float(book.get("features", {}).get(name, 0.0)) for name in feature_names]
        X.append(row)
    X = np.asarray(X, dtype=float)

    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds

    cov = np.cov(Z, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    comps = evecs[:, :2]
    scores = Z @ comps
    total_var = float(np.sum(evals)) if np.sum(evals) else 1.0
    explained = [round(float(v / total_var * 100), 1) for v in evals[:2]]

    arrows = []
    for idx, name in enumerate(feature_names):
        arrows.append({
            "label": name.replace("_", " "),
            "x": round(float(comps[idx, 0]), 4),
            "y": round(float(comps[idx, 1] if comps.shape[1] > 1 else 0.0), 4),
        })

    points = []
    for idx, book in enumerate(books):
        points.append({
            "label": book["nom"],
            "slug": book["slug"],
            "x": round(float(scores[idx, 0]), 4),
            "y": round(float(scores[idx, 1] if scores.shape[1] > 1 else 0.0), 4),
        })

    return {
        "available": True,
        "featureNames": feature_names,
        "explainedVariance": explained,
        "points": points,
        "arrows": arrows,
    }


def main() -> None:
    books = []
    for book_dir in sorted(BOOKS_DIR.iterdir()):
        book = build_book(book_dir)
        if book:
            books.append(book)

    site_data = {
        "books": books,
        "voteOptions": VOTE_OPTIONS,
        "methodologyGithubUrl": METHODOLOGY_GITHUB_URL,
        "siteTitle": "Cartographie de genre dans les œuvres littéraires",
        "themeOrderOptions": [
            {"id": "alpha", "label": "Ordre alphabétique"},
            {"id": "women", "label": "Pourcentage de personnages féminins"},
        ],
        "pca": compute_pca(books),
    }
    SITE_DATA_PATH.write_text(json.dumps(site_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"site-data.json généré avec {len(books)} livre(s).")


if __name__ == "__main__":
    main()
