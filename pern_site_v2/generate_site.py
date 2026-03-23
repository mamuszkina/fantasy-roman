#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata

import pandas as pd
from scipy.stats import chi2_contingency, chisquare, fisher_exact

ROOT = Path(__file__).parent
BOOKS_DIR = ROOT / "books"
SITE_DATA_PATH = ROOT / "site-data.json"

# Configure here
METHODOLOGY_GITHUB_URL = "https://github.com/votre-compte/votre-depot"
VOTE_OPTIONS = ["Le Seigneur des anneaux"]

# Optional: manually override display labels/titles if folder names are not enough.
BOOK_METADATA = {
    "la-ballade-de-pern": {
        "title": "La ballade de Pern",
        "description": "Roman(s) actuellement disponibles pour l'exploration des tableaux de genre.",
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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_table_repartition(path: Path) -> dict:
    df = read_csv(path)
    total = df["Total"].sum()
    df = df.copy()
    df["Pourcentage"] = df["Total"] / total
    observed = df["Total"].tolist()
    expected = [total / len(observed)] * len(observed)
    stat, p = chisquare(observed, f_exp=expected)
    return {
        "id": "repartition",
        "titre": "Répartition des personnages par genre",
        "colonnes": ["Genre", "Effectif", "Pourcentage (%)"],
        "lignes": [[r["Genre"], int(r["Total"]), pct(r["Pourcentage"])] for _, r in df.iterrows()],
        "note": (
            "Note. Les pourcentages sont calculés sur l'ensemble des personnages recensés. "
            f"Test du χ² contre une répartition égale entre genres observés : χ² = {stat:.2f}, {fmt_p(p)}."
        ),
    }


def build_table_morts(path: Path) -> dict:
    df = read_csv(path)
    df = df.copy()
    df["Survie"] = df["Total"] - df["Mort"]
    df["Taux_de_mortalite"] = df["Mort"] / df["Total"]
    contingency = df[["Mort", "Survie"]].to_numpy()
    odds_ratio, p_fisher = fisher_exact(contingency)
    chi2, p_chi, _, _ = chi2_contingency(contingency)
    return {
        "id": "mortalite",
        "titre": "Mortalité des personnages selon le genre",
        "colonnes": ["Genre", "Morts", "Total", "Taux de mortalité (%)"],
        "lignes": [
            [r["Genre"], int(r["Mort"]), int(r["Total"]), pct(r["Taux_de_mortalite"])]
            for _, r in df.iterrows()
        ],
        "note": (
            "Note. Le taux de mortalité correspond à morts / total dans chaque groupe. "
            f"Test exact de Fisher sur le tableau mort/survie × genre : {fmt_p(p_fisher)}. "
            f"À titre indicatif, χ² = {chi2:.2f}, {fmt_p(p_chi)}."
        ),
    }


def build_table_resume(path: Path) -> dict:
    df = read_csv(path)
    total = df["n"].sum()
    df = df.copy()
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
    return {
        "id": "configurations",
        "titre": "Configurations de présence des personnages",
        "colonnes": ["Catégorie", "Effectif", "Pourcentage (%)"],
        "lignes": [[labels.get(r["categorie"], r["categorie"]), int(r["n"]), pct(r["Pourcentage"])] for _, r in df.iterrows()],
        "note": (
            "Note. Les pourcentages sont calculés sur l'ensemble des configurations observées. "
            f"Test du χ² contre une répartition égale entre catégories : χ² = {stat:.2f}, {fmt_p(p)}."
        ),
    }


def build_table_ttr(path: Path) -> dict:
    df = read_csv(path)
    df = df.copy()
    df["TTR_pct"] = df["TTR"] * 100
    return {
        "id": "ttr",
        "titre": "Diversité lexicale associée aux personnages selon le genre",
        "colonnes": ["Genre", "Tokens", "Types", "TTR (%)"],
        "lignes": [[str(r["gender"]).title(), int(r["tokens"]), int(r["types"]), round(float(r["TTR_pct"]), 1)] for _, r in df.iterrows()],
        "note": (
            "Note. Le TTR (type-token ratio) est présenté en pourcentage. "
            "Aucun test de significativité n'est calculé automatiquement pour ce tableau ; "
            "un protocole de rééchantillonnage ou une modélisation dédiée serait préférable."
        ),
    }


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
    return {
        "nom": title,
        "slug": slugify(book_dir.name),
        "folder": book_dir.name,
        "description": get_book_description(book_dir.name),
        "tableaux": [
            build_table_repartition(file_map["genre_persos"]),
            build_table_morts(file_map["morts"]),
            build_table_resume(file_map["resume"]),
            build_table_ttr(file_map["ttr"]),
        ],
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
    }
    SITE_DATA_PATH.write_text(json.dumps(site_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"site-data.json généré avec {len(books)} livre(s).")


if __name__ == "__main__":
    main()
