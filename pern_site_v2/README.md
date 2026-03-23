# Site statique GitHub Pages — cartographie de genre dans les œuvres littéraires

Ce projet contient :
- un site statique en français compatible avec GitHub Pages ;
- un générateur Python (`generate_site.py`) qui lit des CSV par livre ;
- une page de vote avec prise en charge **Supabase** si configuré, sinon repli local ;
- une architecture prête pour plusieurs livres via des dossiers séparés.

## Structure

```text
books/
  la-ballade-de-pern/
    genre persos.csv
    morts.csv
    resume_genre_personnages.csv
    ttr.csv
  le-seigneur-des-anneaux/
    ... mêmes noms de fichiers ...
```

Un livre est automatiquement ajouté au site si son dossier contient les **4 CSV** requis.

## Lancer en local

Depuis le dossier du projet :

```bash
python generate_site.py
python -m http.server 8000
```

Puis ouvrir dans le navigateur :

```text
http://localhost:8000
```

## Déploiement GitHub Pages

1. Créer un dépôt GitHub.
2. Copier tous les fichiers du projet dans le dépôt.
3. Modifier `generate_site.py` pour mettre votre URL GitHub dans `METHODOLOGY_GITHUB_URL`.
4. Commit + push.
5. Dans GitHub → **Settings** → **Pages** → Source = **GitHub Actions**.
6. Le workflow fourni publiera le site.

## Activer le vote partagé avec Supabase

1. Créer un projet Supabase gratuit.
2. Créer une table `votes_books` avec par exemple :

dans le Supabase SQL Editor.

aller à https://supabase.com
ouvrir le projet
à gauche cliquer sur SQL Editor
cliquer “New query”
copier le code suivant et ajouter cliquer sur run

```sql
create table public.votes_books (
  id bigint generated always as identity primary key,
  book_title text not null,
  created_at timestamp with time zone default now()
);
```

3. Activer une policy d'insertion/lecture publique si vous voulez un site ouvert.
4. Remplir `supabase-config.js` :

```js
window.SUPABASE_CONFIG = {
  url: "https://VOTRE-PROJET.supabase.co",
  anonKey: "VOTRE_CLE_ANON",
  table: "votes_books"
};
```

## Ajouter un nouveau livre

1. Créer un nouveau sous-dossier dans `books/`.
2. Y déposer les 4 CSV avec les noms attendus.
3. Optionnel : compléter `BOOK_METADATA` dans `generate_site.py`.
4. Relancer :

```bash
python generate_site.py
```
