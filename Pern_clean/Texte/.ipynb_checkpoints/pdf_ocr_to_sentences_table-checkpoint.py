#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR a PDF, reflow line-wrapped OCR text into running text, then split into sentences.
Output table: 1 sentence = 1 row, with metadata and start/end page.

Outputs:
- .txt (full reflowed text with page markers)
- .csv
- .parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from tqdm import tqdm
import ocrmypdf

# spaCy for sentence segmentation (French)
import spacy


# ---------------------------
# Heuristics / parsing helpers
# ---------------------------

PART_PATTERNS = [
    re.compile(r"^\s*(Première|Deuxième|Troisième|Quatrième|Cinquième)\s+partie\s*$", re.IGNORECASE),
    re.compile(r"^\s*Partie\s+\w+\s*$", re.IGNORECASE),
]

CHAPTER_PATTERNS = [
    re.compile(r"^\s*chapitre\s+(\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(\d+)\s*$"),  # lone number line (common in novels)
]

PRINTED_PAGE_PATTERN = re.compile(r"^\s*(\d{1,4})\s*$")


def normalize_line(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\ufeff", "")  # BOM
    s = s.replace("\u200b", "")  # zero-width space
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def looks_like_header_footer(line: str) -> bool:
    # Conservative: lone digits likely page numbers
    if not line:
        return True
    if PRINTED_PAGE_PATTERN.fullmatch(line) and len(line) <= 4:
        return True
    return False


def update_structure(line: str, current: Dict[str, Optional[str]]) -> None:
    for pat in PART_PATTERNS:
        if pat.match(line):
            current["part"] = line
            current["chapter"] = None
            return

    for pat in CHAPTER_PATTERNS:
        m = pat.match(line)
        if m:
            chap = m.group(1) if m.groups() else line
            current["chapter"] = f"Chapter {chap}"
            return


def extract_printed_page_candidate(lines: List[str]) -> Optional[str]:
    candidates = []
    if lines:
        top = lines[:3]
        bottom = lines[-3:]
        for l in top + bottom:
            l2 = normalize_line(l)
            if PRINTED_PAGE_PATTERN.fullmatch(l2):
                candidates.append(l2)
    return candidates[-1] if candidates else None


# ---------------------------
# OCR + extraction
# ---------------------------

def ocr_pdf_to_searchable_pdf(
    input_pdf: Path,
    output_pdf: Path,
    lang: str = "fra",
    deskew: bool = True,
    rotate_pages: bool = True,
    force_ocr: bool = False,
    jobs: int = 2,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    ocrmypdf.ocr(
        str(input_pdf),
        str(output_pdf),
        language=lang,
        deskew=deskew,
        rotate_pages=rotate_pages,
        force_ocr=force_ocr,
        skip_text=True,
        jobs=jobs,
        output_type="pdf",
        progress_bar=True,
    )


def extract_text_per_page(searchable_pdf: Path) -> List[str]:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out_txt = Path(td) / "out.txt"
        cmd = ["pdftotext", "-layout", str(searchable_pdf), str(out_txt)]
        subprocess.run(cmd, check=True)
        full_text = out_txt.read_text(encoding="utf-8", errors="replace")

    pages = full_text.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return pages


# ---------------------------
# Reflow: turn OCR lines into running text with provenance
# ---------------------------

SENTENCE_END_RE = re.compile(r"[.!?…]+[\"”’»)]?\s*$")
DIALOGUE_DASH_RE = re.compile(r"^\s*[-–—]\s*")  # French dialogue dash lines
HARD_BREAK_RE = re.compile(r"^\s*$")

def should_join_with_space(prev: str, cur: str) -> bool:
    """
    Decide whether to join prev + cur with a space, no space, or paragraph break.
    We handle hyphenation separately.
    """
    if not prev:
        return False

    # If prev ends like a sentence or strong punctuation, likely a new sentence/paragraph can start.
    if SENTENCE_END_RE.search(prev):
        return False

    # If current line looks like a dialogue dash, keep break (often new utterance).
    if DIALOGUE_DASH_RE.match(cur):
        return False

    # Otherwise, typical line-wrap: join.
    return True


def reflow_pages_with_provenance(
    page_texts: List[str],
    drop_headers_footers: bool = True,
    drop_empty: bool = True,
) -> List[Dict[str, Any]]:
    """
    Produce a sequence of "chunks" (running text segments) with page provenance.
    Each chunk is a piece of continuous text we’ll later sentence-split.

    Returns list of dicts:
      { "text": ..., "start_pdf_page": i, "end_pdf_page": j }
    """
    chunks: List[Dict[str, Any]] = []

    current_text = ""
    current_start_page = 1
    current_end_page = 1

    for i, ptxt in enumerate(tqdm(page_texts, desc="Reflowing pages")):
        pdf_page = i + 1
        raw_lines = ptxt.splitlines()
        lines = [normalize_line(x) for x in raw_lines]

        # filter
        filtered: List[str] = []
        for l in lines:
            if drop_empty and not l:
                continue
            if drop_headers_footers and looks_like_header_footer(l):
                continue
            filtered.append(l)

        # page -> running text
        prev_line = ""
        for l in filtered:
            # paragraph break marker from blank lines would have been removed if drop_empty=True
            # If you keep empty lines, treat them as hard breaks.
            if HARD_BREAK_RE.match(l):
                # flush current chunk
                if current_text.strip():
                    chunks.append(
                        {"text": current_text.strip(), "start_pdf_page": current_start_page, "end_pdf_page": current_end_page}
                    )
                current_text = ""
                prev_line = ""
                current_start_page = pdf_page
                current_end_page = pdf_page
                continue

            # hyphenation fix: if previous ends with "-" and current starts with a letter, merge without space and drop hyphen
            if current_text.endswith("-") and l and re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ]", l):
                current_text = current_text[:-1] + l
            else:
                if current_text and should_join_with_space(prev_line, l):
                    current_text += " " + l
                elif current_text:
                    # keep a newline as a soft paragraph/utterance boundary marker
                    current_text += "\n" + l
                else:
                    current_text = l

            prev_line = l
            current_end_page = pdf_page

        # At page boundary:
        # If we are mid-sentence, we want to continue into next page.
        # If the current_text ends with sentence-final punctuation, we can flush now for cleaner provenance.
        if current_text.strip() and SENTENCE_END_RE.search(current_text.splitlines()[-1]):
            chunks.append({"text": current_text.strip(), "start_pdf_page": current_start_page, "end_pdf_page": current_end_page})
            current_text = ""
            prev_line = ""
            current_start_page = pdf_page + 1
            current_end_page = pdf_page + 1

    # flush any remaining
    if current_text.strip():
        chunks.append({"text": current_text.strip(), "start_pdf_page": current_start_page, "end_pdf_page": current_end_page})

    return chunks


# ---------------------------
# Structure tracking + sentence splitting
# ---------------------------

def build_sentences_table(
    page_texts: List[str],
    book_title: str,
    keep_headers_footers: bool = False,
    keep_empty: bool = False,
) -> pd.DataFrame:
    # Load spaCy French pipeline
    nlp = spacy.load("fr_core_news_sm")
    # Ensure sentence boundaries exist (they do in the model, but keep it explicit)
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    # First, we’ll also track part/chapter by scanning normalized lines in order (page by page).
    current = {"part": None, "chapter": None}
    page_struct: Dict[int, Dict[str, Optional[str]]] = {}

    for i, ptxt in enumerate(page_texts):
        pdf_page = i + 1
        raw_lines = ptxt.splitlines()
        norm_lines = [normalize_line(x) for x in raw_lines]
        for l in norm_lines:
            if not l:
                continue
            update_structure(l, current)
        page_struct[pdf_page] = {"part": current["part"], "chapter": current["chapter"]}

    # Reflow text into chunks with start/end page provenance
    chunks = reflow_pages_with_provenance(
        page_texts=page_texts,
        drop_headers_footers=not keep_headers_footers,
        drop_empty=not keep_empty,
    )

    rows: List[Dict[str, Any]] = []
    sent_id = 0

    for ch in tqdm(chunks, desc="Sentence splitting"):
        text = ch["text"]
        start_p = int(ch["start_pdf_page"])
        end_p = int(ch["end_pdf_page"])

        # Choose metadata from the sentence start page (most consistent)
        part = page_struct.get(start_p, {}).get("part")
        chapter = page_struct.get(start_p, {}).get("chapter")

        # Split sentences with spaCy
        doc = nlp(text.replace("\n", " "))  # newlines are typically not meaningful after reflow
        for s in doc.sents:
            sent = s.text.strip()
            if not sent:
                continue
            sent_id += 1
            rows.append(
                {
                    "sentence_id": sent_id,
                    "book_title": book_title,
                    "part": part,
                    "chapter": chapter,
                    "start_pdf_page": start_p,
                    "end_pdf_page": end_p,
                    "text": sent,
                }
            )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pdf", required=True, type=Path)
    ap.add_argument("--book_title", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=Path)

    ap.add_argument("--lang", default="fra", help="Tesseract language(s), e.g. 'fra' or 'fra+eng'")
    ap.add_argument("--force_ocr", action="store_true")
    ap.add_argument("--jobs", default=2, type=int)

    ap.add_argument("--keep_headers_footers", action="store_true")
    ap.add_argument("--keep_empty", action="store_true")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    searchable_pdf = args.out_dir / (args.input_pdf.stem + ".searchable.pdf")
    out_txt = args.out_dir / (args.input_pdf.stem + ".reflowed.txt")
    out_csv = args.out_dir / (args.input_pdf.stem + ".sentences.csv")
    out_parquet = args.out_dir / (args.input_pdf.stem + ".sentences.parquet")

    print(f"[1/4] OCR -> searchable PDF: {searchable_pdf}")
    ocr_pdf_to_searchable_pdf(
        input_pdf=args.input_pdf,
        output_pdf=searchable_pdf,
        lang=args.lang,
        force_ocr=args.force_ocr,
        jobs=args.jobs,
    )

    print("[2/4] Extract text per page (pdftotext)")
    page_texts = extract_text_per_page(searchable_pdf)

    print("[3/4] Write reflowed text preview")
    # For the txt, we keep page markers so you can sanity-check alignment.
    # (Sentences table uses start/end page columns instead.)
    reflow_chunks = reflow_pages_with_provenance(
        page_texts=page_texts,
        drop_headers_footers=not args.keep_headers_footers,
        drop_empty=not args.keep_empty,
    )
    with out_txt.open("w", encoding="utf-8") as f:
        for ch in reflow_chunks:
            f.write(f"\n\n=== PDF PAGES {ch['start_pdf_page']}–{ch['end_pdf_page']} ===\n")
            f.write(ch["text"])
            f.write("\n")

    print("[4/4] Build sentence table and export")
    df = build_sentences_table(
        page_texts=page_texts,
        book_title=args.book_title,
        keep_headers_footers=args.keep_headers_footers,
        keep_empty=args.keep_empty,
    )

    df.to_csv(out_csv, index=False, encoding="utf-8")
    df.to_parquet(out_parquet, index=False)

    print("\nDone.")
    print(f"Sentences: {len(df):,}")
    print(f"CSV: {out_csv}")
    print(f"Parquet: {out_parquet}")
    print(f"Reflowed TXT: {out_txt}")


if __name__ == "__main__":
    main()