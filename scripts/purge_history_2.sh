#!/bin/bash
# ZWEITER Purge-Durchgang: Volltexte, die der erste Lauf nicht erfasst hat.
#
# Beim ersten Durchgang wurden nur die Dateinamen entfernt, die im damals
# geprüften Arbeitsverzeichnis lagen. Ein frischer Klon von GitHub zeigt drei
# weitere Fundstellen in der History:
#
#   1. "data/dilorom_bo anore khamhun.txt"  (alter Name MIT LEERZEICHEN,
#      27.945 Bytes = Volltext Band 2; Commits d222a8b, cf258e6, f54ba11)
#   2. tajik_corpus/exports/contributions_20260118_030113.json
#   3. tajik_corpus/exports/contributions_20260118_030949.json
#      (beide mit den Feldern raw_text und normalized_text für 73 Gedichte;
#       Commits 953fa04, cf258e6)
#
# Voraussetzungen: git-filter-repo, lokales Backup, Repo-Wurzel als Arbeitsverzeichnis.

set -euo pipefail

echo "Backup vorhanden? (Strg-C zum Abbrechen, Enter zum Fortfahren)"
read -r

git filter-repo --force \
  --path "data/dilorom_bo anore khamhun.txt" \
  --path tajik_corpus/exports/contributions_20260118_030113.json \
  --path tajik_corpus/exports/contributions_20260118_030949.json \
  --path tajik_poetry_library/corpus.json \
  --invert-paths

echo
echo "Kontrolle — muss leer bleiben:"
git rev-list --all --objects \
  | grep -iE 'dilorom|contributions_2026' || echo "  sauber"

echo
echo "Jetzt:"
echo "  git remote add origin https://github.com/moschle/tajikpoemanalyzer.git"
echo "  git push origin --force --all && git push origin --force --tags"
