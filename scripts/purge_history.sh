#!/bin/bash
# Entfernt urheberrechtlich geschützte Volltexte aus der GESAMTEN Git-History.
#
# Hintergrund: `git rm --cached` + .gitignore schützen nur zukünftige Commits.
# Die Volltexte (Dilorom 2008, poems.txt, master.json mit `text`-Feldern)
# liegen weiterhin in jedem alten Commit auf GitHub und sind dort abrufbar.
#
# ACHTUNG:
#   - Schreibt die History um -> alle Clones müssen neu geklont werden.
#   - Vorher ein lokales Backup des Repos anlegen (cp -r).
#   - Danach: force-push und auf GitHub unter Settings ggf. gecachte
#     Ansichten löschen lassen (GitHub Support), da Blobs sonst über die
#     alte Commit-SHA noch erreichbar sein können.
#
# Voraussetzung: pip install git-filter-repo

set -euo pipefail

echo "Backup vorhanden? (Strg-C zum Abbrechen, Enter zum Fortfahren)"
read -r

git filter-repo \
  --path data/dilorom_soliboeva_tufonhoi_sokit.txt \
  --path data/poems.txt \
  --path data/shahnama.txt \
  --path data/tajik_corpus.txt \
  --path tajik_corpus/corpus/master.json \
  --path tajik_corpus/contributions \
  --invert-paths --force

echo
echo "History bereinigt. Jetzt:"
echo "  git remote add origin git@github.com:moschle/tajikpoemanalyzer.git"
echo "  git push origin --force --all"
echo "  git push origin --force --tags"
echo
echo "Danach master.json lokal behalten (ist jetzt in .gitignore) und den"
echo "öffentlichen Stand über 'python knowledge_export.py features' erzeugen."
