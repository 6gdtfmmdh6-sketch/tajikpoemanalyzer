# Docker Setup & Workflow

## Schnellstart

```bash
cd ~/tajikpoemanalyzer
./deploy.sh
```

Öffne dann: **http://localhost:8501**

---

## Manuelle Installation

### 1. Docker starten
```bash
docker-compose build
docker-compose up -d
```

### 2. Status prüfen
```bash
docker ps
# Sollte "tajik-poetry-analyzer" anzeigen mit Status "Up"
```

### 3. Logs ansehen
```bash
docker logs tajik-poetry-analyzer
```

### 4. Stoppen
```bash
docker-compose down
```

---

## Verzeichnisstruktur

```
tajikpoemanalyzer/
├── data/           → Corpus-Dateien, Lexikon (wird in Container gemountet)
├── exports/        → Exportierte Excel-Dateien (persistiert)
├── uploads/        → Hochgeladene Dateien (persistiert)
├── Dockerfile      → Container-Definition
├── docker-compose.yml → Service-Konfiguration
└── deploy.sh       → Automatisches Deployment
```

---

## Workflow für Analyse

### A) Via Web-Interface (empfohlen)

1. Starte Container: `docker-compose up -d`
2. Öffne http://localhost:8501
3. Lade .txt-Datei hoch (Gedichte mit `*****` getrennt)
4. Analysiere
5. Exportiere als Excel

### B) Direkt im Container

```bash
# Shell im Container öffnen
docker exec -it tajik-poetry-analyzer bash

# Python-Analyse ausführen
python -c "
from analyzer import EnhancedTajikPoemAnalyzer
analyzer = EnhancedTajikPoemAnalyzer()
result = analyzer.analyze_poem('Ман истодаам\nдар имтидоди ғамгини зиндагӣ')
print(result.structural.lines, 'lines')
print('Free verse:', result.structural.is_free_verse)
"
```

### C) Datei von Host analysieren

```bash
# Datei in uploads/ kopieren, dann:
docker exec -it tajik-poetry-analyzer python -c "
from analyzer import EnhancedTajikPoemAnalyzer
analyzer = EnhancedTajikPoemAnalyzer()
with open('/app/uploads/meine_gedichte.txt') as f:
    text = f.read()
# ... Analyse
"
```

---

## Troubleshooting

### Container startet nicht
```bash
docker-compose logs
# Prüfe Fehlermeldungen
```

### Port 8501 belegt
```bash
# Anderen Port verwenden:
# In docker-compose.yml ändern: "8502:8501"
```

### Änderungen an Code übernehmen
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Container komplett zurücksetzen
```bash
docker-compose down -v
docker system prune -f
docker-compose build --no-cache
docker-compose up -d
```

---

## Ohne Docker (Alternative)

Falls Docker Probleme macht:

```bash
cd ~/tajikpoemanalyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run ui.py
```

---

## Ressourcen

- **Streamlit-UI**: http://localhost:8501
- **Logs**: `docker logs -f tajik-poetry-analyzer`
- **Container-Shell**: `docker exec -it tajik-poetry-analyzer bash`
