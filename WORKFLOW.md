# Text Preparation Workflow

## Why Manual Preparation is Necessary

Tajik Cyrillic OCR is unreliable due to:
- Special characters (ӣ, ӯ, ҷ, ҳ, қ, ғ) being misrecognized
- Poetry layout (line breaks, stanzas) being destroyed
- Mixed Arabic/Cyrillic scripts causing confusion

## Recommended: AI-Assisted Transcription

### Step 1: Upload PDF to AI
Upload your PDF to Claude, ChatGPT, or similar.

### Step 2: Request Transcription
Use this prompt:
```
Please transcribe these Tajik poems into clean text.
- Preserve all line breaks exactly as they appear
- Keep the Tajik Cyrillic characters accurate (especially ӣ, ӯ, ҷ, ҳ, қ, ғ)
- Use ***** (five asterisks) as separator between different poems
- Include poem titles if visible
```

### Step 3: Review Output
Check for:
- Correct Tajik characters (not Russian equivalents)
- Proper line breaks
- Clear poem separations

### Step 4: Save and Analyze
```bash
# Save as .txt file
# Run analyzer
streamlit run ui.py
# Or use Python API
```

## Text Format Example

```
ТӮФОНҲОИ СОКИТ

Дар ин хароси чодуйй
чизе бигӯ
бо ин гумкардаҳои хеш

*****

ҲАҶМИ БОРОНИ

Ман истодаам
дар имтидоди ғамгини зиндагӣ
```

## Character Reference

| Tajik | Russian Equivalent | Unicode |
|-------|-------------------|---------|
| Ӣ ӣ | И и + macron | U+04E2, U+04E3 |
| Ӯ ӯ | У у + macron | U+04EE, U+04EF |
| Ҷ ҷ | Ч ч + cedilla | U+04B6, U+04B7 |
| Ҳ ҳ | Х х + descender | U+04B2, U+04B3 |
| Қ қ | К к + descender | U+049A, U+049B |
| Ғ ғ | Г г + stroke | U+0492, U+0493 |

## Troubleshooting

### "No poems found"
- Check that ***** separators are on their own lines
- Ensure file is UTF-8 encoded

### "Many neologisms detected"
- OCR errors may have corrupted words
- Review text for character substitutions (и→ӣ, у→ӯ)

### "Low meter confidence"
- Free verse poems naturally have low confidence
- Check for missing line breaks in original
