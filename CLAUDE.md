# CLAUDE.md

Instructies voor Claude Code wanneer er in deze map gewerkt wordt.

## Wat is dit project?

Nederlandstalige (Vlaamse) **studiesamenvattingen** van de lesslides (PDF's) voor het vak
**Embedded Machine Learning (E061380)** — UGent / imec (IDLAB), Prof. Adnan Shahid.

Elke `LectNN_*.pdf` wordt omgezet naar een `Samenvatting_HN_Onderwerp.md`. De student
(Arne) heeft op de slides met de hand **rode/gekleurde notities** geschreven; die moeten in
de samenvatting terechtkomen.

## Stijl & vormvereisten (BELANGRIJK — zo wil de student het)

1. **Draai niet rond de pot, maar geef wél de nodige context.** Leg het *waarom* en *hoe*
   uit, niet enkel wat op de slide staat (een slide-kopie is niet de bedoeling). Leid
   formules af, geef intuïtie, benoem wat een blok/techniek oplost.
2. **Volg de PDF structureel, in volgorde van de slides.**
3. **Gebruik foto's** voor dingen die visueel makkelijker zijn: `![alt](map/slide-NNN.png)`.
4. **Rode handgeschreven notities** van de student verschijnen als blockquote:
   `> 📝 **jouw notitie:** …` (letterlijk overnemen wat hij schreef, ook spelfouten/afkortingen).
5. **Extra context** waar nuttig, als `**Extra context — …**`.
6. Sluit elke samenvatting af met, in deze volgorde:
   - `# 🔑 Kernpunten om te onthouden` (genummerde lijst)
   - `# 📝 Oefenvragen` (V1, V2, … met antwoord in `<details><summary>Antwoord</summary> … </details>`)
   - `# 📖 Begrippenlijst` (tabel term | betekenis)
7. Begin met een headerblok: vak, college, docent, en de outline-quote (zie bestaande bestanden).
8. Voeg een sectie **"## 0. Waar past dit college in het vak?"** toe met de lecture-tabel.

Neem een bestaande, afgewerkte samenvatting (bv. `Samenvatting_H4_Quantization.md` of
`Samenvatting_H7_Neural_Architecture_Search.md`) als sjabloon voor toon en structuur.

## Bestands- & mapconventies

- Samenvatting: `Samenvatting_HN_Onderwerp.md` (H-nummer = bestandsnummer van de lecture-PDF;
  let op: de **slide-titel** in de PDF kan een ander nummer dragen — zie H7 hieronder).
- Afbeeldingen: per hoofdstuk een eigen map, bestandsnaam `slide-NNN.png`.
  - **De bestandsnaam = het PDF-paginanummer** (niet het op de slide gedrukte nummer).
  - **Padding = aantal cijfers van de laatste pagina** (bv. 76 pagina's → `slide-01`..`slide-76`
    met 2 cijfers; 126 pagina's → `slide-001`..`slide-126` met 3 cijfers).
- Het op de slide *gedrukte* paginanummer loopt vaak **+1 vóór** op het PDF-paginanummer
  (titelpagina's tellen mee in de PDF maar niet in de gedrukte nummering). Reken bij twijfel
  altijd met het PDF-paginanummer = bestandsnaam.

## Workflow om een nieuwe samenvatting te maken

1. Render de PDF naar PNG's (resolutie 110 dpi werkt goed):
   ```
   pdftoppm -png -r 110 "LectNN_Onderwerp.pdf" "samenvatting_img_hN/slide"
   ```
   (Controleer/normaliseer de padding naar het aantal cijfers van de laatste pagina.)
2. **Lees de PDF visueel** met de Read-tool in batches van **max 20 pagina's** (`pages:"1-20"`).
   Dit is essentieel: enkel zo zie je de **rode handgeschreven notities** op de slides.
3. Schrijf de samenvatting in slide-volgorde, met de stijl hierboven; vang **alle** rode
   notities op als `📝 jouw notitie`.
4. **Verifieer** dat elke `slide-NNN.png` waarnaar je verwijst echt bestaat
   (`grep -oE 'map/slide-[0-9]+\.png' bestand.md | while read f; do [ -f "$f" ] || echo MISSING $f; done`).

## Status (per 2026-05-26)

| H | Lecture-PDF | Samenvatting | Afbeeldingenmap |
|---|---|---|---|
| 1 | Lect01 Introduction | ✅ | `samenvatting_img/` |
| 2 | Lect02 Overview Embedded Systems & ML/DL | ✅ | `samenvatting_img_h2/` |
| 3 | Lect03 Pruning and Sparsity | ✅ | `samenvatting_img_h3/` |
| 4 | Lect04 Quantization | ✅ | `samenvatting_img_h4/` |
| 7 | Lect07 Neural Architecture Search | ✅ | `samenvatting_img_h7/` |
| 8 | Lect08 Knowledge Distillation | ✅ | `samenvatting_img_h8/` |
| 9 | Lect09 On-device training & transfer learning | ✅ | `samenvatting_img_h9/` |

**Let op bij H7:** de PDF heet `Lect07_…` en de samenvatting `Samenvatting_H7_…` (zo wou de
student het), maar de **slides zelf zijn gelabeld "Lecture – 06 / Neural Architecture Search"**.
Die discrepantie staat genoteerd in de header van de samenvatting; niet "corrigeren".
