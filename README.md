# Embedded Machine Learning (E061380) — Samenvattingen

Studiesamenvattingen (in het Nederlands/Vlaams) van de lesslides voor het vak
**Embedded Machine Learning (E061380)** aan UGent / imec (IDLAB), Prof. Adnan Shahid.

Elke samenvatting volgt de slides in volgorde, met afbeeldingen erbij, eigen handgeschreven
notities verwerkt als **📝 jouw notitie**, en eindigt met **Kernpunten**, **Oefenvragen**
(met uitklapbare antwoorden) en een **Begrippenlijst**.

## Samenvattingen

| Hoofdstuk | Onderwerp | Bestand |
|---|---|---|
| H1 | Introduction | [`Samenvatting_H1_Introduction.md`](Samenvatting_H1_Introduction.md) |
| H2 | Overview of Embedded Systems & ML/DL | [`Samenvatting_H2_Embedded_Systems.md`](Samenvatting_H2_Embedded_Systems.md) |
| H3 | Pruning and Sparsity | [`Samenvatting_H3_Pruning_Sparsity.md`](Samenvatting_H3_Pruning_Sparsity.md) |
| H4 | Quantization | [`Samenvatting_H4_Quantization.md`](Samenvatting_H4_Quantization.md) |
| H7 | Neural Architecture Search | [`Samenvatting_H7_Neural_Architecture_Search.md`](Samenvatting_H7_Neural_Architecture_Search.md) |
| H8 | Knowledge Distillation | [`Samenvatting_H8_Knowledge_Distillation.md`](Samenvatting_H8_Knowledge_Distillation.md) |

> Nog te doen: H9 (On-device training & Transfer Learning) — de PDF staat al in de map.

> Noot bij H7: de PDF heet `Lect07_…`, maar de slides zijn intern gelabeld als
> "Lecture – 06 / Neural Architecture Search".

## Hoe lees je deze samenvattingen?

- Open een `Samenvatting_HN_*.md` in een markdown-viewer (de afbeeldingen en de uitklapbare
  oefenvraag-antwoorden renderen dan correct — bv. VS Code preview, of op GitHub).
- De afbeeldingen staan in de bijhorende `samenvatting_img*`-map. **Verplaats de
  samenvattingen en hun afbeeldingenmap samen**, anders breken de links.

## Mapstructuur

```
.
├── Lect0N_*.pdf                  # originele lesslides
├── Samenvatting_HN_*.md          # de samenvattingen
├── samenvatting_img/             # afbeeldingen H1   (slide-NN.png = PDF-paginanummer)
├── samenvatting_img_h2/          # afbeeldingen H2   (slide-NNN.png)
├── samenvatting_img_h3/          # afbeeldingen H3
├── samenvatting_img_h4/          # afbeeldingen H4
├── samenvatting_img_h7/          # afbeeldingen H7   (slide-NNN.png)
├── samenvatting_img_h8/          # afbeeldingen H8   (slide-NN.png)
├── CLAUDE.md                     # instructies/conventies voor Claude Code
└── README.md                     # dit bestand
```

De afbeeldingen zijn met `pdftoppm` uit de PDF's gerenderd; de bestandsnaam `slide-NNN.png`
komt overeen met het **PDF-paginanummer** (let op: dat kan +1 verschillen van het op de slide
gedrukte nummer).
