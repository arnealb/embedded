# 🎯 Examen H4 — Quantization

> Uitgewerkte antwoorden op de kernvragen van H4 (Lecture 04). Alle info komt **uitsluitend uit de H4-slides/samenvatting**. Volgorde volgt jouw vragenlijst.
> Vak: Embedded Machine Learning (E061380) — Prof. Adnan Shahid.

---

## 1. Integer-representaties

Een computer kan een geheel getal op verschillende manieren in `n` bits voorstellen. Drie vormen:

| Representatie | Hoe | Bereik | Bijzonderheid |
|---|---|---|---|
| **Unsigned Integer** | elke bit = een macht van 2 | `[0, 2ⁿ−1]` | enkel positief |
| **Signed — Sign-Magnitude** | 1e bit = teken, rest = grootte | `[−2ⁿ⁻¹+1, 2ⁿ⁻¹−1]` | **twee nullen** (`000…0` én `100…0` = 0) ❌ |
| **Signed — Two's Complement** | tekenbit krijgt gewicht **−2ⁿ⁻¹** | `[−2ⁿ⁻¹, 2ⁿ⁻¹−1]` | **één unieke nul**; standaard in hardware ✅ |

- **Unsigned voorbeeld:** `00110001₂ = 2⁵ + 2⁴ + 2⁰ = 49`.
- **Two's complement:** `000…0 = 0` (één nul), `100…0 = −2ⁿ⁻¹` (de meest negatieve waarde).

**Waarom wint two's complement?** Optellen/aftrekken werkt met **dezelfde hardware** voor positieve én negatieve getallen, en er is maar **één representatie van 0**. Sign-magnitude heeft dat dubbele-nul-probleem en vraagt aparte logica.

---

## 2. Fixed-point & Floating-point — formules + oefeningen

### 2.1 Fixed-point (vaste komma)
Een **fixed-point**-getal splitst de bits in een **integer-deel** en een **fractie-deel**, met een **vaste (denkbeeldige) komma** ertussen. Twee manieren om hetzelfde bitpatroon te lezen:
- **Signed Fractional:** elke bit een macht van 2, óók negatieve machten voor de fractie (`… 2¹, 2⁰, 2⁻¹, 2⁻² …`).
- **Scaled Integer:** lees het patroon als gewone integer en vermenigvuldig met een vaste schaalfactor. Bv. `49 × 2⁻⁴ = 49 × 0.0625 = 3.0625`.

→ De **komma ligt vast** → eenvoudig en goedkoop in hardware, maar **beperkt bereik én precisie**.

### 2.2 Floating-point (IEEE 754) — bewegende komma
Het belangrijkste formaat voor neurale netwerken vóór quantization. Een 32-bit float heeft **3 velden**: **1 sign** + **8 exponent** + **23 fraction** (= mantissa/significand).

**De formule voor een normaal getal:**

$$(-1)^{\text{sign}} \times (1 + \text{Fraction}) \times 2^{\text{Exponent} - 127}$$

>  **jouw notitie:** *formule: (1 + mantissa) · 2^(exponent − 127)*

De **127** is de **exponent bias** (`= 2⁸⁻¹ − 1`): zo kan de 8-bit exponent zowel negatieve als positieve machten voorstellen.

**De vuistregel die je moet kennen:** **Exponent Width → Range** · **Fraction Width → Precision.**

| Formaat | Exponent | Fraction | Totaal | Idee |
|---|---|---|---|---|
| **IEEE FP32** | 8 | 23 | 32 | basis |
| **IEEE FP16** | 5 | 10 | 16 | meer precisie, **kleiner bereik** |
| **Google BF16** | 8 | 7 | 16 | **zelfde range als FP32**, minder precisie → slim voor DL |

**FP8 (Nvidia), twee smaken:** **E4M3** (4 exp + 3 mantissa, géén INF, max 448) → meer precisie, voor de **forward pass**; **E5M2** (5 exp + 2 mantissa, wél INF/NaN) → meer bereik, voor de **gradiënten in de backward pass**.

#### Speciale gevallen van de exponent
| Exponent | Fraction = 0 | Fraction ≠ 0 | Formule |
|---|---|---|---|
| `00` (= 0) | ±0 | **subnormal** | `(−1)^sign × Fraction × 2^(1−127)` |
| `01…FE` (1…254) | normal | normal | `(−1)^sign × (1 + Fraction) × 2^(Exp−127)` |
| `FF` (= 255) | **±∞** | **NaN** | — |

> **Subnormaal (exp = 0):** de impliciete 1 **verdwijnt** (`Fraction` i.p.v. `1 + Fraction`) en de exponent wordt `1−127 = −126`. Zo kan `0 = 0 × 2⁻¹²⁶` exact voorgesteld worden én kan je hele kleine getallen halen (kleinste = `2⁻²³ × 2⁻¹²⁶ = 2⁻¹⁴⁹`). Dit heet *gradual underflow*.

### 2.3 Oefening — FP16 lezen
**Vraag:** Wat is `1 10001 1100000000` in decimaal (IEEE FP16, bias = 15)?

**Antwoord:**
- **Sign** = 1 → negatief (−)
- **Exponent** = `10001₂ = 17`, minus bias → `17 − 15 = 2`
- **Fraction** = `1100000000₂ = 0.5 + 0.25 = 0.75`
- **Waarde** = `−(1 + 0.75) × 2² = −1.75 × 4 = `**`−7.0`**

>  **jouw notitie:** *formule: sign · (1 + fraction) · 2^exponent* (exponent bevat al de bias-correctie).

### 2.4 Oefening — decimaal → BF16 schrijven
**Vraag:** Schrijf decimaal **2.5** in Brain Float (BF16, bias = 127).

**Antwoord:**
- `2.5 = 1.25 × 2¹`
- **Sign** = + → `0`
- **Exponent** = `1 + 127 = 128 = 10000000₂`
- **Fraction** = `0.25 = 0100000₂` (7 bits)
- **Binair** = `0 10000000 0100000`

> **De "fraction-truc":** zet de fractie om naar een som van negatieve machten van 2. `0.75 = 2⁻¹ + 2⁻² → 1100000000`. `0.25 = 2⁻² → 0100000`. `0.5 = 2⁻¹ → 1000000000`.

---

## 3. Wat is quantization? (+ uniform vs non-uniform)

> **Quantization is the process of constraining an input from a continuous (or otherwise large) set of values to a discrete set.**

Je dwingt dus een **continue/grote** verzameling waarden in een **kleine, discrete** verzameling. Het verschil tussen een originele waarde en zijn gequantizeerde waarde heet de **quantization error**. (Voorbeelden: een continu sinussignaal wordt naar vaste niveaus afgerond → bij elk sample een foutje; en een foto teruggebracht naar 16 kleuren → je ziet blokken ontstaan = "palettization".)

>  **jouw notitie:** *continu signaal afronden naar vaste stappen: ontstaat quantization error.*

**Uniform vs non-uniform** (de input→output-stappenfunctie):

| | Stappen | Voordeel | Nadeel |
|---|---|---|---|
| **Uniform** | alle stappen **even breed** | eenvoudig, **hardware-vriendelijk** | past niet bij de dataverdeling |
| **Non-uniform** | stappen met **verschillende breedtes** (fijner waar veel waarden liggen) | past beter bij de **dataverdeling** | complexer |

---

## 4. Wat kan je allemaal quantizeren?

Twee tabellen die je moet kennen. Eerst **wát** je quantizeert, dan welk soort quantization wat met *storage* en *computation* doet.

### 4.1 De drie hoofdcategorieën
| Wat | Wat is het? | Wanneer |
|---|---|---|
| **Weights** (parameters) | geleerde gewichten + biases | vóór deployment (**statisch**) |
| **Activations** | tussenresultaten van elke laag | tijdens inferentie (**dynamisch**) |
| **Gradients** | updates tijdens training | tijdens training (enkel bij QAT) |

**Waarom de focus op weights?** 
① ze zijn **statisch** (na training veranderen ze niet → één keer quantizeren),
② ze **nemen veel geheugen in** (domineren de model size → directe geheugenwinst), 
③ **weights → flash, activations → SRAM** (weights quantizeren verkleint je **flash-footprint** → past sneller op een MCU).

**Maar enkel weights volstaat niet altijd:** wil je **volledig int8-rekenen tijdens inferentie** (int8 × int8, geen FPU) en **SIMD** benutten (Cortex-M4 doet 4× int8 tegelijk), dan moet je **ook de activaties** quantizeren. Quantizeer je enkel de weights, dan moet elke int8-weight nog steeds met float32-activaties vermenigvuldigd worden → je houdt de **geheugenwinst** maar verliest veel **snelheidswinst**.

### 4.2 De vier niveaus (storage vs computation)

| Type | Storage | Computation |
|---|---|---|
| **Geen quantization** | Floating-Point weights | Floating-Point arithmetic |
| **K-Means-based** | Integer weights + Floating-Point **codebook** | **Floating-Point** arithmetic |
| **Linear** | **Integer** weights | **Integer** arithmetic |

→ Cruciaal onderscheid: bij **K-means** zijn de gewichten *integers die naar een float-codebook wijzen* (rekenen blijft **float**). Bij **linear** worden zowel opslag als **rekenen integer** → dit is wat echte int8-versnelling oplevert.

---

## 5. K-Means-based Weight Quantization (alles)

*(Deep Compression, Han et al., ICLR 2016.)*

### 5.1 De methode

**Idee:** cluster alle gewichten met **K-means** in een klein aantal groepen. Elk cluster krijgt één **centroid** (gemiddelde). Je slaat dan op:
1. een **cluster index** per gewicht (klein integer, bv. 2-bit),
2. een **codebook** = de lijst centroids (in float).

**Hoe:** neem een 4×4 float-matrix → clusteren → je krijgt een **cluster index** (2-bit ints, waarden 0–3) + een **codebook** (de 4 centroids, bv. `−1.00, 0.00, 1.50, 2.00`). **Reconstructed weights** = vervang elke index door zijn centroid; het verschil met het origineel = **quantization error**.

### 5.2 De bit-/opslagwinst
**Voorbeeld (16 gewichten, 4 clusters, 2-bit index):**
- Origineel: `32 bit × 16 = 512 bit = 64 B`
- Gequantizeerd: indexen `2 bit × 16 = 4 B` + codebook `32 bit × 4 = 16 B` = **20 B** → **3.2× kleiner**
- **Algemeen** (N-bit, M gewichten, M ≫ 2ᴺ): van `32M` bit → `N·M` bit → **≈ 32/N × kleiner** (het codebook `32 × 2ᴺ` is verwaarloosbaar).

**Hoeveel bits heb je nodig?** Conv-lagen ~**4 bits** (gevoeliger), FC-lagen ~**2 bits** (redundanter). Daarboven wint accuracy bijna niets; daaronder stort ze in.

### 5.3 Fine-tuning van de centroids
Na het clusteren regel je de **centroids** bij met gradient descent (`W_new = W_old − η·G`). De truc: de gradiënten worden **per cluster gegroepeerd en opgeteld (reduce)**, en die som past de centroid aan (× learning rate).

> �� **jouw notitie:** *gradients van de loss t.o.v. de gewichten → groepeer die per cluster → update de centroids met de formule.*

In de gewichts-histogrammen zie je: vóór = continu (bell-shaped); ná K-means = **discrete pieken** op de K centroids; ná fine-tuning = pieken licht verschoven naar optimalere posities.

### 5.4 Complementair met pruning

De grafiek (accuracy loss vs model size ratio, AlexNet/ImageNet) toont drie curves:
- **Quantization Only** (geel): zakt onder ~8% modelgrootte.
- **Pruning Only** (paars): zakt onder ~8%.
- **Pruning + Quantization** (rood): blijft vlak tot **~3%** van de originele grootte!

→ **Pruning en quantization zijn complementair** en versterken elkaar: pruning verlaagt het **aantal** gewichten, quantization het **aantal bits per** gewicht. Ze pakken een ander aspect aan en stapelen op.

---

## 6. Deep Compression — samenvatting (schema om te tekenen)

De volledige **Deep Compression**-pijplijn = **3 stappen die elk de accuracy behouden**. Dit is het schema dat je moet kunnen tekenen:

```
                 ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
  origineel ───► │   PRUNING    │ ──► │   QUANTIZATION   │ ──► │ HUFFMAN ENCODING │ ──► model
   (32-bit)      │ Train conn.  │     │ Cluster→Codebook │     │ frequente waarden│   35–49×
                 │ →Prune→Train │     │ →Quantize→Retrain│     │ korter coderen   │   kleiner
                 └──────────────┘     └──────────────────┘     └──────────────────┘
   cumulatief:        9–13×                 27–31×                    35–49×
```

| Stap | Wat | Cumulatieve reductie |
|---|---|---|
| **Pruning** | minder gewichten (Train Connectivity → Prune → Train Weights) | **9× – 13×** |
| **+ Quantization** | minder bits per gewicht (Cluster → Codebook → Quantize → Retrain) | **27× – 31×** |
| **+ Huffman Encoding** | frequente waarden korter coderen (lossless) | **35× – 49×** |

**Huffman coding** = een **lossless** stap bovenop quantization: frequente gewichten → **minder** bits, zeldzame → **meer** bits. Het histogram is niet uniform, dus variabele-lengte-codering wint opslag **zonder accuracy-verlies**.

**Resultaat:** tot **49× kleiner** zonder accuracy-verlies (VGGNet 550 MB → 11.3 MB, acc 88.68% → 89.09%). De open vraag erna: *"Can we make compact models to begin with?"* → leidt naar NAS (Lect07).

---

## 7. Linear Quantization — wat / hoe / formules

In het overzicht: **integer weights én integer arithmetic** → dit levert echte int8-versnelling op.

### 7.1 Wat — de affine mapping

**Linear quantization = een affine mapping van integers naar reële getallen:**

$$r = S \cdot (q - Z)$$

- **r** = de echte floating-point waarde (reconstructed)
- **q** = de opgeslagen quantized **integer**
- **Z** = het **zero point** (een **integer**)
- **S** = de **scale** (een **float**, de stapgrootte)

>  **jouw notitie:** *r = (q − Z) · S — r = floating point waarde; q = opgeslagen integer; Z = zero point (int): zorgt dat r = 0 ook exact kan; S = scale: stapgrootte.*

Een belangrijk doel: de reële **r = 0 moet exact** representeerbaar zijn door een integer → dáárvoor dient het zero point.

### 7.2 Hoe — de scale S berekenen
Trek `r_min = S(q_min − Z)` af van `r_max = S(q_max − Z)` → het zero point Z **valt weg**:

$$S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}$$

>  **jouw notitie:** *het zero point heeft geen invloed op de breedte van het bereik, enkel op waar 0 ligt. S = afstand tussen twee opeenvolgende quantized integerwaarden in floating-point schaal.*

**Voorbeeld:** `r_max = 2.12`, `r_min = −1.08`, 2-bit (`q_max = 1`, `q_min = −2`): `S = (2.12 − (−1.08))/(1 − (−2)) = 3.20/3 = `**`1.07`**.

### 7.3 Hoe — het zero point Z berekenen
Uit `r_min = S(q_min − Z)` los je Z op en **rond af** (Z moet integer zijn):

$$Z = \text{round}\left(q_{min} - \frac{r_{min}}{S}\right)$$

**Voorbeeld:** `Z = round(−2 − (−1.08/1.07)) = round(−2 + 1.009) = round(−0.99) = `**`−1`**.

### 7.4 Volledig uitgewerkt voorbeeld (7×7, 3-bit)
**Opgave:** weight-matrix 7×7, bit width = 3, vind Z en S (gegeven `r_max = 4.30`, `r_min = −4.90`).

>  **jouw notitie (uitwerking):**
> - bit width = 3 → `q_min = −4`, `q_max = 3`
> - `S = (4.30 − (−4.90))/(3 − (−4)) = 9.20/7 = 1.314`
> - `Z = round(−4 − (−4.90/1.314)) = round(−4 + 3.73) = round(−0.27) = 0`

**Resultaat (Jacob et al., CVPR 2018, int8):** ResNet-50 76.4% → 74.9%, Inception-V3 78.4% → 75.4% — klein accuracy-verlies, grote snelheidswinst (int8 haalt bij dezelfde latency hogere accuracy dan float).

---

## 8. Symmetric Quantization

**Symmetric quantization is een vereenvoudigde vorm van linear quantization waarbij `Z = 0`.** De formules worden dan:

$$r = S \cdot q \qquad\qquad S = \frac{|r|_{max}}{q_{max}}$$

>  **jouw notitie:** *Symmetric quantization is een vereenvoudigde vorm van linear quantization waarbij Z = 0, zodat de formule wordt: r = S × q en S = |r|_max / q_max. Dit vermijdt de zero-point berekening en maakt hardware-implementatie eenvoudiger.*

**Waarom is dit handig?** Je hoeft **geen zero point** te berekenen of bij te houden → eenvoudiger en sneller in hardware. De prijs: het bereik wordt symmetrisch rond 0 verondersteld, wat verspilling geeft als de echte verdeling scheef is.

`q_max` voor signed N-bit = `2^(N−1) − 1` (bv. N = 2 → `q_max = 1`; N = 8 → `q_max = 127`).

---

## 9. Post-Training Quantization (PTQ) + granularity + voorbeeld

### 9.1 Wat is PTQ?

**PTQ** verlaagt de numerieke precisie **ná het trainen.** Het model behoudt zijn **originele structuur en parameters**; enkel de numerieke representatie verandert. Pijplijn: `Pre-trained model + Calibration data → Calibration → Quantization → Quantized model`.

**Calibration** = hoe bepaal je het bereik (`r_min`/`r_max`) van de **activaties**, die je niet op voorhand kent? Drie methodes:
- **Max** — neem de absolute max (gevoelig voor outliers).
- **Entropy** — minimaliseer informatieverlies (bv. 99.99%).
- **Percentile** — knip een klein percentage outliers weg.

>  **jouw notitie:** *hoe bepaal je r_min en r_max voor activaties die je niet op voorhand kent? → Via een kleine representatieve dataset (calibration data) door het netwerk te sturen en de activatieverdeling te meten.* (Gewichten ken je exact; activaties hangen af van de input → vandaar calibratie.)

### 9.2 Granularity: per-tensor vs per-channel

- **Per-Tensor:** **één** scale (en zero point) voor de **hele tensor**.
- **Per-Channel:** een **aparte** scale per **channel/rij**.

**Wanneer faalt per-tensor?** Als de gewicht-ranges van verschillende output-channels **>100× verschillen** (een **outlier-channel**), domineert die ene uitschieter de gemeenschappelijke scale → alle andere channels worden grof gequantizeerd. Dit doet vooral de accuracy van **kleine modellen** zakken (grote modellen verdragen per-tensor beter). **Oplossing → per-channel.**

### 9.3 Voorbeeld — per-tensor vs per-channel (symmetrisch, 2-bit)
**Per-tensor:** één `|r|_max = 2.12` voor de hele 4×4 matrix → `S = |r|_max/q_max = 2.12/1 = 2.12` (met `q_max = 2^(2−1) − 1 = 1`).

**Per-channel:** elke rij zijn eigen `|r|_max` en dus eigen scale:
- Rij 0: `|r|_max = 2.09` → S₀ = 2.09
- Rij 1: `|r|_max = 2.12` → S₁ = 2.12
- Rij 2: `|r|_max = 1.92` → S₂ = 1.92
- Rij 3: `|r|_max = 1.87` → S₃ = 1.87

**Resultaat (reconstructie-fout, Frobenius-norm `‖W − S·q‖`):** per-channel = **2.08**, per-tensor = **2.28** → per-channel heeft de **kleinere fout** want elke rij krijgt een scale op maat.

>  **jouw notitie:** *aka deze geeft per rij een nauwkeurigere scale.* (Prijs: meer scales opslaan + hardware die per-channel scales ondersteunt. Voor gewichten is dit standaard; voor activaties blijft per-tensor gangbaar.)

---

## 10. Quantization-Aware Training (QAT)

**Het probleem met PTQ:** het houdt **geen rekening met de impact** van de precisieverlaging — het quantizeert "blind" ná het trainen.

**QAT** past quantization toe op een **pre-trained model** en doet dan **retraining / fine-tuning met de trainingsdata**, zodat het netwerk **leert omgaan met de quantization error**. Tijdens de forward pass wordt de quantization **gesimuleerd**, zodat de gradiënten er rekening mee houden.

>  **jouw notitie:** *PTQ weet niet wat het effect is van die quantisatie — het doet dit blind na het trainen. QAT: quantization toepassen op een pre-trained model → dan hertraining/fine-tuning om de quantization error te compenseren.*

**QAT kan starten van een PTQ-model:** eerst PTQ als **snelle** stap (Quantize → Calibrate → PTQ-model), daarna QAT als **verfijnende** stap (Finetune met training data → QAT-model). Het hoeft dus niet per se van een full-precision model te vertrekken.

---

## 11. Tabel — PTQ vs QAT

>  **jouw notitie:** *PTQ: snel maar mogelijk minder accuraat. QAT: trager maar betere accuracy.*

| Aspect | **QAT** | **PTQ** |
|---|---|---|
| **Accuracy Retention** | minimaliseert accuracy-verlies | kan accuracy-degradatie hebben |
| **Inference Efficiency** | geoptimaliseerd voor low-precision HW (bv. INT8 op TPU) | geoptimaliseerd, maar vereist calibratie |
| **Training Complexity** | retraining met quantization-constraints nodig | **géén** retraining nodig |
| **Training Time** | trager (gesimuleerde quantization in forward pass) | sneller (post hoc) |
| **Deployment Readiness** | best voor modellen gevoelig aan quantization-fouten | snelste manier om te optimaliseren |

**Concrete cijfers (NVIDIA TAO, PeopleNet):** INT8 verdubbelt de inference-snelheid (ResNet18: 762 → 1517 FPS), maar **PTQ kost veel accuracy en QAT herstelt die bijna volledig**:

| Model | FP32 mAP | INT8 + PTQ | INT8 + QAT |
|---|---|---|---|
| PeopleNet-ResNet18 | 78.37 | 59.06 | **78.06** |
| PeopleNet-ResNet34 | 80.2 | 62 | **79.57** |

→ Hét argument voor QAT: **bijna gratis int8-snelheid mét behoud van accuracy.**

---

## 12. Case study — Interference Cancellation (toepassing van alles)

*(Stond niet in je lijst, maar het is het slot van het college: eigen onderzoek van de vakgroep waar álle compressietechnieken samenkomen om een draadloos model edge-ready te maken.)*

**Het probleem:** een gewenst draadloos signaal scheiden van interferentie. De vakgroep deed mee aan de **MIT RF Challenge 2023** (top-5 team). Drie varianten:

| Variant | Architectuur | Resultaat |
|---|---|---|
| **Known IC** | **U-Net** scheidt signaal van interferentie in het **tijd-frequentie-domein** | **63% betere MSE** dan de baseline |
| **Blind IC** | U-Net met **Fast Fourier Convolution (FFC)**-blokken + **LSTM** in de bottleneck | **26.5% MSE-verbetering** + minder MACs |
| **Optimization IC** | U-Net + **quantization** + **depthwise separable convolutions** | zie hieronder |

**Optimization IC — hier komt het hoofdstuk samen:**
- **Depthwise convoluties op het LSTM-model:** slechts **0.72% MSE-degradatie** maar **58.66% minder MACs**.
- **Volledig convolutioneel model:** zelfs **0.63% MSE-verbetering** met **61.10% minder MACs**.
- De gequantizeerde modellen `M1(Q)`/`M2(Q)` zien hun grootte fors dalen (M1: **3.78 → 1.90 MB**) → exact wat quantization belooft.

**AI wireless foundation model (toekomstvisie):** van **task-specific AI** (Fase I) → **transfer learning** (Fase II) → een **task-agnostic wireless foundation model** (Fase III) dat met self-supervised pre-training breed inzetbaar is (technology recognition, NLOS-detectie, environment sensing, human activity).

> **Kort samengevat voor het examen:** dit illustreert dat **quantization + pruning + efficiënte convoluties (depthwise separable)** samen een groot model **drastisch kleiner en lichter** maken (~60% minder MACs, ~2× kleiner) met **verwaarloosbaar of geen** accuracy/MSE-verlies.

---

# ➕ Extra vragen — zaken die je lijst niet expliciet noemt (maar examen-waardig zijn)

**V-A. Waarom is quantization überhaupt nuttig (energie-argument)?**

**Antwoord:** Minder bits → **minder rekenenergie** én **minder data te bewegen**. Een 32-bit float ADD (0.9 pJ) kost ~30× een 8-bit int ADD (0.03 pJ); een 32-bit int MULT (3.1) ~16× een 8-bit int MULT (0.2). En 1 DRAM-toegang ≈ 200× een MAC. Pruning verlaagt het **aantal** operaties, quantization de **kost per** operatie → complementair.

**V-B. Reken-oefening linear quantization (8-bit).** Een laag heeft `r_min = −3.0`, `r_max = 5.0`, gequantizeerd naar 8-bit signed (`q_min = −128`, `q_max = 127`). Bereken S en Z.

**Antwoord:**
- `S = (5.0 − (−3.0))/(127 − (−128)) = 8.0/255 ≈ `**`0.03137`**
- `Z = round(−128 − (−3.0/0.03137)) = round(−128 + 95.6) = round(−32.4) = `**`−32`**

**V-C. Waarom gebruikt een subnormaal getal een andere formule dan een normaal getal?**

**Antwoord:** Een normaal getal gebruikt `(1 + Fraction) × 2^(Exp−127)` — door de verborgen 1 is de kleinste normale waarde `2⁻¹²⁶`, en daaronder ontstaat een "gat" rond 0. Bij exponent = 0 forceert IEEE 754 daarom `Fraction × 2^(1−127)` (**zonder** verborgen 1): zo kan je **0 zelf** exact voorstellen én steeds kleinere getallen tot `2⁻¹⁴⁹` (gradual underflow).

**V-D. Wat is het verschil tussen K-means-based en linear quantization qua storage en computation?**

**Antwoord:** **K-means** → storage = integer-indexen + een **float-codebook**; computation blijft **float** (centroid opzoeken, in float rekenen). **Linear** → storage = **integer** weights; computation = **integer** arithmetic → dit levert echte int8-hardwareversnelling. K-means wint dus vooral **geheugen**, linear wint **geheugen én snelheid**.

**V-E. Waarom heb je calibration data nodig voor activaties maar niet voor gewichten?**

**Antwoord:** Gewichten zijn na het trainen **vaste, gekende getallen** → hun `r_min`/`r_max` ken je exact. **Activaties** hangen af van de **input** en zijn niet op voorhand gekend. Door een kleine representatieve dataset (calibration data) door het netwerk te sturen, meet je de activatieverdeling en schat je hun bereik.

---

> ⚠️ **Disclaimer examen:** het exacte examenformat staat **niet** in de slides. Wat wél bekend is (uit Lecture 01): **theorie 75% / labs 25%**, en *"sommige labo-oefeningen kunnen direct examenstof zijn"* (de lab over pruning/quantization is Lecture 5). Verifieer bij twijfel bij de lesgever.
