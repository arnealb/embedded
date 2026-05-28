# 🎯 Examen H8 — Knowledge Distillation (KD)

> Examen-oefenbestand voor H8. Alle info komt **uitsluitend uit de H8-slides/samenvatting**.
> **Opzet:** per onderwerp (= de titel) een kort, gestructureerd antwoord om van te leren. Volgorde volgt jouw lijst.
> Vak: Embedded Machine Learning (E061380) — Prof. Adnan Shahid.

---

## 1. Wat is KD — wat, waarom + het doel van de formele definitie

**Antwoord:**

**Wat:** **Knowledge distillation** is een **trainingstechniek**, géén compressietechniek. Je houdt de **architectuur van het kleine "student"-model vast** en verandert enkel *hoe* je het traint: het kijkt mee met een groot, al getraind **"teacher"-model** en leert niet alleen het echte label, maar ook *hoe de teacher over de data denkt*. Het kleine model wordt dus niet kleiner gemaakt → het wordt **beter getraind**.

**Waarom:**
- Tiny hardware (MCU ~256 kB, MFLOPs) kan grote modellen niet draaien → je hebt een tiny model nodig.
- Maar tiny modellen zijn **moeilijk te trainen** (ze underfitten — zie extra-vraag A).
- Kernvraag van het college: *"How to train a tiny model with the help of a large model?"*
- KD is daarom de **brug** tussen *efficient inference* (PART A) en *efficient training* (PART B): het is de laatste inference-techniek, maar fundamenteel een trainingstechniek.

**Het doel van de formele definitie:** een netwerk zet logits $z_i$ om in klassekansen met de (getemperde) softmax

$$p(z_i, T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}, \qquad i,j = 0,1,\dots,C-1$$

met $C$ = aantal klassen en $T$ = temperatuur (normaal 1). Het **doel van KD** = de **klassekansverdelingen van teacher en student op elkaar afstemmen (align)**.

---

## 2. Dark knowledge

**Antwoord:** De extra informatie die in de **zachte teacher-verdeling** zit en die een hard label niet geeft.

- **Hard label:** kat = 1, hond = 0, auto = 0 → zegt enkel "het is een kat".
- **Soft probabilities (teacher):** kat = 0.85, hond = 0.12, auto = 0.03 → zegt óók: "een kat lijkt een béétje op een hond, en helemaal niet op een auto".
- Die **relaties tussen de klassen** in de niet-winnende kansen = **dark knowledge**.
- Gevolg: de student die de **volledige verdeling** naleert (i.p.v. één 0/1-label) krijgt veel **rijkere feedback per sample** → leert sneller en beter, ook met weinig capaciteit.

---

## 3. Temperatuur — wat, waarom, formule

**Antwoord:**

**Formule** (deel de logits door $T$ vóór de softmax):

$$p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**Wat doet het:** een grotere $T$ trekt de logits naar elkaar toe en **smoothet** de verdeling.
- $T = 1$: gewone, scherpe softmax (bv. Cat 0.982 / Dog 0.017).
- $T = 10$: veel vlakker (bv. Cat 0.599 / Dog 0.401).

**Waarom $T > 1$ tijdens distillatie:** als de teacher heel zelfzeker is, staat bijna alle info in één getal en zijn de kleine kansen (= de klasse-relaties / dark knowledge) nauwelijks zichtbaar. Een hogere $T$ **vergroot die kleine kansen uit**, zodat de student er meer leersignaal uit haalt.
- **Bij inference:** zet $T$ terug op **1**.

*(Waarom werkt dit: voor grote logits gaat de softmax richting een harde one-hot — "winner takes all" — waarbij alle dark knowledge in de bijna-nul-kansen verdwijnt. Delen door $T$ herstelt de verhoudingen tussen de niet-winnende klassen.)*

---

## 4. What to match (+ conventional vs relational KD)

**Antwoord:** Tot nu matchten we de **output-logits**, maar een teacher bevat veel meer kennis. *Welke* signalen je van teacher → student kan overdragen:

| Signaal | Wat match je | Detail |
|---|---|---|
| **Output logits** (standaard) | de output-kansen | distillation loss = cross-entropy of L2 |
| **Intermediate weights** (FitNets) | een tussenlaag | extra **FC-laag $W_r$** om de verschillende teacher/student-**shapes te aligneren** |
| **Intermediate features** | feature maps (activaties) | afstand via **MMD** (Maximum Mean Discrepancy) |
| **Attention maps** | $\partial L / \partial x$ (gradiënt) | wáár het netwerk naar kijkt; sterke modellen lijken hierin op elkaar |
| **Sparsity patterns** | welke neuronen actief zijn na ReLU | binair patroon $\rho(x)=\mathbb{1}[x>0]$ |
| **Relational information** | *relaties* i.p.v. punten | zie hieronder |

**Conventional vs Relational KD (de hoofdtweedeling):**
- **Conventional KD** = *point-to-point*: voor **één input** matcht de student één teacher-grootheid (logits/features/attention/sparsity). → rijen 1–5.
- **Relational KD** = *structure-to-structure*: match de **onderlinge relaties** tussen meerdere samples, niet de absolute features.
  - *tussen lagen:* inner product / **Gram-matrix** $G$, gematcht met L2-loss.
  - *tussen samples:* de vector van **paarsgewijze afstanden** $\psi(s_1,\dots,s_n) = (\lVert s_1-s_2\rVert^2, \dots)$ van teacher én student matchen → de student leert *hoe samples zich tot elkaar verhouden*.

---

## 5. Het nadeel van een vaste grote teacher

**Antwoord:** In de standaard-KD is de teacher **groot én vast (fixed)**. Nadelen:
1. Je moet eerst een **groot model trainen** → duur.
2. Je hebt dat grote model nodig **tijdens élke student-training** → extra geheugen- en rekenkost.
3. Je bent **afhankelijk** van zijn beschikbaarheid en kwaliteit.

→ Dit motiveert **self-** en **online distillation**: distilleren *zonder* aparte, voorgetrainde grote teacher.

---

## 6. Self-distillation / Born-Again Networks

**Antwoord:** Bij **self-distillation** is er **geen apart groot model**: het netwerk distilleert in zichzelf.

**Born-Again Networks** (Furlanello et al., ICML 2018):
1. Train eerst een model $T$ normaal (Step 0).
2. Train dan een **identiek** model $S_1$ dat als teacher het zonet getrainde $T$ gebruikt (classification loss + distillation loss).
3. Herhaal: $S_2$ leert van $S_1$, enz.

Kenmerken:
- **Architectuur:** $T = S_1 = S_2 = \dots = S_k$ (allemaal hetzelfde net).
- **Accuracy:** $T < S_1 < S_2 < \dots < S_k$ — elke generatie wordt een tikje beter, puur door zichzelf als teacher te gebruiken.
- Bonus: je kan de **generaties ensemblen** ($T, S_1, \dots, S_k$ samen) voor nog betere prestaties.

---

## 7. Online distillation: Deep Mutual Learning

**Antwoord:** Bij **online distillation** train je niet eerst de teacher en dan de student, maar **alles tegelijk**.

**Deep Mutual Learning** (Zhang et al., CVPR 2018): twee netwerken $\Theta_1, \Theta_2$ worden **from scratch** samen getraind. Elk netwerk heeft twee loss-termen:

$$\mathcal{L}(S) = \text{CrossEntropy}(S(I), y) + \text{KL}(S(I), T(I))$$
$$\mathcal{L}(T) = \text{CrossEntropy}(T(I), y) + \text{KL}(T(I), S(I))$$

- Elk netwerk leert van: (1) de **ground-truth labels** (cross-entropy) én (2) de **outputverdeling van het andere** netwerk (KL-divergentie).
- De rollen zijn **symmetrisch**: geen "echte" teacher; beide zijn elkaars teacher én student.
- **Geen voorgetraind model nodig**, en zelfs $S = T$ (gelijke architectuur) mag.
- Resultaat: **beide** netwerken verbeteren t.o.v. onafhankelijk trainen (ze maken verschillende fouten → geven elkaar nuttig extra signaal).

---

## 8. Be Your Own Teacher

**Antwoord:** **Be Your Own Teacher** (Zhang et al., ICCV 2019) combineert self- én online distillation **binnen één netwerk**.

- Hang aan elke ResBlock een eigen **tussen-classifier** (bottleneck + FC + softmax).
- De **diepere lagen distilleren de ondiepere** (deep supervision + distillation).
- **Intuïtie:** voorspellingen in latere stadia zijn betrouwbaarder → gebruik die om de vroegere stadia te superviseren.

Drie loss-bronnen per tussen-classifier:
1. **cross-entropy** loss van de labels,
2. **KL-divergentie** loss van de distillatie (diepere → ondiepere),
3. **L2-loss** van de hints (features).

**Voordelen:**
- Consistente verbetering t.o.v. de baseline (CIFAR100).
- De tussen-classifiers (1/4, 2/4, 3/4 van het net) evenaren soms al de baseline → **early exit**: vroeger stoppen met rekenen → hogere **inference-efficiëntie**.

---

## 9. Network Augmentation (NetAug)

**Antwoord:**

**(a) De conventionele aanpak = regularisatie tegen overfitting.** De klassieke trucs maken de taak *moeilijker* of het netwerk *minder zeker*, zodat een groot model de trainingsset niet vanbuiten leert:
- **Data augmentation:** Cutout (blok wegknippen), Mixup (twee beelden + labels lineair mengen), AutoAugment (policies leren).
- **Dropout:** willekeurig neuronen uitzetten; varianten SpatialDropout (hele kanalen) en DropBlock (aaneengesloten blokken).

**(b) Het probleem: regularisatie schaadt tiny modellen.**
- Voor een **groot** model (ResNet50): Mixup/AutoAugment/DropBlock liggen **boven** de baseline. ✓
- Voor een **tiny** model (MobileNetV2-Tiny): diezelfde trucs liggen **onder** de baseline. ✗
- **Waarom?** Een tiny netwerk **underfit** al (te weinig capaciteit) — het probleem is *niet* overfitting. Technieken die de taak nóg moeilijker maken of het netwerk nóg onzekerder maken, duwen het verder de verkeerde kant op. Tiny modellen hebben het **omgekeerde** nodig: **méér signaal, niet minder**.

**(c) Wat is NetAug + hoe lost het dit op?** (Cai et al., ICLR 2022) Draai het idee om: augmenteer niet de **data** (tegen overfitting), maar het **model** (tegen underfitting).
1. Start met het tiny model met gewichten $W_{base}$ → **base supervision** $L(W_{base})$.
2. Bouw rond dat tiny model een **breder "augmented" model**: het tiny model is een **subset** met extra gewichten $W_{aug}$, waarbij $W_{base}$ **gedeeld** wordt (zit in beide).
3. Voeg een **auxiliary supervision**-term toe via dat brede model:

$$L_{aug} = \underbrace{L(W_{base})}_{\text{base supervision}} + \underbrace{L([W_{base}, W_{aug}])}_{\text{auxiliary supervision}}$$

- Per stap sample je een ander breder sub-netwerk; de gradiënt daarvan ($g_{aug}$) stroomt mee terug naar de **gedeelde** $W_{base}$ → het tiny model krijgt **extra leersignaal/capaciteit**, precies wat een underfittend model mist.
- **Bij inference:** gebruik enkel het tiny model ($W_{base}$); het brede deel is louter een trainings-hulpmiddel.

**Belangrijke eigenschappen:**
- Verbetert voor tiny modellen **zowel train- als val-accuracy** (+1.3 % / +1.6 %) → bevestigt dat het underfitting aanpakt.
- **Schaadt** grote modellen (ResNet50: −0.3 % val) → die hebben geen extra capaciteit nodig en gaan overfitten. NetAug is dus **specifiek voor tiny modellen**.
- **Orthogonaal aan KD:** combineerbaar → **NetAug + KD** geeft het beste resultaat.
- Verschil met dropout/KD: dropout *verwijdert* capaciteit (slecht voor tiny), NetAug *voegt toe* via gedeelde gewichten; en je hebt geen voorgetrainde teacher nodig (het brede model wordt mee getraind en weggegooid).

---

# ➕ Extra vragen — zaken die je lijst niet expliciet noemt (maar examen-waardig zijn)

**V-A. Waarom zijn tiny modellen moeilijk te trainen (en wat is hier het probleem — niet overfitting)?**

**Antwoord:**
- Trainingscurves: **ResNet50** (groot) loopt mooi op tot ~82 % / ~76 % (train/val).
- **MobileNetV2-Tiny** (klein) blijft veel lager (~52 % / ~47 %) en — cruciaal — **ook de train-accuracy blijft laag**.
- Bij grote modellen is het gevaar **overfitting** (train ≫ val). Bij tiny modellen is het omgekeerd: te weinig **capaciteit** om zelfs de trainingsset te passen → **underfitting**.
- Gevolg: méér regularisatie zou hier *schaden*; tiny modellen hebben extra **signaal** nodig → dat is precies wat KD (dark knowledge) en NetAug (extra capaciteit) leveren.

**V-B. De twee losses bij standaard KD — schrijf ze en zeg wat elk meet.**

**Antwoord:** De student wordt op **twee** dingen tegelijk getraind:
1. **Classification loss** — verschil tussen student-output en het **echte label** (gewone supervised term).
2. **Distillation loss** — verschil tussen student-output en de **teacher-output**, met $p_t, p_s$ = teacher/student-kansen:
   - **Cross-entropy:** $\mathbb{E}(-p_t \log p_s)$
   - **L2:** $\mathbb{E}(\lVert p_t - p_s \rVert_2^2)$

De **som** van beide wordt geminimaliseerd. De teacher staat vast (wordt niet bijgetraind).

**V-C. Waarom is een attention map ($\partial L/\partial x$) een nuttig distillatie-signaal?**

**Antwoord:**
- Definitie: attention $= \partial L / \partial x$ (gradiënt van de loss naar de feature map). Als $\partial L/\partial x_{i,j}$ **groot** is, beïnvloedt een kleine verstoring op positie $(i,j)$ de output sterk → het netwerk "let" daar sterk op.
- **Sterke modellen lijken op elkaar** in *waar ze naar kijken*: de attention maps van ResNet34 (73 %) en ResNet101 (77 %) lijken sterk op elkaar, een zwak model (NIN, 62 %) heeft heel andere maps.
- Dus: laat de student de teacher-attention overnemen → je duwt hem richting "kijken zoals een goed model".

---

> ⚠️ **Disclaimer examen:** het exacte examenformat staat **niet** in de slides. Wat wél bekend is (uit Lecture 01): **theorie 75% / labs 25%**, en *"sommige labo-oefeningen kunnen direct examenstof zijn"*. Verifieer bij twijfel bij de lesgever.
