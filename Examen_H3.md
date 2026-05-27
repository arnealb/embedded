# 🎯 Examen H3 — Pruning and Sparsity

> Uitgewerkte antwoorden op de kernvragen van H3 (Lecture 03). Alle info komt **uitsluitend uit de H3-slides/samenvatting**. Volgorde volgt jouw vragenlijst.
> Vak: Embedded Machine Learning (E061380) — Prof. Adnan Shahid.

---

## 1. Wat is MLPerf?

**MLPerf** is dé benchmark — de "olympische spelen" — voor **AI-hardware**: een gestandaardiseerde manier om te vergelijken hoe snel verschillende systemen ML draaien (in **samples/seconde**). Het bestaat uit **twee divisies**:

| Divisie | Mag je het model aanpassen? | Wat meet het | Resultaat (voorbeeld) |
|---|---|---|---|
| **Closed Division** | **Nee** — model ligt vast | een **eerlijke, pure hardware-vergelijking** (iedereen draait exact hetzelfde model) | 1029 samples/sec |
| **Open Division** | **Ja** — je mag optimaliseren (prunen, distilleren, quantizeren) | hoe goed je het model + de hardware **samen** kan optimaliseren | 4609 samples/sec |

**De kernboodschap:** door in de Open Division te **prunen + distilleren + quantizeren** haal je een **4.5× speedup** terwijl je **99% van de accuracy** behoudt. Dat is precies waarom compressietechnieken zoals pruning ertoe doen: dezelfde hardware, veel meer doorvoer, nauwelijks accuracy-verlies.

> 📝 **jouw notitie:** *door die pruning, distillation en quantization te doen → 4.5x speedup met geen acc verlies.*

De optimalisatieketen op **BERT-Large**:
- Quantization alleen (QAT) → 1×, 607 MB
- + Pruning → 2.6×, 177 MB
- + Distillation → **4.5×**, ~177 MB (telkens **99% accuracy** behouden)

---

## 2. "Memory is duur" — waarom data verplaatsen het echte probleem is

Dit is **een van de belangrijkste slides van het vak**. De kerngedachte: **data verplaatsen kost veel meer energie dan ermee rekenen.**

Voor elke MAC-bewerking (`a·b + c`) moet je:
1. **weight uit geheugen halen** (b) + **activations uit geheugen halen** (a, c) → **duur**
2. **vermenigvuldigen + optellen** → goedkoop (cheap)
3. **resultaat terug naar geheugen schrijven** → **duur**

De rekenstap zelf is dus spotgoedkoop; het **heen-en-weer halen van data** domineert het energieverbruik.

| Operatie | Energie [pJ] |
|---|---|
| 32-bit int ADD | 0.1 |
| 32-bit float ADD | 0.9 |
| 32-bit Register File | 1 |
| 32-bit int MULT | 3.1 |
| 32-bit float MULT | 3.7 |
| 32-bit SRAM Cache | 5 |
| **32-bit DRAM Memory** | **640** |

**Het cijfer dat je moet kennen:** één toegang tot **DRAM kost ~200×** zoveel als een vermenigvuldiging/optelling (~3 pJ). De energiehiërarchie loopt van goedkoop (register < SRAM, dicht bij de ALU) naar duur (**DRAM**, ver weg) — dit is de befaamde *"memory wall"*.

**Waarom dit pruning motiveert:** minder gewichten = minder data uit DRAM te halen = **minder energie**. Pruning is dus niet enkel goed voor de **opslag**, maar vooral voor het **energieverbruik** — cruciaal voor batterij-gevoede embedded toestellen. Dit ene inzicht motiveert *alle* compressietechnieken (pruning, quantization, distillation, NAS).

> **De kern in vier feiten:** ① DRAM-access ≈ 200× duurder dan een MAC · ② hiërarchie register < SRAM < DRAM met grote sprongen · ③ "data movement is the bottleneck, not computation" · ④ design-implicatie: minimaliseer geheugen-toegang, vooral naar DRAM.

---

## 3. Minder parameters = minder MACs? (Nee — param-reductie ≠ MAC-reductie)

**Kort antwoord: nee, die twee zijn niet hetzelfde.** Je kan veel *parameters* wegsnoeien zonder evenveel *rekenwerk (MACs)* te besparen — en omgekeerd.

| Netwerk | #Param reductie | MAC reductie |
|---|---|---|
| AlexNet | **9×** | 3× |
| VGG-16 | **12×** | 5× |
| GoogleNet | 3.5× | 5× |
| ResNet50 | 3.4× | **6.3×** |
| SqueezeNet | 3.2× | 3.5× |

**Waarom lopen ze uiteen?**
- **Dense (fully-connected) lagen** bevatten de **meeste parameters**, maar relatief weinig MACs.
- **Convolutielagen** bevatten **weinig parameters** (een kleine kernel), maar verbruiken de **meeste MACs** (die kernel schuift over het hele beeld → enorm veel bewerkingen).

> 📝 **jouw notitie:** *param reductie ≠ mac reductie / params in dense layers (weinig params per param) / conv layers domineren de macs / → pruning niet even hard op de compute-zwaarste lagen.*

**De praktische gevolgtrekking:**
- Wil je vooral **geheugen** winnen → pak de **dense lagen** aan (daar zitten de parameters).
- Wil je vooral **snelheid** winnen → pak de **conv-lagen** aan (daar zitten de MACs).

---

## 4. De formules: MACs en parameters berekenen

Twee formule-paren, één voor dense lagen en één voor conv-lagen.

### Dense (fully-connected) laag
Met **N = input-grootte** en **M = output-grootte**:

$$\text{MAC} = N \times M \qquad\qquad \text{Params} = (N \times M) + M$$

De **`+ M`** bij de parameters is de **bias** (één per output-neuron).

### Convolutielaag
Eerst de **output-dimensie** (hoogte/breedte) via:

$$H_{out} = W_{out} = \frac{W - K + 2P}{S} + 1$$

(W = input-grootte, K = kernel-grootte, P = padding, S = stride.)

Dan:

$$\text{MAC}_{conv} = H_{out} \times W_{out} \times C_{out} \times (k \times k \times C_{in})$$

$$\text{Params}_{conv} = (k \times k \times C_{in} + 1) \times C_{out}$$

met `C_out` = aantal filters, `C_in` = aantal input-channels, `k` = kernel-grootte. De **`+1`** is opnieuw de **bias** (één per filter).

> **Let op de formule-asymmetrie:** bij **MAC** tel je de bias **niet** mee (een bias is enkel een optelling, geen multiply-accumulate over een input); bij **parameters** tel je de bias **wél** (het is een geleerd getal dat moet worden opgeslagen). **Flattening** (reshape) kost **0 MACs en 0 params** — er wordt niets geleerd.

---

## 5. Oefening: MACs en parameters van een volledig netwerk

**Netwerk:** Input 128×128×1 → **Conv** (64 filters, 3×3, stride 1, padding 0) → Dense 512 → Dense 256 → Softmax 10.

<img src="samenvatting_img_h3/slide-21.png" alt="Eenvoudig MAC-voorbeeld (enkel dense lagen)" width="82.5%">

<img src="samenvatting_img_h3/slide-23.png" alt="MAC-berekening met conv-laag" width="82.5%">

<img src="samenvatting_img_h3/slide-24.png" alt="Parameter-berekening met conv-laag" width="82.5%">

### Stap voor stap — MACs

**① Conv-laag.** Output-dimensie: `(128 − 3 + 0)/1 + 1 = 126` → output **126×126×64**. Elke filter heeft `3×3×1 = 9` gewichten.

$$\text{MAC}_{conv} = 126 \times 126 \times 64 \times (3\times3\times1) = \mathbf{9\,144\,576}$$

**② Flattening:** **0** MACs (enkel reshape).

**③ Dense 1:** `(126×126×64) × 512 = 520 224 768`
**④ Dense 2:** `512 × 256 = 131 072`
**⑤ Softmax/output:** `256 × 10 = 2 560`

$$\text{MAC}_{totaal} = 9\,144\,576 + 520\,224\,768 + 131\,072 + 2\,560 \approx \mathbf{529.5\ M\ MACs}$$

### Stap voor stap — Parameters

**① Conv-laag:** `((3×3×1) + 1) × 64 = 640` (de +1 is de bias per filter)
**② Flattening:** **0**
**③ Dense 1:** `(126×126×64 × 512) + 512 = 520 225 280`
**④ Dense 2:** `(512 × 256) + 256 = 131 328`
**⑤ Output:** `(256 × 10) + 10 = 2 570`

$$\text{Params}_{totaal} = 640 + 520\,225\,280 + 131\,328 + 2\,570 = \mathbf{520\,359\,818}$$

### 📝 Jouw handgeschreven uitwerking (van papier)

<img src="samenvatting_img_h3/slide-023.png" alt="Handgeschreven oplossing van de MAC-berekening (529,5M MACs)" width="82.5%">

> **Observatie uit de oefening:** **Dense 1** domineert volledig — `520M` van de `529.5M` MACs én `520M` van de `520M` parameters zitten daar. Dat is precies omdat je een grote conv-output (126×126×64 ≈ 1M waarden) **plat slaat** en volledig verbindt met 512 neuronen. Dit illustreert §3: dense lagen = veel parameters; en hier ook veel MACs door de enorme platgeslagen input.

---

## 6. De 5 manieren om te prunen (granularity-spectrum) + voor/nadelen

De vraag *"in welk patroon prunen we?"* heet **granularity**. Het spectrum loopt van **irregular (links)** naar **regular (rechts)**: hoe regelmatiger het patroon, hoe makkelijker te versnellen op echte hardware, maar hoe minder flexibel.

<img src="samenvatting_img_h3/slide-34.png" alt="Granularity spectrum van irregular naar regular" width="82.5%">

> 📝 **jouw notitie:** *like Tetris :)* — de patronen lijken op Tetris-blokjes.

| # | Granulariteit | Wat wordt gepruned | Flexibel? | Versnelbaar? |
|---|---|---|---|---|
| 1 | **Fine-grained / unstructured** | losse gewichten | zeer | slecht |
| 2 | **Pattern-based (N:M)** | vaste patronen (bv. 2:4) | gemiddeld | met HW-support |
| 3 | **Vector-level** | rijen binnen een kernel | minder | beter |
| 4 | **Kernel-level** | hele kernels | weinig | goed |
| 5 | **Channel-level** | hele channels | weinig | zeer goed |

**De centrale trade-off van het hele verhaal: flexibiliteit ↔ versnelbaarheid.**

### 1. Fine-grained / Unstructured
Je mag **elk individueel gewicht** vrij wegknippen (totaal vrij patroon).
- ➕ Zeer flexibel → meestal de **hoogste compressie ratio** (je vindt flexibel de redundante gewichten).
- ➖ Onregelmatig patroon → **moeilijk te versnellen** op gewone hardware; versnelt enkel op **custom hardware** (bv. EIE), **niet zomaar op een GPU**.

> 📝 **jouw notitie:** *totally random → ieder gewicht vrij, moeilijk versnellen.*

### 2. Pattern-based — N:M sparsity
In **elke groep van M** opeenvolgende elementen worden er **N** geprund. Het klassieke geval is **2:4 sparsity** (50%): van elke 4 gewichten houd je er 2.
- ➕ Regelmatig patroon → opslaan als **non-zero values + 2-bit indices**; **NVIDIA A100 (Ampere)** versnelt dit in **hardware (~2×)**.
- ➖ Minder vrij dan fine-grained (je moet het N:M-stramien volgen), heeft hardware-support nodig.
- Behoudt de accuracy goed (BERT-Large F1: 91.9 → 91.9).

> 📝 **jouw notitie:** *in elke groep van M opeenvolgende elementen worden er N gepruned.*

### 3. Vector-level
Prune **rijen binnen een kernel**. Tussenniveau: regelmatiger dan fine-grained, minder flexibel, beter versnelbaar.

### 4. Kernel-level
Prune **hele kernels**. Nog regelmatiger → goed versnelbaar, weinig flexibel.

### 5. Channel-level (channel pruning)
Verwijder **hele output-channels** → een echt **kleiner netwerk** (minder kanalen).
- ➕ **Directe speedup op elke GPU** (het is gewoon een kleinere, dense berekening).
- ➖ **Kleinste compressie ratio** (grofste granulariteit, minst flexibel).

> 📝 **jouw notitie:** *je krijgt netwerk met minder kanalen → het is kleiner netwerk → sneller.*

> **De onderliggende as: fine-grained vs coarse-grained (structured).** Fine-grained = flexibel maar onregelmatig → moeilijk versnellen. Coarse-grained/structured (rijen/kernels/channels) = het resultaat is gewoon een **kleinere dense matrix** → makkelijk te versnellen, maar minder vrijheid.

---

## 7. Hoe kies je *welke* je prunet? (het basisprincipe)

Dit is de vraag **criterion**: gegeven dat je een aantal gewichten/neuronen wil weggooien, **welke** kies je?

<img src="samenvatting_img_h3/slide-49.png" alt="Selectie van te prunen synapsen" width="82.5%">

**Het grondprincipe:** *hoe **minder belangrijk** de verwijderde parameter is, hoe **beter** de prestatie van het geprunde netwerk blijft.* Je gooit dus altijd **eerst de minst belangrijke** parameter weg.

**Voorbeeld:** neuron `y = ReLU(10·x₀ − 8·x₁ + 0.1·x₂)`, dus W = [10, −8, 0.1]. Moet je één gewicht weggooien → **0.1** (kleinste impact op de output). Dit illustreert meteen waarom een **magnitude-based** criterium logisch is: kleine gewichten dragen het minst bij.

De hamvraag wordt dan: **hoe meet je "belangrijkheid"?** Daarvoor zijn er drie methoden (volgende sectie).

---

## 8. Hoe kies je het criterium? — Magnitude / Scaling / APoZ

Er zijn **drie manieren** om te bepalen wat "belangrijk" is. Het volledige overzicht — met de samenvattende tabel en de kernzin — staat in **§6.6 van `Samenvatting_H3_Pruning_Sparsity.md`** ("De drie criteria naast elkaar"). Heel kort:

| Methode | Kijkt naar | "Belangrijk" = | Statisch of data-gedreven |
|---|---|---|---|
| **Magnitude-based** | de **gewichten** `\|W\|` | grote absolute waarde | **statisch** (geen data nodig) |
| **Scaling-based** | een **trainbare γ** per filter | grote aangeleerde γ | aangeleerd tijdens training |
| **APoZ** | de **activaties** (outputs) | weinig nul-outputs na ReLU | **data-gedreven** (data nodig) |

➡️ **Zie het overzicht (§6.6) in de samenvatting voor de volledige uitleg.** Hieronder per methode nog een **verdiepende examenvraag**.

### 8a. Magnitude-based — verdiepend

**Vraag:** Gegeven gewichtsmatrix `W = [[4, −1], [−3, 2]]`. Welke **rij** gaat eruit bij row-wise pruning met (a) de L1-norm en (b) de L2-norm? Wat leert dit je over het verschil tussen L1 en L2?

<details><summary>Antwoord</summary>

- **(a) L1** (`Σ|wᵢ|`): rij 0 = |4|+|−1| = **5**; rij 1 = |−3|+|2| = **5** → **gelijk** → geen voorkeur.
- **(b) L2** (`√Σ|wᵢ|²`): rij 0 = √(16+1) = √17 ≈ **4.12**; rij 1 = √(9+4) = √13 ≈ **3.61** → rij 1 heeft de **kleinste** importance → **rij 1 wordt gepruned**.

**De les:** L1 en L2 kunnen **verschillende keuzes** maken. L2 weegt **grote gewichten zwaarder** door (door het kwadraat), waardoor een rij met één groot gewicht (hier de 4) als belangrijker telt. Beide blijven heuristieken: snel en goedkoop, maar niet gegarandeerd optimaal.
</details>

### 8b. Scaling-based — verdiepend

**Vraag:** Waar komt de schaalfactor γ vandaan, en wat is het concrete verschil met magnitude-based pruning?

<details><summary>Antwoord</summary>

De γ-factoren zijn **trainbare parameters** die aan elk filter (output-channel) hangen en met de output van dat kanaal worden vermenigvuldigd vóór het naar de volgende laag gaat. Ze komen vaak **gratis uit de Batch-Normalization-laag**. Tijdens het trainen **leert** het netwerk welke kanalen ertoe doen; filters met een **kleine γ** worden na de training weggesnoeid (bv. γ=0.10 en γ=0.29 eruit, γ=1.17 en γ=0.82 blijven). Dit is **Network Slimming** (Liu et al., ICCV 2017).

**Het verschil met magnitude:** magnitude kijkt naar de **individuele gewichten binnen** het filter; scaling kijkt naar **één aparte, aangeleerde parameter** die de relevantie van het **hele kanaal** uitdrukt. Daarom is scaling-based een natuurlijke keuze voor **filter/channel pruning** in conv-lagen.
</details>

### 8c. APoZ — verdiepend

**Vraag:** Een channel geeft over een batch deze output-activaties: 4 nullen op 20 waarden in feature map 1, en 6 nullen op 20 in feature map 2. Bereken de APoZ. Is dit channel een goede prune-kandidaat als de andere channels een APoZ ≈ 0.20 hebben? Waarom kan je dit niet zonder data berekenen?

<details><summary>Antwoord</summary>

`APoZ = totaal aantal nullen / totaal aantal waarden = (4+6)/(20+20) = 10/40 = `**`0.25`**.

Hoe **hoger** de APoZ, hoe **minder** het channel bijdraagt (ReLU zet veel outputs op 0 → het neuron is bijna altijd "uit"). `0.25 > 0.20` → dit channel heeft méér nullen dan de andere → **ja, een betere prune-kandidaat**.

Je kan dit **niet zonder data** berekenen, want APoZ meet de **activaties** (de outputs van neuronen). Je moet dus echte input-data door het netwerk sturen om te tellen hoe vaak een neuron nul produceert. Dat is precies het verschil met magnitude/scaling, die naar de **parameters** kijken (statisch, data-onafhankelijk).
</details>

---

## 9. Sensitivity analysis — wat en waarom nodig

**De vraag hierachter (pruning ratio):** *hoeveel* sparsity geef je **per laag**? Het naïeve antwoord — overal even hard prunen (uniform) — is slecht, want **niet elke laag is even gevoelig**. De **eerste laag** is vaak gevoeliger (prune je die te hard, stort de accuracy in), andere lagen zijn redundanter en kan je harder prunen.

<img src="samenvatting_img_h3/slide-65.png" alt="Sensitivity analysis proces" width="82.5%">

**Het proces** (voorbeeld VGG-11 op CIFAR-10):
1. Kies een laag `Lᵢ`.
2. Prune die laag met ratio `r ∈ {0, 0.1, 0.2, …, 0.9}`.
3. Observeer de **accuracy-degradatie ΔAcc** voor elke ratio.
4. Herhaal voor **alle** lagen → je krijgt per laag een **curve** (accuracy vs pruning rate).
5. Kies een **degradatie-threshold T**: trek een horizontale lijn. Waar elke curve die lijn snijdt → dat is de **pruning rate voor die laag**.

<img src="samenvatting_img_h3/slide-75.png" alt="Threshold T bepaalt de pruning rate per laag" width="82.5%">

Zo krijgt een **gevoelige** laag automatisch een **lagere** ratio en een **redundante** laag een **hogere** → een betere accuracy-latency trade-off dan uniform krimpen.

**Waarom nodig?** Omdat uniform prunen geheugen/snelheid verspilt: je kan redundante lagen veel harder prunen zonder accuracy te verliezen, zolang je de gevoelige lagen spaart.

**Beperking (belangrijk!):** sensitivity analysis bekijkt elke laag **apart** en negeert de **interactie tussen lagen** → de gevonden ratios zijn **niet noodzakelijk globaal optimaal**. Dit motiveert **automatic pruning (AMC)**.

---

## 10. AMC — Automatic Pruning (setup en werking)

Manueel per-laag ratios kiezen steunt op **menselijke expertise + trial-and-error** → traag en suboptimaal (en negeert laag-interacties). **AMC (AutoML for Model Compression, He et al., ECCV 2018)** formuleert pruning als een **reinforcement learning (RL)**-probleem dat de ratios **automatisch** zoekt.

<img src="samenvatting_img_h3/slide-81.png" alt="AMC reinforcement-learning setup" width="82.5%">

**De RL-setup:**
- **State** — **11 features** per laag: laag-index, channel numbers, kernel sizes, FLOPs, … (beschrijven de laag).
- **Action** — een **continu getal** = de pruning ratio voor die laag.
- **Agent** — **DDPG** (Deep Deterministic Policy Gradient), gekozen omdat het **continue acties** ondersteunt (een ratio is continu, geen discrete keuze).
- **Reward** — **−Error** (minder fout = hogere reward). Je kan ook **latency** optimaliseren via een vooraf opgebouwde **lookup table (LUT)** — zo hoef je niet telkens echt te meten.

**Resultaten:**
- Op **ResNet50** vindt AMC lagere densities dan de menselijke expert (totaal **29% → 20%**), en ontdekt het **zélf** dat gevoelige lagen (bv. Conv1) minder geprund mogen worden.
- Op **MobileNet** (Samsung Galaxy S7 Edge, TF-Lite): **2× speedup** (119 ms → 59.7 ms) bij nauwelijks accuracy-verlies (70.6% → 70.2%) — beter dan de handmatig geschaalde 0.75 MobileNet.

> **De winst:** AMC optimaliseert het **hele netwerk samen** (niet laag-per-laag zoals sensitivity analysis) en kan **direct latency** als doel nemen.

---

## 11. Fine-tuning — wat en waarom

Na het prunen **daalt de accuracy** (vooral bij een hoge pruning ratio — je hebt net verbindingen weggegooid). **Fine-tuning = de overgebleven gewichten hertrainen** zodat de accuracy zich herstelt. Dat laat je meteen de pruning ratio **hoger pushen** dan zonder fine-tunen.

<img src="samenvatting_img_h3/slide-85.png" alt="Fine-tuning herstelt de accuracy na pruning" width="82.5%">

> **Vuistregel (examen):** gebruik voor fine-tuning een **learning rate van 1/100 tot 1/10** van de originele learning rate. Je wil de overgebleven gewichten **bijregelen**, niet kapotmaken — een te grote LR zou het al geleerde verstoren.

**De drie curves (accuracy-loss vs pruning ratio)** tonen het belang ervan:
- **Pruning alleen** → accuracy stort al snel in.
- **Pruning + fine-tuning** → veel beter, blijft lang vlak.
- **Iterative pruning + fine-tuning** → het best.

**Iteratief prunen** = de target-sparsity **geleidelijk** verhogen (bv. 30% → 50% → 70%), met **telkens een fine-tune-stap** ertussen (de lus `Prune Connections ⇄ Train Weights`). Eén keer aggressief 90% prunen verwijdert te veel tegelijk → de accuracy herstelt niet meer. Geleidelijk + fine-tunen laat het netwerk zich na elke stap aanpassen → veel hogere haalbare compressie: op **AlexNet van 5× naar 9×** bij gelijke accuracy.

---

# ➕ Extra vragen — zaken die je lijst niet expliciet noemt (maar examen-waardig zijn)

**V-A. Wat is pruning eigenlijk, en wat is het verschil tussen synapsen en neuronen prunen?**

<details><summary>Antwoord</summary>

Pruning = het netwerk kleiner maken door onbelangrijke **synapsen (gewichten)** of **neuronen** te verwijderen.
- **Synapsen prunen:** individuele gewichten worden op **0** gezet → het netwerk behoudt **dezelfde structuur**, maar heeft minder actieve verbindingen (fine-grained).
- **Neuronen prunen:** volledige neuronen (met al hun verbindingen) worden verwijderd → het netwerk wordt **structureel kleiner** (coarse-grained = een hele rij in de gewichtsmatrix of een heel channel weg).

Biologische analogie: ook het **menselijk brein** snoeit — van ~2500 synapsen/neuron bij de geboorte, een piek rond ~15 000 op 2–4 jaar, naar ~7 000 bij volwassenen. Eerst veel verbindingen bouwen, dan de overbodige weghalen → efficiënter.
</details>

**V-B. Wat zijn de 5 ontwerpvragen die het hele pruning-college structureren?**

<details><summary>Antwoord</summary>

1. **Motivatie** — waarom efficiënte ML? (modellen groeien sneller dan hardware; memory is duur)
2. **Granularity** — *in welk patroon* prunen we? (fine-grained → channel)
3. **Criterion** — *welke* synapsen/neuronen gooien we weg? (magnitude / scaling / APoZ)
4. **Pruning ratio** — *hoeveel* sparsity per laag? (sensitivity analysis / AMC)
5. **Fine-tune** — hoe herstellen we de accuracy na het prunen?
</details>

**V-C. Waarom is "Today's AI too BIG" en wat is "bridge the gap"?**

<details><summary>Antwoord</summary>

Modellen groeien **exponentieel** (Transformer 0.05B → BERT 0.34B → GPT-2 1.5B → GPT-3 175B → MT-NLG 530B), terwijl het **GPU-geheugen maar lineair** groeit (V100 32GB → A100 80GB). Er ontstaat dus een groeiend **gat** tussen wat modellen nodig hebben en wat de hardware biedt. **"Bridge this gap"** = dat gat dichten met efficiënte technieken zoals **pruning, quantization en distillation**. In de bubble chart (MACs vs accuracy, bubbelgrootte = #params) wil je voor embedded naar de **linkeronderhoek** (weinig MACs, weinig params, toch goede accuracy — zoals MobileNet/ShuffleNet).
</details>

**V-D. Welke hardware-ondersteuning bestaat er voor sparsity?**

<details><summary>Antwoord</summary>

- **NVIDIA A100 (Ampere):** ondersteunt **2:4 sparsity** in de tensor cores → tot **2× peak performance**, ~1.5× gemeten BERT-speedup.
- **Custom accelerators** (Han et al.): **EIE, ESE, SpArch, SpAtten** — speciaal ontworpen om sparse netwerken te versnellen (vooral nodig voor fine-grained pruning, dat niet op een gewone GPU versnelt).
- **AMD/Xilinx Vitis AI Optimizer:** doet automatisch `Prune → Finetune` op een dense FP32-net → reduceert de complexiteit **5× tot 50×** met minimale accuracy-impact.
</details>

**V-E. Hoe ziet de klassieke pruning-workflow eruit (Han et al., 2015)?**

<details><summary>Antwoord</summary>

Drie stappen: **Train Connectivity → Prune Connections → Train Weights.**
1. **Train Connectivity:** train het netwerk eerst volledig. In het gewichts-histogram liggen de meeste gewichten dicht bij 0 (klein), enkele zijn groot (= belangrijk).
2. **Prune Connections:** gooi de kleine gewichten weg → het histogram krijgt een "gat" rond 0.
3. **Train Weights (fine-tune):** hertrain de overgebleven gewichten zodat de accuracy herstelt.

En het beste is dit **iteratief** herhalen (geleidelijk meer prunen + telkens fine-tunen).
</details>

---

> ⚠️ **Disclaimer examen:** het exacte examenformat staat **niet** in de slides. Wat wél bekend is (uit Lecture 01): **theorie 75% / labs 25%**, en *"sommige labo-oefeningen kunnen direct examenstof zijn"* (de lab over pruning/quantization is Lecture 5). Verifieer bij twijfel bij de lesgever.
