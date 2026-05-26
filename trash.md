## Wat kan je allemaal quantizeren?

In een neuraal netwerk zijn er **drie hoofdcategorieën** die je kan quantizeren:

| Wat | Wat is het? | Wanneer? |
|---|---|---|
| **Weights** (parameters) | Geleerde gewichten + biases | Vóór deployment (statisch) |
| **Activations** | Tussenresultaten van elke laag | Tijdens inferentie (dynamisch) |
| **Gradients** | Updates tijdens training | Tijdens training (alleen bij QAT/training) |

## Waarom de focus op weights?

Er zijn praktische redenen waarom weight-quantization het eerste en meest voor de hand liggende is:

### 1. Weights zijn statisch
Na training **veranderen weights niet meer**. Je kan ze één keer quantizeren en klaar. Geen complicaties tijdens runtime.

### 2. Weights nemen veel geheugen in
In een typisch CNN domineren weights de model size (denk aan de FC-laag berekening die we deden: miljoenen parameters). Quantizeren = direct geheugenwinst.

### 3. Weights → flash, activations → SRAM
Herinner je nog: weights in flash, activations in SRAM. Door **weights** te quantizeren verklein je je **flash footprint** → past sneller op een MCU.

## Maar pas op: enkel weights quantizeren is niet altijd voldoende

Voor maximale efficiency wil je vaak **ook activations** quantizeren, want dan kan je:

- **Volledig int8-rekenen tijdens inferentie**: int8 × int8 = int (geen FPU nodig)
- **SIMD-instructies benutten**: Cortex-M4 doet 4× int8 ops tegelijk

Als je **enkel weights** quantizeert, moet je elke int8-weight nog steeds terug vermenigvuldigen met float32-activations → je verliest een groot deel van de snelheidswinst. Je krijgt wel nog steeds de **geheugenwinst**.
