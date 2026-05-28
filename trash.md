Goeie kritische vraag — en hier is de subtiliteit: OFA wordt **niet hardware-specifiek getraind**.

**Wat OFA train wél vs niet:**

- **Wat ze trainen:** de **gewichten** van het super-netwerk. Dat is **hardware-agnostisch** — gewichten zijn gewoon getallen, ze "weten" niks van een GPU of MCU.
- **Wat hardware-specifiek is:** welk **sub-netwerk** je eruit haalt (welke depth, width, kernel sizes). Die keuze hangt af van latency-constraints van de target hardware.

**De flow:**

```
1. Train OFA-netwerk één keer  ──►  gewichten (hw-agnostisch)
                                     │
                                     ├──► sample sub-net A → past op iPhone
                                     ├──► sample sub-net B → past op Pixel  
                                     ├──► sample sub-net C → past op MCU
                                     └──► sample sub-net D → past op GPU
```

Voor elke nieuwe hardware doe je **alleen stap 2 opnieuw**: zoek het beste sub-netwerk binnen de latency-budget van die hardware. De gewichten zijn al getraind, je hoeft niks te hertrainen.
