# 02 · Tin seal economics

**Status: planned.** Scope written; notebooks to come.

## The question

How much can oxygen protection cost before a tin perovskite module loses to lead on the cost of electricity? And is it cheaper to buy a tighter seal, or to pair a cheaper seal with an oxygen scavenger?

## Why this project

[Project 01](../01-perovskite-stability-gap/) found that tin perovskite cells in the Perovskite Database lasted a median of 447 hours in nitrogen but only 7 hours in air. A rough estimate there puts the oxygen allowance for a tin module at about 3.7 cm³ per m² over 25 years, or about 0.0004 cm³ per m² per day. That is tighter than OLED displays need. This project asks what meeting that allowance would cost, and whether there is a cheaper way to meet it than a better seal.

## The oxygen budget (same for every scenario)

Oxygen reaching the cells over 25 years must stay under about 3.7 cm³ per m². That is enough to oxidize about 10% of the Sn²⁺ in a 500 nm tin perovskite film, assuming every oxygen molecule oxidizes two tin ions (a worst case).

## Three scenarios

| | Scenario | How oxygen is kept out | Main added cost | Biggest unknown |
|---|---|---|---|---|
| A | Seal only | A barrier that leaks no more than 0.0004 cm³/m²/day | Premium barrier materials and processing | What an OLED-grade barrier costs at solar-module scale |
| B | Seal + permanent scavenger | A cheaper barrier, plus enough scavenger to absorb everything that leaks in over 25 years | Scavenger material and integration | A scavenger that works dry and gives off no oxidizing byproducts |
| C | Seal + replaceable scavenger | A smaller scavenger, swapped every few years | Labor for each swap | Cost per swap, and how often swaps are needed |

All three are compared with two baselines: a lead perovskite module, and simply replacing the whole tin module on a schedule, as modeled by Jean, Woodhouse and Bulović (2019).

### Starting numbers for scenarios B and C

Calcium and iron each absorb roughly 280–300 cm³ of oxygen per gram.

| Barrier leak rate (cm³/m²/day) | Oxygen in over 25 years (cm³/m²) | Scavenger needed (g/m²) |
|---|---|---|
| 0.0004 (scenario A) | 3.7 | none |
| 0.01 | 91 | about 0.3 |
| 0.1 | 913 | about 3 |

## Method sketch

1. **Oxygen model.** Oxygen reaching the cells = barrier leak rate × time − scavenger capacity (never below zero). While the scavenger lasts, the cells see almost no oxygen. That protected period equals capacity ÷ leak rate. This assumes the scavenger sits in the oxygen's path and reacts faster than oxygen can slip past it.
2. **From tin loss to power loss.** Convert oxidized tin into an annual efficiency loss. This is the weakest assumption in the model and will be tested in the sensitivity analysis.
3. **Cost of electricity.** Compute levelized cost of electricity (LCOE) using NREL Annual Technology Baseline inputs (installed cost, operations and maintenance, financing), with a module cost, efficiency and degradation grid like McGovern et al. (2023).
4. **Break-even.** Solve for the most each option can cost and still match the baseline: the barrier premium (A), the scavenger cost (B), and the cost per swap (C).
5. **Sensitivity.** Find which assumptions move the answer most.

## What this project won't settle

Which scavenger chemistry suits tin. A scavenger for tin needs to work without moisture, react faster than Sn²⁺, and leave behind nothing that oxidizes tin. Common choices each fall short somewhere: iron absorbers need moisture, drying oils give off peroxides and aldehydes as they cure, and calcium also reacts with water and releases hydrogen. The economics here will show how good such a scavenger would need to be, and how cheap.

## Related work

- US Patent 11,968,847 claims a perovskite module edge layer containing a desiccant and/or a deoxidizer (iron powder, sulfite, ascorbic acid or oleic acid).
- Schelhas et al. (MRS Spring Meeting 2026) paired an iodine-scavenging POE layer with desiccated PIB edge seals in perovskite modules.

## Starting references

- NREL, *Annual Technology Baseline* (2024).
- L. McGovern, E. C. Garnett, S. Veenstra and B. van der Zwaan, techno-economic study of perovskite PV, *Sustainable Energy & Fuels* (2023). doi:10.1039/d3se00828b
- J. Jean, M. Woodhouse and V. Bulović, "Accelerating Photovoltaic Market Entry with Module Replacement," *Joule* 3, 2824–2841 (2019).
- Project 01, tin case study and oxygen estimate: [01-perovskite-stability-gap](../01-perovskite-stability-gap/).
