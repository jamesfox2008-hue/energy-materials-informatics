# 04 · Room-temperature thermoelectrics

**Status: planned.** Scope written; notebooks to come.

## The question

Near room temperature, how does Ag₂Se compare with bismuth telluride, the material in today's Peltier coolers and thermoelectric generators? What does the published record leave out that a company would need before choosing either one? And what does tellurium's supply picture mean for scaling up?

## Why this project

[Project 03](../03-seeing-past-silicon/) screened lead-free materials that could absorb infrared light past silicon's edge, and found the shortlist was mostly thermoelectric families: 47% of candidates belong to element systems studied as thermoelectrics, against 17% of all formulas screened. Narrow band gaps are a design rule for both jobs. Ag₂Se sits on both sides, as an infrared quantum dot and as one of the leading tellurium-free thermoelectrics near room temperature.

## Data

Starrydata, an open database of property curves digitized from published thermoelectric papers: 55,493 thermoelectric samples in the 20 October 2026 release, each linked to its paper by DOI. The curves (Seebeck coefficient, electrical conductivity, thermal conductivity and zT against temperature) are published as compressed CSV files on GitHub, so the notebook can download them directly, like Projects 01 and 03.

## Planned analysis

1. **Data audit, in the style of Project 01.** What share of samples report all three properties needed to calculate zT independently, rather than zT alone? How many papers report repeat measurements, heating and cooling cycles, or stability over time? These are the questions a licensee asks first.
2. **Near room temperature.** Distributions of zT between 300 and 350 K by material family: Bi₂Te₃ and its alloys, Ag₂Se and other silver chalcogenides, Mg₃(Sb,Bi)₂, and others. Compare the best reported values with typical ones.
3. **The tellurium question.** Tellurium is a byproduct of copper refining; China produced about 80% in 2025 and has required end-use documentation for exports since February 2025 (USGS). Compare element supply concentration and price across families, and estimate materials cost per watt of cooling or generating capacity.
4. **A simple model, if the data support it.** Predict room-temperature zT from composition, with the same grouped cross-validation and honest error reporting as Projects 01 and 03.

## What this project won't settle

Device-level questions such as contacts, mechanical durability of flexible films, and module manufacturing cost. The database records materials measurements, not devices.

## Starting references

- Katsura, Y. *et al.* Starrydata: from published plots to shared materials data. *Sci. Technol. Adv. Mater.: Methods* (2025), [doi:10.1080/27660400.2025.2506976](https://doi.org/10.1080/27660400.2025.2506976).
- Sofo, J.O. & Mahan, G.D. Optimum band gap of a thermoelectric material. *Phys. Rev. B* 49, 4565 (1994).
- U.S. Geological Survey. *Mineral Commodity Summaries 2026* (tellurium, selenium, silver, bismuth).
