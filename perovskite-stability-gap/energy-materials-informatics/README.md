# Energy Materials Informatics

Data-driven projects on energy and optoelectronic materials (solar cells, infrared detectors, thermoelectrics and energy storage), by James Fox. Each project lives in its own folder with Python notebooks that run in Google Colab, plus its own README, figures and sources.

I started this collection in fall 2026. The first project began as my final project for Introduction to AI & Machine Learning at CUNY, a course I'm taking to prepare for a PhD; later projects build on questions it raised.

## Projects

| # | Project | Question | Status |
|---|---|---|---|
| 01 | [Mind the Stability Gap](01-perovskite-stability-gap/) | How often does the perovskite solar cell literature report lifetime, what can machine learning learn from it, and what would it take to replace lead? | Complete: 5 notebooks |
| 02 | [Tin seal economics](02-tin-seal-economics/) | How much can oxygen protection cost before a tin perovskite module loses to lead on the cost of electricity, and is a tighter seal or a cheaper seal plus an oxygen scavenger the better buy? | Planned: scope written |
| 03 | [Seeing Past Silicon](03-seeing-past-silicon/) | Which lead-free materials could absorb infrared light past silicon's edge, how well can machine learning find them, and how do Ag₂Se and InAs quantum dots compare on supply and regulation? | Complete: 1 notebook |
| 04 | [Room-temperature thermoelectrics](04-room-temperature-thermoelectrics/) | How does Ag₂Se compare with bismuth telluride near room temperature, what does the published record leave out, and what does tellurium supply mean for scale-up? | Planned: scope written |

## Ideas in the queue

Questions raised by project 01 that could become later projects (details in [project 01's research ideas](01-perovskite-stability-gap/README.md#research-ideas-parked-for-later)):

- **A pseudo-silver.** Is there a cheap, stable +1 ion, cluster or molecule that can do silver's job in lead-free double perovskites?
- **Protecting Sn²⁺ from the inside.** Which additives slow tin's oxidation without acting as oxidants themselves?
- **Tin recycling.** Lead can be recovered from used perovskite modules; no equivalent study exists for tin.

## Running the notebooks

Open any project folder and click its **Open in Colab** badge, then choose **Runtime → Run all**. Each notebook downloads its own data, so nothing needs to be installed locally.

## About

I hold an M.S. in chemistry from Stony Brook University and spent nearly four years in academic technology transfer, evaluating inventions in materials, energy and medicine (Certified Licensing Professional). I'm interested in lead-free and quantum-dot photovoltaics and in what it takes to move energy materials from the lab to market.

[LinkedIn](https://www.linkedin.com/in/james-m-fox)

## Data and licenses

Each project credits its data sources in its own README. The Perovskite Database is licensed CC BY 4.0; USGS Mineral Commodity Summaries are U.S. government works in the public domain. Code and analysis were developed with help from Claude (Anthropic).
