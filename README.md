<p align="center">
  <img src="assets/banner.png" alt="What does physical exertion buy? Athlete load × pitch control in elite football" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-in%20progress-ffd23f" alt="status: in progress">
  <img src="https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13">
  <img src="https://img.shields.io/badge/floodlight-1.2-2a78d6" alt="floodlight 1.2">
  <img src="https://img.shields.io/badge/tracking-25%20Hz-eb6834" alt="25 Hz tracking">
  <img src="https://img.shields.io/badge/MITxT-2.980x%20Sports%20Analytics-A31F34" alt="MITxT 2.980x">
  <img src="https://img.shields.io/badge/raw%20data-not%20included-lightgrey" alt="raw data not included">
</p>

# BraMex · What does physical exertion buy in a football match?

Team project for **MITxT 2.980x Sports Analytics**. We study what physical exertion, positioning and team behaviour are associated with in professional football, using 25 Hz tracking and event data from six matches, and turn that into predictions evaluated on an unseen game.

> **Status:** work in progress. Methods and results will be published here after team review.

## Questions

| Workstream | Owner | Focus |
|---|---|---|
| A · Players and physical load | Leonardo De Andrade | Is individual exertion linked to pitch control? Do load metrics agree? Is fatigue visible? |
| B · Teams: possession, progression, defense | Monica D. Morales-Hernandez | Possession vs pitch control, ball progression, where teams win the ball back |
| C · Chances and possession changes | Pablo Saraiva | What precedes shots and turnovers, and can they be predicted seconds ahead? |

## Data availability

Raw tracking and event data are **not included** in this repository. They were provided to participants of MITxT 2.980x Sports Analytics (MIT) and originate from DFL Deutsche Fußball Liga, published by Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195. Course materials remain the property of MIT. To reproduce the analysis, obtain the data from its original source and set `BRAMEX_DATA_DIR` (see `.env.example`).

## Team BraMex

| | Role |
|---|---|
| **Leonardo De Andrade** | PM & integration lead |
| **Monica D. Morales-Hernandez** | Modeling lead (ML & DS) |
| **Pablo Saraiva** | Evaluation & validation lead |
