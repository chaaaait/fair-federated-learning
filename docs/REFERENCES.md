# References — verified list (17 Oct 2026)

**47 references** = the **30 surveyed papers** from Review 1 + **17 added** for the methods, datasets and tools we
actually use and for recent related work. BibTeX: `paper/refs.bib` (keys below). The paper numbers them in citation
order (IEEE style), so old R1 numbers no longer apply.

## Recency
| Measure | Count | Share |
|---|---|---|
| Published 2021 or later | 26 / 47 | **55%** |
| Published 2020 or later | 33 / 47 | **70%** |
| Non-foundational works from 2021+ (excluding seminal methods, datasets, tools) | 25 / 28 | **89%** |

Foundational/seminal (exempt from recency, label them if asked): Dwork 2006, Abadi 2016, Shokri 2017, Mohri 2019,
Zhu 2019, Bagdasaryan 2019, Li 2020 (SPM), Fallah 2020, Dinh 2020, Li 2020 (q-FFL) — the R1 ten — plus McMahan 2017
(FedAvg), Arivazhagan 2019 (FedPer), Krizhevsky 2009 (CIFAR-10), Becker & Kohavi 1996 (Adult), Mironov 2017 (RDP),
Yeom 2018 (loss-threshold attack), Jain 1984 (Jain's index), Beutel 2020 (Flower) and Yousefpour 2021 (Opacus).

## Fixes to the Review 1 list
| R1 | Problem | Fix |
|---|---|---|
| [30] FedGraph-Fair | Author "Zhang, Y., et al." was never verified — **it was wrong** | **Koffka Khan** (single author), *Information Sciences*, vol. 728, art. 122710, 2026 (found via a paper that cites it; ScienceDirect PII S0020025525008461) |
| [11] DP-Ditto | No DOI | DOI **10.1109/TMLCN.2025.3528901** (from arXiv 2506.14251); authors confirmed |
| [28] PUFFLE | Pages "462–4469" in the refs file vs "4462–4469" in the manuscript | 4462–4469 kept; DOI not confirmed online → omitted |
| [13] pFedMe | Author written "T Dinh, C." | Canh T. Dinh |
| [27] Padala | DOI unconfirmed | Cited by venue + pages (ICONIP 2021, CCIS 1517, pp. 692–699), no DOI |

## Status of every entry
✅ = checked online this session · ☑️ = standard, well-known metadata (DOI pattern matches publisher) · ⚠️ = please
spot-check (2 min each)

**Survey (30)**
| Key | Paper | Status |
|---|---|---|
| li2020federated | Li et al., IEEE SPM 2020 | ☑️ |
| fallah2020personalized | Per-FedAvg, NeurIPS 2020 | ☑️ |
| dinh2020personalized | pFedMe, NeurIPS 2020 | ☑️ |
| li2021ditto | Ditto, ICML 2021 | ☑️ |
| huang2021personalized | FedAMP, AAAI 2021 | ✅ (AAAI article 16960) |
| tan2023towards | Tan et al., TNNLS 2023 | ☑️ |
| mohri2019agnostic | Agnostic FL, ICML 2019 | ☑️ |
| li2020fair | q-FFL, ICLR 2020 | ☑️ |
| duan2021self | Duan et al., TPDS 2021 | ✅ (Aug) |
| du2021fairness | Du et al., SDM 2021 | ✅ (SIAM DOI page) |
| ezzeldin2023fairfed | FairFed, AAAI 2023 | ✅ (AAAI article 25911) |
| shi2024towards | Shi et al., TNNLS 2024 | ✅ (DOI on arXiv) |
| rafi2024fairness | Rafi et al., Information Fusion 2024 | ⚠️ DOI 10.1016/j.inffus.2023.102198 |
| dwork2006calibrating | Dwork et al., TCC 2006 | ☑️ |
| abadi2016deep | Abadi et al., CCS 2016 | ☑️ |
| hu2020personalized | Hu et al., IEEE IoT-J 2020 | ⚠️ DOI 10.1109/JIOT.2020.2991416 |
| wei2020federated | Wei et al., TIFS 2020 | ☑️ |
| kim2021federated | Kim et al., ICASSP 2021 | ✅ (Aug) |
| bagdasaryan2019differential | Bagdasaryan et al., NeurIPS 2019 | ☑️ |
| bietti2022personalization | Bietti et al., ICML 2022 | ✅ (PMLR page seen) |
| chen2023privacy | Chen et al., ACM CSUR 2023 | ☑️ |
| padala2021federated | Padala et al., ICONIP 2021 | ✅ authors (arXiv); no DOI |
| corbucci2024puffle | PUFFLE, ECAI 2024 | ✅ authors (arXiv); ⚠️ pages |
| esipova2023disparate | Esipova et al., ICLR 2023 | ☑️ |
| zhao2025convergence | DP-Ditto, IEEE TMLCN 2025 | ✅ |
| khan2026fedgraph | FedGraph-Fair, Information Sciences 2026 | ✅ (corrected) |
| shokri2017membership | Shokri et al., S&P 2017 | ☑️ |
| zhu2019deep | Deep Leakage from Gradients, NeurIPS 2019 | ☑️ |
| nasr2019comprehensive | Nasr et al., S&P 2019 | ☑️ |
| yin2021comprehensive | Yin et al., ACM CSUR 2021 | ☑️ |

**Added (17)** — why each one is needed
| Key | Paper | Why | Status |
|---|---|---|---|
| mcmahan2017communication | FedAvg, AISTATS 2017 | our baseline algorithm | ☑️ |
| arivazhagan2019federated | FedPer, arXiv 2019 | mechanism we implement | ☑️ |
| ghosh2022efficient | IFCA, IEEE Trans. Inf. Theory 2022 | mechanism we implement | ✅ |
| collins2021exploiting | FedRep, ICML 2021 | recent parameter-decoupling work | ✅ (title/venue) |
| krizhevsky2009learning | CIFAR-10 tech report | dataset | ☑️ |
| becker1996adult | Adult, UCI 1996 | dataset | ☑️ |
| ding2021retiring | Retiring Adult, NeurIPS 2021 | Adult's limitations; future work | ☑️ |
| li2022federated | Non-IID silos benchmark, ICDE 2022 | Dirichlet partition protocol | ✅ (title/venue) |
| yousefpour2021opacus | Opacus, arXiv 2021 | tool (per-example grads, accountant) | ☑️ |
| beutel2020flower | Flower, arXiv 2020 | tool (live demo) | ☑️ |
| mironov2017renyi | Rényi DP, CSF 2017 | accountant | ☑️ |
| yeom2018privacy | Yeom et al., CSF 2018 | the loss-threshold attack we run | ☑️ |
| carlini2022membership | LiRA, S&P 2022 | stronger attack (limitations) | ✅ (IEEE Xplore 9833649) |
| jain1984quantitative | Jain's index, DEC 1984 | fairness metric | ☑️ |
| noble2022differentially | DP-SCAFFOLD, AISTATS 2022 | DP on heterogeneous FL | ✅ (PMLR page) |
| tramer2021differentially | Tramèr & Boneh, ICLR 2021 | small models under DP-SGD | ☑️ |
| kairouz2021advances | Kairouz et al., FnTML 2021 | client- vs record-level DP | ☑️ |

## What Chaitanya should do (10 min, any time before submission)
1. Open the ⚠️ entries on Google Scholar / the publisher page and confirm the DOI or pages.
2. Make sure you can say one sentence about each *added* paper (the "Why" column is that sentence).
3. The deck says **30 papers surveyed** (the literature review); the paper cites 47 in total.
