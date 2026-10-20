# T16 / P20 — Project Progress Tracker
**Single source of truth across all chats in this Project. Read this first. Update it whenever progress is made.**

Last updated: 2026-10-21 01:20 (Claude Code — S1 checkpoint before Wi-Fi drop)

---

## 0. Update protocol (for Claude)
- Read this doc at the start of any project task. Don't re-ask anything answered here.
- After any real progress (file built, decision made, review done, feedback received), update this doc in the same chat / session.
- Edit sections in place. Keep **Now** to ≤5 lines. Add one line to **Changelog** per update (newest first).
- Move items from **Open** → **Done** when finished. Move confirmed choices into **Locked decisions** so they are never re-litigated.
- Unknown = write `?` and list it under **Need from Chaitanya**. Never guess.
- **From Tue 20 Oct, Claude Code on the laptop does the build.** Its copy of this tracker is `C:\Users\Chaitanya\T16_P20\docs\PROGRESS.md`; Chaitanya pastes it into the Project (`claude/PROGRESS.md`) when he switches back to a claude.ai chat. Several chats/sessions may run in parallel → re-read before writing.

**Docs:** `claude/PLAN_weekend.md` (= repo `docs/PLAN.md`, the weekend plan v2) · `claude/R1_references_30.md` · `claude/R1_manuscript_to_3.1.md` · `claude/R1_deck.md` · repo `CLAUDE.md` (Claude Code hand-off) · repo `docs/REFERENCES.md` (verified reference list).

---

## 1. Now
- **Review date moved later (Chaitanya, 20 Oct night)** — RE-PLAN times in CLAUDE.md no longer apply; finish ASAP. **Full queue, no cuts, ε₀ = 2 (locked).**
- **Project folder is `C:\Users\Chaitanya\T16_P20_handoff_final\T16_P20`** (not `C:\Users\Chaitanya\T16_P20` as in CLAUDE.md).
- **S1:** venv + torch 2.11.0+cu128 (sm_75 ✅) · opacus 1.6.0 · flwr **1.41.0** · CPU tests 4/4 PASS · datasets downloaded (offline OK now) · Adult Gate 1 PASS · CIFAR Gate 1 = see §9 item 6.
- **Next:** CIFAR Gate 1 → commit → Chaitanya runs the full sweep overnight in his own PowerShell → S2 demo on CPU → S3 → P2 → S4.
- Journal: parked (Chaitanya decides later).

## 2. Need from Chaitanya
- Extract the zip yourself (folder access from the chat was blocked) · laptop plugged in + sleep off all Tue · Gate 2 ~12:00 (ε₀, default 2) · stop the sweep at 19:00.

---

## 3. Project facts
| Item | Value |
|---|---|
| Course | Computer Networks (theory) project, VIT |
| Team / Project | T16 / Project 20 |
| Title | "A Controlled Comparison of Personalisation Mechanisms for Fairness in Differentially Private Federated Learning" (project title: Privacy-Fairness Trade-off in Personalized FL) |
| Authors (order) | Chaitanya Patil (corresponding, chaitanya.patil2024@vitstudent.ac.in) · Aprark · Tanishq Jain · **Dr. Kalaavathi B** |
| Faculty | Dr. Kalaavathi B — Professor Grade 2, Department of IoT, School of Computer Science and Engineering, VIT; kalaavathi.b@vit.ac.in |
| Members | Chaitanya Patil 24BCE2938 · Aprark 24BCI0052 · Tanishq Jain 24BCI0211 |
| Author consent | ✅ all authors incl. faculty consent to journal submission |
| Catalog (Excel) | Domain **Wireless/FL** · Target venue: SCI journal – trustworthy ML, or top ML conference · Objective: quantify the trade-off; propose a tuning that improves fairness at fixed privacy |
| Publication evidence | 8 = acceptance/DOI · 6 = submission ID + venue · **4 = full manuscript in the venue template** · 2 = abstract + outline |
| SDGs | SDG 9, SDG 10 |
| Review 1 | 24–31 Aug, 40 marks — ✅ went well. Held in a cabin, no projector |
| Final Review | **Wed 21 Oct 2026**, cabin, **10–15 min max**, feedback → changes may be requested afterwards |
| Format | **IEEE journal template** (IEEEtran, two-column; compile on Overleaf) + final PPT |
| GitHub | `chaaaait` · repo `t16-pfl-dp-fairness` (public, to create Tue ~22:00) · git installed |
| Turnitin targets | AI < 20%; similarity as low as possible. Chaitanya does an own-voice pass |
| IEEE AI policy | AI-generated content must be disclosed in the Acknowledgment (system, sections, extent). Placeholder in paper |
| Laptop | Windows x64, **GTX 1650 4 GB**, driver 581.42 (CUDA 13.0); PowerShell |
| Stack | Python 3.12, PyTorch (CUDA), Opacus 1.6, Flower 1.40 |

## 4. Review 2 rubric (60 marks)
| # | Criterion | Max | Full-marks requirement |
|---|---|---|---|
| 1 | Tools | 8 | PyTorch/Flower demonstrated live; scenario config shown; versions + scripts; reproducible |
| 2 | Dataset | 6 | Source, size, class distribution, features, licence; preprocessing; split justified |
| 3 | Implementation | 12 | Architecture diagram; hyperparameter table; baseline + alternative; design justified; Git. Not matching R1 → < 50% → justify every change |
| 4 | Parameters | 8 | All R1 metrics + overhead/cost; units; repeated runs mean ± variance; baseline throughout |
| 5 | Results | 12 | Labelled graphs/tables, error bars, interpretation, failure cases, limitations |
| 6 | Conclusion | 6 | Answers problem, quantified gain, specific future work |
| 7 | Publication | 8 | Accepted 8 · Under review with ID 6 · Full draft in template 4 |

## 5. Methodology / experiments
| ID | Spec | Status |
|---|---|---|
| **M1** | PFL + DP; sweep ε; accuracy spread | ✅ code · ⬜ runs (Q1: FedAvg + fine-tuning, ε ∈ {∞, 8, 4, 2, 1, 0.5, 0.3}) |
| **M2** | Fine-tuning, Per-FedAvg, clustered FL, FedPer at one identical budget | ✅ code · ⬜ runs (Q2, Q3, Q6) |
| **M3** | q-FFL: is the fairness gain paid in accuracy or privacy? | ✅ code · ⬜ runs (Q4, Q7) |
Design D1–D10 (PLAN §3): per-example DP-SGD · identical (ε, δ = 1e-5) via per-run σ calibration · single-process simulator + Flower demo · CIFAR-10 Dirichlet α 0.3, 20 clients + Adult by occupation · FEMNIST dropped · GroupNorm CNN ~0.14 M params · ε₀ = 2 · 3 seeds · loss-threshold MIA · cost metrics.
Old synthetic finding (ε = 0.3: fine-tuning worse than DP global) → re-tested in Q1.

## 6. Manuscript status (`paper/`)
| Section | Status |
|---|---|
| Title, authors, abstract | ✅ draft — results sentence Tue night |
| I Introduction · II Related work + Table 1 | ✅ draft (new prose; Table 1 updated) |
| III Method (fixed §3.1) · IV Setup | ✅ draft (`\todo` numbers filled by Claude Code in S1) |
| V Results · VI Discussion · VII Conclusion | ⬜ Tue 19:45–22:00 (Claude Code, from results only) |
| Acknowledgment (AI disclosure) | ✅ draft — adjust extent at the end |

## 7. References
- **47 references** = 30 surveyed + 17 added (methods, datasets, tools, recent work) → `paper/refs.bib`; status table in `docs/REFERENCES.md` (= Project doc `claude/REFERENCES_verified.md`). 55% from 2021+, 70% from 2020+, 89% of non-foundational from 2021+.
- Fixed: ref FedGraph-Fair author was wrong in R1 ("Zhang, Y.") → **Koffka Khan**, Inf. Sci. 728:122710 (2026); DP-Ditto DOI 10.1109/TMLCN.2025.3528901; PUFFLE pages 4462–4469. ⚠️ spot-check: Rafi DOI, Hu DOI, PUFFLE pages.
- 30 vs 45 explained: R1 prep briefly expanded to 45 (recency), Chaitanya reverted to the original 30; the R1 deck file is from before the revert. Correct survey size = 30.

---

## 8. Locked decisions
- Survey = the original 30 papers; **additional citations allowed for methods, datasets, tools and recent related work** (17 Oct). Keep most references recent (≥ 2021).
- Explanations to Chaitanya stay conceptual, no heavy math.
- Final paper in IEEE journal template; final PPT required.
- Journal submission: Chaitanya decides venue and timing.
- Faculty Dr. Kalaavathi B is co-author (last); Chaitanya corresponding; all authors consent.
- Mostly Chaitanya + Claude do the work; Claude Code runs the laptop work from Tue 20 Oct.
- Writing: original prose; minimise similarity and AI score; IEEE AI disclosure kept.
- Compute: laptop GPU (GTX 1650); Colab only as backup.
- Per-example DP-SGD (not client-level); FEMNIST dropped.

## 9. Open items
| # | Item | Owner | When | Status |
|---|---|---|---|---|
| 1 | Weekend plan | planning chat | Fri | ✅ v2 |
| 2 | Experiment code + CPU tests | planning chat | Fri night | ✅ |
| 3 | CLAUDE.md + README | planning chat | Fri night | ✅ |
| 4 | References verified + refs.bib | planning chat | Sat early | ✅ |
| 5 | Paper §I–IV in IEEE LaTeX | planning chat | Sat early | ✅ |
| 6 | S1: laptop env, data, Gate 1, launch sweep | Claude Code + Chaitanya | Tue 09:30 | 🟡 env ✅ tests ✅ data ✅ (CIFAR-10 60,000 imgs, 20 clients, sizes 771/2,696/5,484 · Adult 44,976 rows (45,222 after missing removed), 12 occupation clients, 976/3,889/6,020) · Adult Gate 1 PASS (~0.4 min/run) · CIFAR Gate 1: all checks PASS except **time_budget** (DP run 7.2–12.0 min at 40 rounds; IFCA slowest; peak GPU ~0.84 GB) → full queue ≈ 14–16 h at 40 rounds; **rounds 40 vs 30 = Chaitanya's call** · setup.tex filled except timing + demo sentence |
| 7 | S2: Flower demo built ✅ (Linux) → test on Windows; env.txt, lock file | Claude Code | Tue day | ⬜ |
| 8 | Gate 2 (ε₀) | Chaitanya + Claude Code | Tue ~12:00 | ⬜ |
| 9 | S3: aggregate, Gate 3, figures (sweep hard stop 19:00) | Claude Code | Tue 19:45 | ⬜ |
| 10 | P2: Results, Discussion, Conclusion, abstract | Claude Code | Tue 22:00 | ⬜ |
| 11 | Turnitin + own-voice pass | Chaitanya | Tue night | ⬜ |
| 12 | S4: GitHub repo + README final + v1.0 | Claude Code + Chaitanya | Tue 22:30 | ⬜ |
| 13 | D1: final deck | claude.ai chat | Tue 23:00 | ⬜ |
| 14 | PDF emailed to faculty | Chaitanya | Tue 23:00 | ⬜ |
| 15 | Demo dry runs + backup video | Chaitanya | Tue eve / Wed AM | ⬜ |
| 16 | Rehearsals + offline dry run | All | Wed AM | ⬜ |

---

## 10. Changelog (newest first)
- 2026-10-21 01:20 — Claude Code S1 checkpoint: venv/CUDA torch OK, CPU tests pass, data downloaded (Python SSL check failed on the CIFAR mirror → fetched with Windows curl, md5 verified), Adult Gate 1 PASS, setup.tex dataset numbers filled. Review moved later; full queue with ε₀ = 2.
- 2026-10-20 01:20 — Re-plan: Sat–Mon lost, nothing ran. Everything moved to **Tue 20 Oct** (S1 08:00 → sweep 09:30 → hard stop 19:00 → results/paper/GitHub/deck → PDF 23:00). Pre-approved cut rules added to CLAUDE.md. Folder access from chat blocked → Chaitanya extracts the zip. Final zip rebuilt.
- 2026-10-17 01:40 — Architecture diagram added (`scripts/architecture.py` → paper Fig. 1, also for the deck). Final hand-off zip `T16_P20_handoff_final.zip` supersedes v1/v2.
- 2026-10-17 01:25 — Planning chat: Flower 1.40 live demo built + tested (SuperLink + 3 SuperNodes over HTTP, DP FedPer, ε per round, Jain); IEEE paper §I–IV written (IEEEtran, compiles, 8 pp before results, figures/tables auto-included); 47 references verified/extended (FedGraph-Fair author corrected); zip v2 delivered.
- 2026-10-17 01:15 — Decisions answered (journal parked; refs: add; faculty details; cabin 10–15 min; GitHub chaaaait). Planning chat wrote + CPU-tested `pfl/` (6 methods + q-FFL, per-example DP-SGD with Opacus, RDP calibration, MIA, group gaps, cost), resumable sweep, aggregate/figures (tested end-to-end on synthetic data), CLAUDE.md, README, PLAN v2.
- 2026-10-16 23:58 — Planning chat: read Excel + R1 docs; wrote PLAN_weekend.md v1; flagged R1 issues; venue shortlist.
- 2026-10-16 23:31 — `nvidia-smi` confirmed GTX 1650 (4 GB), driver 581.42. Compute locked = laptop.
- 2026-10-16 late — Review on Wed 21 Oct. IEEE format + final PPT required. Faculty co-author; all consent. R1 files copied into Project docs. Old M1 code not found.
- 2026-10-16 night — Read project Excel: Review 2 rubric (60 marks). Weekend deadline. Turnitin targets.
- 2026-10-16 — Review 1 went well. Reviews 2 + 3 merged into one Final Review.
- 2026-10-16 — Tracker created from chats of 07 Sep and 10 Sep.
- 2026-09-10 — Manuscript to §3.1 built; reference split; numbering mismatch flagged.
- 2026-09-07 — M1 code built (synthetic data); ε = 0.3 finding.
