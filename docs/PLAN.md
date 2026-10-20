> ⚠️ **Schedule superseded (Tue 20 Oct 01:20).** Sat–Mon were lost; everything now runs on **Tue 20 Oct**, review **Wed 21 Oct**. The live schedule, sweep hard stop (19:00) and pre-approved cut rules are in `CLAUDE.md` → **RE-PLAN**. Design (§3), queue, gates and rubric map below are unchanged.

# T16 / P20 — Weekend Plan 🗓️ (v2)
**Fri 16 Oct night → Sun 18 Oct · Final Review Wed 21 Oct (cabin, 10–15 min + feedback)**
v2 written Sat 17 Oct ~01:00 by the planning chat. Same text lives in the claude.ai Project as `claude/PLAN_weekend.md`.
Owners: **C** = Chaitanya · **Cl** = Claude (claude.ai chat) · **CC** = Claude Code on the laptop · **A** = Aprark · **T** = Tanishq

---

## 0. Bottom line
- 🌙 **Tonight (claude.ai chat) — done:** experiment code written + tested on CPU · Flower live demo built + tested (Linux) · IEEE LaTeX paper (Intro → Setup, compiles, 8 pages before results) · 47 references verified/extended · `CLAUDE.md` hand-off → one zip.
- ☀️ **Sat ~08:30 → switch to Claude Code** (desktop app → Code tab → Local → folder `C:\Users\Chaitanya\T16_P20`). Claude Code owns all laptop work from then on.
- 🚀 **Sweeps start Sat ~10:00** on the GTX 1650: ~147 runs, 12–20 h → done by Sun ~06:30 at worst.
- ✂️ **Scope:** CIFAR-10 (full ε sweep) + Adult (group fairness). FEMNIST dropped → limitation / future work.
- 🔧 **DP level fixed:** per-example DP-SGD (Opacus). Client-level DP with 20 clients drowns the model in noise.
- 📮 **Journal:** Chaitanya handles it later. We deliver the full paper in the IEEE journal template (= 4/8 marks shown; 6/8 once a submission ID exists).
- ✅ **Done by Sun 21:00:** paper PDF, deck, GitHub repo, demo + backup video.

---

## 1. ⚠️ Problems found in the Review 1 material → status
| # | Problem | Fix | Status |
|---|---|---|---|
| 1 | R1 promised 3 datasets × 6 ε × 4 mechanisms — not possible on a 4 GB GPU in 2 days | FEMNIST dropped; full sweep on CIFAR-10; Adult at ∞ and ε₀ | ✅ in queue + paper |
| 2 | R1 contradicts itself on DP level (manuscript §3.1 + deck slide 7 = client-level; deck slide 8 = per-example) | Per-example DP-SGD (D1); §3.1 rewritten | ✅ code + paper |
| 3 | R1 deck says 45 papers / 71% recent; paper says 30 / 85% | Final deck: 30 surveyed (+ method/data/tool citations) | ⬜ deck (Sun) |
| 4 | Ref [30] author unverified; [11], [30] DOIs missing; 20 DOIs pattern-guessed; [28] pages differ | Verified online, fixed in `paper/refs.bib` | ✅ see `docs/REFERENCES.md` |
| 5 | FedPer, IFCA, CIFAR-10, Adult not in the 30 refs | Added (+ other method/data/tool and recent refs) | ✅ |
| 6 | Corresponding author mismatch | Chaitanya = corresponding author | ✅ |
| 7 | Abstract 237 words; IEEE cap 250 | Rewritten; results sentence added Sun | ⬜ Sun |
| 8 | IEEE requires an AI-use disclosure (which AI, which sections, how much) | Placeholder in the Acknowledgment — keep it when submitting | ✅ placeholder |
| 9 | All authors must approve before any submission | Send PDF to Dr Kalaavathi B Sun 16:30 | ⬜ Sun |
| 10 | Changes from R1 must be justified ("does not match R1 methodology" loses marks) | "Changes since Review 1" (slide + paper §IV) | ✅ paper · ⬜ deck |

---

## 2. Decisions (answered 17 Oct 00:08)
| # | Decision | Answer |
|---|---|---|
| 1 | Venue | Parked — Chaitanya handles the journal. Paper in the **IEEE journal template** (IEEEtran, two-column) |
| 2 | References | **Add** method/dataset/tool citations + relevant recent papers; keep most references recent (≥ 2021) |
| 3 | Authors | Chaitanya Patil · Aprark · Tanishq Jain · **Dr. Kalaavathi B** (Professor Grade 2, Dept of IoT, School of Computer Science and Engineering, VIT; kalaavathi.b@vit.ac.in). Corresponding: Chaitanya |
| 4 | Review format | **Cabin, 10–15 min max**, feedback given, changes may be requested after → present from the laptop |
| 5 | GitHub | `chaaaait` · git installed · repo `t16-pfl-dp-fairness` (public) |

---

## 3. 🔒 Design (implemented in `pfl/`)
| # | Item | Choice | Why |
|---|---|---|---|
| D1 | DP level | **Per-example DP-SGD** in every client (Opacus per-example gradients, clip, Gaussian noise; RDP accountant) | Client-level DP needs 1000s of clients per round. Matches R1 deck slide 8 |
| D2 | Identical budget | Every client answers the same queries; σ is calibrated per run so every mechanism ends at the **same (ε, δ = 10⁻⁵)**. Loss reports (q-FFL weights, IFCA cluster choice) are clipped (3.0), noised (30·σ) and charged | "Per unit of privacy" becomes a fair test |
| D3 | Engine | Fast single-process PyTorch simulator for all sweeps; **Flower** (1 SuperLink + 3 SuperNodes talking HTTP on localhost) for the live demo | Flower's simulation mode needs Ray (experimental on Windows); 147 runs need speed |
| D4 | Data | **CIFAR-10**: train+test pooled (60k), Dirichlet α = 0.3, 20 clients, each split 70/10/20. **Adult**: 1 client per occupation (≥ 500 rows → ~12 clients), sensitive attribute = sex | CIFAR-10 lets us dial heterogeneity; Adult is the only one with group labels → DP/EO gaps |
| D5 | Methods | Baselines: **FedAvg**, **Local-only**. Mechanisms: **Fine-tuning**, **Per-FedAvg** (first-order), **Clustered FL** (IFCA-style, 3 models), **FedPer** (last layer stays local). M3: **q-FFL** (q = 1, 5) on the global model; + FedPer if time. AFL → future work | One per family, as in R1 |
| D6 | Model | CNN: 5 conv layers (32-32-64-64-128) + **GroupNorm** + global average pool + linear head (~0.14 M params). Adult: MLP 64-32 | BatchNorm breaks per-example DP; 4 GB VRAM |
| D7 | Grid | ε ∈ {∞, 8, 4, 2, 1, 0.5}; M1 also 0.3 (re-tests the old synthetic finding). **ε₀ = 2** (Gate 2). Seeds 0, 1, 2 → mean ± SD | R1 promised the 6-point sweep |
| D8 | Privacy check | **Loss-threshold membership inference** per client (members = train, non-members = test, balanced): AUC + attack accuracy + advantage | Tests "same ε = same leakage?" without shadow models |
| D9 | Cost | MB per round + total, s per round, peak GPU MB, DP time overhead (DP ÷ no-DP) | Rubric asks for overhead/cost |
| D10 | Shortcut | FedAvg and Fine-tuning share one run: fine-tuning branches after round R−4 and spends the last 10% of the step budget locally | Same step count → same ε; halves M1 |

**Starting hyperparameters** (`configs/base.yaml`): 40 rounds · 20 private steps per round · Poisson rate 0.03 (≈ 63 records per batch for a median client) · clip 1.0 · local LR 0.05 · server LR 1.0 · Per-FedAvg α 0.03 / β 0.05 / 5 adaptation steps · IFCA k = 3 · δ = 10⁻⁵. Noise multiplier at ε = 8 / 4 / 2 / 1 / 0.5 / 0.3 ≈ 0.88 / 1.25 / 2.02 / 3.57 / 6.61 / 10.5.

**Changes since Review 1** (say them first, with the reason — slide + paper §IV):
1. DP level fixed to per-example — R1 text contradicted itself; client-level DP fails with 20 clients.
2. FEMNIST dropped — compute (one 4 GB GPU, one weekend). Listed as future work.
3. q-FFL applied to the global model — a clean test of "fairness objective vs personalisation".
4. Membership inference = loss-threshold attacker — cheap, no shadow models, gives a lower bound on leakage.

---

## 4. 🧪 Experiments (`configs/queue.yaml`)
**Rules:** one DP run on CIFAR-10 ≤ **6 min** (else cut rounds). One JSON per run in `results/runs/`; the sweep **resumes** after any crash and refuses to start twice.

| Queue | Block | Methods | ε | Runs (× 3 seeds) | Priority |
|---|---|---|---|---|---|
| Q1 | M1 privacy sweep | FedAvg + Fine-tuning (1 run → both) | ∞, 8, 4, 2, 1, 0.5, 0.3 | 21 | Must |
| Q2 | Non-private ceilings | Local, Per-FedAvg, Clustered, FedPer | ∞ | 12 | Must |
| — | **Gate 2:** confirm ε₀ (silence = 2) | | | | |
| Q3 | M2 core | Local, Per-FedAvg, Clustered, FedPer | ε₀ | 12 | Must |
| Q4 | M3 fairness objective | q-FFL q = 1, 5 (q = 0 is FedAvg) | ε₀, ∞ | 12 | Must |
| Q5 | Adult — group fairness | FedAvg + FT, Local, Per-FedAvg, Clustered, FedPer | ∞, ε₀ | 30 (fast) | Should |
| Q6 | M2 full sweep | Local, Per-FedAvg, Clustered, FedPer | 8, 4, 1, 0.5 | 48 | Should |
| Q7 | Extras | q-FFL + FedPer; heterogeneity check α = 1.0 | ε₀ | 12 | Nice |

**≈ 147 runs** (117 CIFAR-10 + 30 Adult). 6 min/run → done Sat ~22:00. 10 min/run → Sun ~06:30.

**Gates**
- **Gate 1** (`scripts/smoke.py`): GPU + sm_75 OK · Opacus = torch.func gradients · every method runs · accountant ε within 0.05 · est. ≤ 6 min/run.
- **Gate 2** (after Q1+Q2): curves sane · ε₀ = where DP-FedAvg still learns but loses ≥ 5 points vs ∞.
- **Gate 3** (`scripts/aggregate.py` → `results/sanity.txt`): non-private ≥ private · small seed SD · MIA AUC ≈ 0.5 at small ε · re-run failures.

**Logged per run:** per-client accuracy (global + personalised) → mean, worst-10%, SD, variance, Jain · personalisation gain · ε spent, δ, σ, queries · MIA AUC/accuracy/advantage · MB, s/round, GPU MB · Adult: demographic-parity + equal-opportunity gaps. *Accuracy loss per unit ε* and the **recommended configuration (R1 Eq. 5)** come from `aggregate.py`.

**Figures/tables** (`scripts/figures.py`): accuracy vs ε (M1, all mechanisms) · fairness vs ε · mechanisms at ε₀ · per-client gain box plot (failure cases) · q-FFL trade-off · MIA vs ε · tables: main results, Adult gaps, cost, recommended configuration, datasets, hyperparameters.

---

## 5. 🗓️ Schedule

### 🌙 Fri night → Sat early (Cl, while C sleeps)
Code + tests ✅ · Flower demo ✅ (Linux) · CLAUDE.md + README ✅ · references verified + `refs.bib` ✅ · paper LaTeX through §IV ✅ · zip delivered in the chat ✅

### 📅 Sat 17 Oct
| Time | Who | Block | What |
|---|---|---|---|
| 08:30 | C | — | Download the zip from the claude.ai chat → extract to `C:\Users\Chaitanya\T16_P20`. Claude desktop app → **Code** tab → **Local** → Select folder → permission mode **Auto**. Type: `Read CLAUDE.md and start Block S1.` Send A & T their tasks |
| 08:30–10:00 | CC | **S1** | venv + CUDA PyTorch · tests · dataset stats · **Gate 1** · git init |
| ~10:00 | C | S1 | Start the sweep in **your own PowerShell window** (CC gives the command); leave it running |
| 10:00–13:00 | CC | **S2** | Test the (already built) Flower demo on Windows, CPU · README · `env.txt` + `requirements-lock.txt` |
| ~13:00 | C + CC | **Gate 2** | Look at M1 curves → confirm ε₀ |
| 13:00–14:00 | C | — | Read paper §I–IV on Overleaf (CC zips `paper/`). Mark every line you can't defend |
| 14:00–18:00 | CC | **P1** | Fix your marks · interim figures from finished runs |
| 20:00 | C | — | 15-min look at interim figures: do they make sense? |
| 23:00 | C | — | §13 checklist → sleep. Queue keeps running |

### 📅 Sun 18 Oct
| Time | Who | Block | What |
|---|---|---|---|
| 08:00 | C + CC | **S3** | Queue done? retry failures · `aggregate.py` → **Gate 3** · `figures.py` |
| 09:00–12:30 | CC | **P2** | §V Results · §VI Discussion · §VII Conclusion · abstract → **paper v1.0** (Overleaf zip) |
| 12:30–13:30 | C | — | Compile on Overleaf; read it all; mark issues |
| 13:30–14:30 | CC | P2 | Fixes → v1.1 · **C runs Turnitin** |
| 13:30–15:00 | CC + C | **S4** | GitHub repo (C creates `t16-pfl-dp-fairness` at github.com/new) → push · README final · tag v1.0 |
| 13:30–16:30 | Cl | **D1** | Final deck in a **claude.ai chat** in this Project (upload `paper/figures/*.png` + `results/summary_table.csv`) → .pptx + PDF |
| 14:30–16:30 | C | — | Own-voice rewrite of Intro, Discussion, Conclusion (Turnitin AI flags); CC re-checks facts → v1.2 |
| 15:00 | A / T | — | A: proofread v1.1 (labels, units, legends, captions, error bars) · T: clone repo, run the demo from the README on his laptop |
| 16:30 | C | — | Email v1.2 PDF to Dr Kalaavathi B. Deck to A & T |
| 16:30–17:30 | C | — | Demo dry run ×2 · record a backup video |
| 19:00–20:00 | All | — | Rehearsal 1 (timed, 10–12 min + demo) + Q&A drill |
| 20:00–21:00 | C | — | Buffer · update PROGRESS · **done ✅** |

### 📅 Mon 19 → Wed 21
- **Mon:** faculty feedback → fixes · rehearsal 2 · (journal: Chaitanya's call).
- **Tue:** offline dry run (Wi-Fi off) of talk + demo · PDF + video on USB · charge laptop.
- **Wed:** Final Review 🎯 → note every requested change → fix Thu–Fri.

### 👥 Teammate tasks
- **Aprark:** Sun 15:00 — proofread v1.1: every figure/table has axis labels, units, legend, caption, error bars. Rehearse his slides.
- **Tanishq:** Sun 15:00 — clone the repo, follow the README, run the demo on his laptop (CPU); report any step that fails (= reproducibility evidence). Rehearse his slides.

---

## 6. 📄 Paper (`paper/`, IEEEtran journal template · compile on Overleaf · 8–10 pages)
| Part | Content | Status |
|---|---|---|
| Title, authors | 4 authors incl. Dr. Kalaavathi B; Chaitanya corresponding | ✅ |
| Abstract (≤ 250 words) | Per-example DP; 2 datasets; results sentence | 🟨 placeholder for results |
| I Introduction | Rewritten from R1; contributions match the study | ✅ draft |
| II Related work + Table 1 | Rewritten from R1 + new refs; "This work" row updated | ✅ draft |
| III Method | Problem (fixed §3.1) · DP-SGD + accountant · mechanisms · q-FFL · MIA · metrics | ✅ draft |
| IV Setup | Datasets · partition · model · hyperparameters · baselines · hardware/software · changes since R1 | ✅ draft (numbers marked `\todo` until Gate 1) |
| V Results / VI Discussion / VII Conclusion | From `results/` only | ⬜ Sun (CC) |
| Acknowledgment | IEEE AI-use disclosure | ✅ placeholder |

---

## 7. 🎤 Final deck (10–15 min cabin review → ~11 slides + 3-min demo)
| # | Slide | Rubric | Speaker |
|---|---|---|---|
| 1 | Title · authors · paper status | 7 | C |
| 2 | Problem + gap (R1 recap) + changes since R1 | 3 | C |
| 3 | Tools & reproducibility (versions, GPU, repo QR) | 1 | C |
| 4 | Datasets (source, size, classes, licence, partition plot, split) | 2 | A |
| 5 | Architecture + mechanisms + hyperparameters | 3 | A |
| 6 | Metrics (units, 3 seeds, baselines) | 4 | A |
| 7 | M1: accuracy + fairness vs ε | 5 | T |
| 8 | M2: mechanisms at ε₀ + per-client gain (failure cases) | 5 | T |
| 9 | M3 q-FFL + privacy check (MIA) + cost | 4, 5 | T |
| 10 | Recommended configuration + limitations + conclusion (numbers) + future work | 5, 6 | C |
| 11 | Paper status + **live demo** | 1, 7 | C |
**Backups:** full hyperparameters · DP as the 5-step chain (no math) · mechanism details · Adult gaps · per-seed table · references.

---

## 8. 🖥️ Live demo + Git
- **Demo (2–3 min, offline):** Flower SuperLink + 3 SuperNodes on localhost (HTTP) → 5 rounds of DP FedPer on a CIFAR-10 subset → per round: accuracy, worst client, ε spent, bytes sent → per-client table + Jain's index. Backup: recorded video.
- **Git:** `github.com/chaaaait/t16-pfl-dp-fairness` · README (setup, versions, 1 command per figure) · pinned `requirements-lock.txt` · `env.txt` · MIT LICENSE · results CSV/JSON + figures · tag `v1.0`.

---

## 9. 📮 Journal (Chaitanya, later)
When you submit anywhere IEEE: ☐ Dr Kalaavathi B approves the final PDF ☐ keep the AI-use disclosure in the Acknowledgment ☐ one journal at a time ☐ ORCID if the venue asks. Shortlist from v1 (verified): IEEE IoT Journal (hybrid, $0 unless OA; $175/page over 8), IEEE Access ($2,160 APC), IEEE TMLCN ($2,160 APC).

---

## 10. 🎯 Rubric → evidence
| Criterion | Max | Evidence | Where |
|---|---|---|---|
| 1 Tools | 8 | Live Flower demo · versions file · 1-command scripts · Tanishq reproduces | Demo, slide 3, README |
| 2 Dataset | 6 | Source, size, classes, partition plot, licence, preprocessing, 70/10/20 split + reason | §IV, Table datasets, slide 4 |
| 3 Implementation | 12 | Diagram · hyperparameter table · 2 baselines + 4 mechanisms + q-FFL · design reasons · Git | §III–IV, slides 5 |
| 4 Parameters | 8 | All Excel metrics + MB, s/round, GPU MB · units · 3 seeds mean ± SD · baseline in every table | §V, tables, slide 6 |
| 5 Results | 12 | Labelled figures with error bars · interpretation · failure cases · limitations | §V–VI, slides 7–10 |
| 6 Conclusion | 6 | Answers the problem · quantified gain · future work from limitations | §VII, slide 10 |
| 7 Publication | 8 | Full draft in IEEE template (4) → submission ID (6) | Slide 11 |

---

## 11. 🧯 Risks and fallbacks
| Risk | Fallback |
|---|---|
| CUDA wheel doesn't see the GTX 1650 | Try the cu126 index; check `torch.cuda.get_arch_list()` has `sm_75` |
| Opacus hooks fail on this torch | `grad_backend: functorch` in `configs/base.yaml` (tests prove both agree) |
| Runs > 6 min | rounds 40 → 30; Q6 to 2 seeds or ε {8, 1} |
| Flower won't start on native Windows | Demo in WSL2 → else recorded video + code walk-through |
| Laptop sleeps or crashes | Re-run the same sweep command (resumes). Colab T4 for Q6 |
| Results collapse at low ε | That is a finding (privacy threshold). Keep ε₀ where models still learn |
| Turnitin AI > 20% after the pass | Second own-voice pass on Mon |
**Cut order:** Q7 → Q6 seeds 3→2 → Q6 ε 4→2 → Q5 Adult (state as a limitation). **Never cut:** Q1–Q4, MIA, cost, 3 seeds on core runs.

---

## 12. 💬 Starters
**Claude Code (Sat 08:30):** `Read CLAUDE.md and start Block S1.`
**Deck chat (Sun 13:30, claude.ai, this Project):** "Read claude/PROGRESS.md and claude/PLAN_weekend.md §7. Build the final deck from the attached figures and summary table. 10–12 min talk in a faculty cabin, laptop screen. Export .pptx + PDF."

---

## 13. ✅ Laptop overnight checklist
🔌 plugged in · battery saver off · 😴 Sleep = Never when plugged in · lid open · 🔄 Windows Update paused 1 week · 🌡️ hard flat surface, `nvidia-smi` < ~85 °C · 🪟 sweep window open · Chrome/games closed
