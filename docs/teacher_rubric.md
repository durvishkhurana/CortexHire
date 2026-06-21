# Teacher Rubric — Senior AI Engineer (Redrob, Founding Team)

**Implementation:** labels generated offline by `src/pseudo_teacher.py` (no hosted LLM). **Latest label/run stats:** [`RESULTS.md`](./RESULTS.md).

**Purpose.** This is the labeling instruction handed to the LLM teacher (STEP 8/9).
It is reconstructed *near-verbatim* from the organizers' own materials —
`data/job_description.docx` (`_txt_job_description.txt`), `data/submission_spec.docx`
(`_txt_submission_spec.txt`), and `data/redrob_signals_doc.docx`
(`_txt_redrob_signals_doc.txt`) — so the labels reproduce *their* rubric, not ours.
The relevance scale is **0–5** (six levels). Honeypots are forced to **tier 0**.
P@10 counts **tier ≥ 3** as "relevant".

> **Fairness instruction (apply to every judgment).** Ignore the candidate's
> name and any institution-prestige cue **beyond the structured `education[].tier`
> field**. Judge career evidence, not pedigree. (This is both the honest answer and
> a Stage-5 defensibility point — see Soboroff, "Don't use LLMs to make relevance
> judgments", mitigated here by anchoring + self-consistency.)

---

## 0. What the role actually is (the organizers' words)

> "Own the intelligence layer of Redrob's product … the ranking, retrieval, and
> matching systems that decide what recruiters see." Series A, building a new AI
> Engineering org from scratch. They want **deep technical depth in modern ML
> systems — embeddings, retrieval, ranking, LLMs, fine-tuning** *and* a **scrappy
> product-engineering attitude** ("willing to ship a working ranker in a week even
> if the underlying ML is obviously suboptimal"). They explicitly say: **"we'd
> rather you tilt slightly toward shipper than toward researcher."**

The single most important instruction from the JD's note to hackathon participants:

> "The right answer involves reasoning about the **gap between what the JD *says*
> and what the JD *means*.** A Tier 5 candidate may not use the words 'RAG' or
> 'Pinecone' in their profile, but if their career history shows they **built a
> recommendation system at a product company**, they're a fit. A candidate who has
> all the AI keywords listed as skills but whose title is 'Marketing Manager' is
> **not** a fit, no matter how perfect their skill list looks."

So: **score career evidence over skill-name keyword matches.** Reward plain-language
builders; punish keyword stuffers.

---

## 1. The ideal profile (the organizers' "read between the lines")

> - 6–8 years total experience, of which **4–5 are in applied ML/AI roles at
>   product companies (not pure services).**
> - Has **shipped at least one end-to-end ranking, search, or recommendation
>   system** to real users at meaningful scale.
> - Strong opinions about retrieval (hybrid vs dense), evaluation (offline vs
>   online), and LLM integration (when to fine-tune vs prompt) — **defensible with
>   reference to systems they actually built.**
> - Located in or willing to relocate to **Noida or Pune** (Hyderabad, Mumbai,
>   Delhi NCR also welcome).
> - **Active on the Redrob platform** (or clear signal of being in the market) "so
>   we can actually talk to them."

> "We are aware this is a narrow profile … we'd rather see **10 great matches than
> 1000 maybes**." → The top of the ranking should be *sparse and confident*; do not
> inflate borderline candidates.

### "Absolutely need" (the JD's must-haves)
- Production experience with **embeddings-based retrieval** deployed to real users
  (embedding drift, index refresh, retrieval-quality regression).
- Production experience with **vector DBs / hybrid search** (Pinecone, Weaviate,
  Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, "or something similar" — the
  specific tech does not matter, the operational experience does).
- **Strong Python** (code quality matters).
- Hands-on **evaluation frameworks for ranking** — NDCG, MRR, MAP,
  offline-to-online correlation, A/B-test interpretation.

### "Like to have, won't reject for" (mild plus, never decisive)
LLM fine-tuning (LoRA/QLoRA/PEFT); learning-to-rank (XGBoost-based or neural);
HR-tech / recruiting / marketplace exposure; distributed systems / large-scale
inference; open-source AI/ML contributions.

---

## 2. Hard disqualifiers (the organizers "actually apply" these)

A candidate hitting any of these is **capped low** (tier 0–1) regardless of skills:

1. **Pure-research / no production.** "Pure research environments (academic labs,
   research-only roles) without any production deployment — **we will not move
   forward.** We've tried it twice and it didn't work."
2. **Recent-LangChain-only.** "AI experience" that is primarily **<12-month
   LangChain-calling-OpenAI** projects, **without substantial pre-LLM-era ML
   production experience.** ("We're looking for people who understood retrieval and
   ranking before it became fashionable.")
3. **18-months-no-IC.** A senior who "**hasn't written production code in the last
   18 months**" because they moved into "architecture"/"tech lead" roles. "This
   role writes code."
4. **Title-chasers.** Career trajectory optimizing "Senior → Staff → Principal" by
   **switching companies every ~1.5 years.** "We need someone who plans to be here
   for 3+ years."
5. **Framework enthusiasts.** GitHub full of LangChain tutorials / "How I used [hot
   framework] to build [demo]" — "we need people who think about **systems, not
   frameworks**."
6. **Consulting-services-only career.** Spent **entire career** at consulting firms
   (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.). *Exception (verbatim):*
   "If you're **currently** at one of these but have **prior product-company
   experience, that's fine.**"
7. **CV/speech/robotics without NLP/IR.** "Primary expertise is computer vision,
   speech, or robotics **without significant NLP/IR exposure** … you'd be
   re-learning fundamentals here."
8. **Closed-source 5+ yrs, no external validation.** "Entirely closed-source
   proprietary systems for 5+ years without external validation (papers, talks,
   open-source)."

> Judge disqualifiers from **career evidence**, never from names. (6) is the only
> employer-name signal, and it carries the explicit "currently-at-but-prior-product"
> exception.

---

## 3. Behavioral signals — a learned **down-weight modifier** (never a multiplier we hand-pick)

From `redrob_signals_doc` and the JD:

> "A perfect-on-paper candidate who **hasn't logged in for 6 months** and has a
> **~5% recruiter response rate** is, for hiring purposes, **not actually
> available. Down-weight them appropriately.**"

The 23 signals (see `redrob_signals_doc`) modulate fit. Strongest directions:
- **Up:** `recruiter_response_rate`↑, recency of `last_active_date`↑,
  `interview_completion_rate`↑, `open_to_work_flag`, `profile_completeness_score`↑,
  `saved_by_recruiters_30d`↑.
- **Down:** long `notice_period_days` (JD: "love sub-30-day; can buy out up to 30;
  30+ still in scope but the bar gets higher"), stale `last_active_date`, very low
  `recruiter_response_rate`.
- **Sentinels are NOT penalties:** `github_activity_score == -1` (no GitHub linked,
  ~⅔ of the pool) and `offer_acceptance_rate == -1` (no offer history) mean
  *absent*, not *bad*. Treat as missing.

A strong-on-paper but unavailable candidate should land **mid-pack**, not top-10.

---

## 4. Location & logistics (soft)

Pune/Noida preferred; Hyderabad/Mumbai/Delhi-NCR welcome; **outside India =
case-by-case, no visa sponsorship.** Already in a preferred city → plus. Elsewhere in
India + willing to relocate → fine. Outside India + unwilling → near-gate.
**No explicit salary band is stated in the JD → do not gate on salary**; expected
salary is at most a weak soft signal.

---

## 5. Honeypots → tier 0 (never relevant)

The dataset contains **~80 honeypots** with *subtly impossible* profiles, **forced to
relevance tier 0** in the ground truth. Examples (verbatim): **"8 years of experience
at a company founded 3 years ago"**; **"'expert' proficiency in 10 skills with 0 years
used."** If a profile is internally impossible (tenure precedes company founding;
expert-with-zero-duration; dates that cannot be reconciled; experience exceeding the
career timeline), **score it tier 0** no matter how good the keywords look.

---

## 6. The 0–5 tier scale (assign exactly one integer)

| Tier | Meaning | Typical evidence |
|---|---|---|
| **5** | **Bullseye.** The "10 great matches" archetype. 6–8 yrs, 4–5 in applied ML at **product** companies; **shipped an end-to-end ranking/search/recsys** to real users; clear retrieval+evaluation depth; available & responsive. *Counts even if they never write "RAG"/"Pinecone"* — career evidence is decisive. | "Built recommendation/search ranking at Swiggy/Flipkart/Razorpay; evaluated with NDCG; production embeddings + vector search." |
| **4** | **Strong fit, minor gap.** Almost all must-haves with one soft concern (e.g. notice 90 d; slightly outside YoE band; one must-have only adjacent). | Senior ML at a product co, recsys/search shipped, but long notice or modest engagement. |
| **3** | **Relevant (P@10 boundary).** Real, relevant production ML/retrieval experience but a **material** gap: a must-have missing or weak, OR product experience thin, OR availability weak. Still genuinely worth a recruiter's time. | Applied ML at a product co without a clearly-shipped ranking system; or strong retrieval skills with only services-company tenure but real product evidence somewhere. |
| **2** | **Adjacent / weak.** Tangential ML (analytics, data engineering, CV/speech-only) without NLP/IR; or product-company experience but the AI work is thin/keyword-shaped; or a soft disqualifier present. | "Data/backend engineer transitioning to ML"; CV-only engineer; framework-tutorial GitHub. |
| **1** | **Poor fit.** Hits a hard disqualifier (pure research; recent-LangChain-only; 18-mo-no-IC; title-chaser; services-only-career; closed-source-no-validation) OR essentially unrelated role with a few AI keywords. | "Marketing/HR/Sales/Ops with AI skills listed"; tech-lead who hasn't coded in 2 yrs. |
| **0** | **Not relevant / impossible.** **Honeypots** (internally impossible profiles) and pure keyword-stuffers with no career evidence whatsoever. | "Marketing Manager with FAISS/Pinecone/LangChain as expert skills, 0 months each." |

**Tie-shaping for NDCG@10 (50% of score):** be *stingy* with 5s and 4s. The JD wants
≤~10 true matches in 100K. Reserve tier 5 for unambiguous bullseyes; when unsure
between 4 and 5, pick 4. When unsure between honeypot-0 and a low real tier, prefer 0
only if there is a concrete internal impossibility.

Also emit a **continuous 0–100** score (finer-grained than the tier, used to
top-weight the pointwise student) and **one literal evidence quote** copied from the
profile (career-history sentence, skill+duration, or a signal value) justifying the
tier.

---

## 7. Anchors (from the organizers' published examples)

The submission spec ships three example reasonings (illustrative of *what good
judgment sounds like*, not high-quality rankings):

- *Rank 1* — "Senior AI Engineer with 7 years building RAG systems **at product
  companies**; strong recent engagement…" → **tier 5** (product + shipped + engaged).
- *Rank 2* — "6 years applied ML; previously **shipped vector search at scale**;
  matches the **'product over research'** profile…" → **tier 5** (the JD's own framing).
- *Rank 3* — "Strong NLP + retrieval background; **some concern on notice period
  (120 days)** but otherwise strong fit." → **tier 4** (must-haves met, one soft
  availability concern → not a 5).

These show the intended *tone*: specific facts + JD connection + **honest
acknowledgement of concerns**, with the concern moving a near-5 to a 4.

### Hand-derived calibration anchors (from `sample_candidates.json`, cheap)
- **CAND_0000001 (sample)** — "Backend/Analytics Engineer at **Mindtree** (services),
  data-eng career, skills tilt to **CV/speech** (Image Classification, Speech
  Recognition, TTS), self-describes as transitioning to ML." → **tier 1–2**: services
  employer + CV/speech-without-IR + self-admitted not-yet-ML. A keyword-aware system
  that sees "NLP, Fine-tuning LLMs, LoRA" must **not** over-rank this.
- *(More anchors added in STEP 8 when the teacher pilot runs; the union pool labeling
  set is the real calibration. ~50 anchors are not required to start.)*

---

## 8. Output contract (per candidate)

```
{ "candidate_id": "...", "tier": 0-5, "score_0_100": 0-100,
  "evidence_quote": "<literal sentence/value copied from the profile>" }
```

No prose beyond the evidence quote. Do not invent skills, employers, numbers, or
dates not present in the profile.
