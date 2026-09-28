**Answer: yes, but no single public suite is fair on its own.** None has all of: human gold labels, text Laya is not known to have trained on, support-ticket text, and published scores from both vendors. The fairest option is a small composite of human-labelled public sets. LocalLLaMA/typed-decisions, the benchmark both vendors quote, should only be used to check that our harness reproduces published numbers.

## 1. Ranked shortlist (all kept by the verifiers)

1. **CFPB consumer complaint narratives** (consumerfinance.gov FOIA narratives archive; public domain/CC0).
   - Why: real consumer-written complaints, and the labels were picked by the consumer, not an LLM. It is the only real support-style set with a proper long tier. In a sample of 271, 134 have 128 tokens or fewer and 42 fall in 250-450.
   - Caveat: consumers sometimes pick the wrong category. Since 2022 some narratives are LLM-assisted (Liang et al. estimate about 18% by late 2024), and about 24% of the sample are credit-repair template letters. Publication of narratives stopped on 14 Aug 2026. So a post-release slice is not possible, contrary to one researcher's plan; the freshest data is the Mar-Aug 2026 exports. The ZIPs return 403 to curl, so download them in a browser. No published scores exist.
2. **Customer-service complaint tweets plus severity** (github.com/danielpreotiuc/complaints-social-media, and archive.org complaint_severity_data).
   - Why: real tweets sent to 93 support accounts. Two experts labelled complaint vs not (kappa 0.731). Three annotators rated severity on 4 ordered levels (kappa 0.64). This gives a yes/no question and a graded-score question, and the score type is where our current suite is weakest.
   - Caveat: tweets only, so there is no long tier. There is no license. The data is also redistributed as the RAFT/HELM task twitter_complaints, so Jev may well have seen it. Use only the 1,974 customer-service rows. The other 1,475 control rows were never annotated.
3. **Banking77, with 12 options per question** (mteb/banking77 test; CC-BY-4.0).
   - Why: expert labels, explicitly held out of Laya's training, and both vendors have published numbers on it.
   - Caveat: messages are very short (median 11 tokens). The text was written by experts rather than taken from real traffic, and about 14% of the training labels may be wrong (Ying & Thomas 2022). Jev has probably seen it. Laya's author wrote on dev.to that training included "banking intents", which conflicts with the hold-out claim.
4. **ENRON-FFP** (github.com/kushalchawla/Frustration-Prediction-In-Emails; 960 real business emails, 10 crowd raters each).
   - Why: real business emails rated for frustration and politeness, with lengths from 60 to 250 tokens.
   - Caveat: about 22% of the emails also appear in SetFit/enron_spam, which is in Laya's training data, so drop those. There is no license. Frustration is rare: 74 emails have 30% or more of raters marking it, and 12 have at least half.
5. **Twitter US Airline Sentiment** (osanseviero/twitter-airline-sentiment).
   - Why: real customer tweets, crowd-labelled for a 3-level sentiment and a complaint-reason category.
   - Caveat: the license is non-commercial (CC BY-NC-SA). It is a famous dataset, so Jev has probably seen it. Filtering to high annotator confidence keeps only easy items.
6. **LegalBench learned_hands_consumer** (nguha/legalbench, test split, 614 posts, 307 yes / 307 no).
   - Why: real help requests from members of the public, filling both length tiers (161 posts at 250-450 tokens).
   - Caveat: the license is non-commercial (CC BY-NC-SA). The "no" examples are random posts from other topics, which makes the task easy. The framing is legal, not product support.

Not used as fairness evidence:
- **Harness checks only:** typed-decisions, sysone-bench, jev-bench, SST-5.
- **Rejected:**
  - DAIR emotion: labels come from hashtags, not people.
  - Tobi-Bueck, Bitext, AG News, BoolQ, Enron-spam, phishing: synthetic text, or in Laya's training data.
  - The federal-solicitations benchmark: its gold labels are private.

## 2. Run first: Banking77 with 12 options

It is the only set that meets all four criteria: human gold labels, explicitly held out of Laya's training, customer-service text, and published scores from both sides. Start with sysone-bench's exact 96 items, where Jev scored 0.906 and Laya 0.802, and raw outputs for each item are published. That checks our harness item by item before we trust any new number. Then run the full 385-item version below.

The headline fairness result should come from the CFPB set plus the complaint tweets. Banking77 is too short and too well known to carry the verdict alone.

**How to treat typed-decisions:** run the full test split once as a harness calibration.
- Expected results: Jev about 0.727 (yes/no 0.775, choice 0.720, score 0.696); base Laya about 0.362; majority-class baseline 0.461.
- Report it as "agreement with a 4B LLM teacher", not as correctness. Both the text and the gold labels are LLM-made, and the format is TypeSafe's own.
- 31 of the 400 states are longer than the roughly 320 tokens Laya can read, so report results with and without those.
- Leave laya-typed-decisions out of the fairness comparison. It was fine-tuned on the same generator, teacher and question schemas, and the dataset card itself calls specialist and generalist scores not comparable. It scores 0.766, above the teacher's own self-agreement of 0.735, which suggests it is fitting the teacher rather than being more correct.

## 3. Composite suite spec

**Length tiers.** Measure each state with Laya's tokenizer and each full row with our length tool.
- **T128:** state of 128 tokens or fewer. Prefer 60-128 where the source has enough; tweet sets are all under 100.
- **T512:** state of 250-450 tokens. Split it into rows that fit (full row 512 tokens or fewer) and rows Laya truncates.
- Headline numbers use only rows that fit. Report truncated rows as a separate long-ticket result.

**Run rules**
- Same items, options and option order for both models.
- Ask every yes/no question twice: once as a yes/no question, and once as a two-option choice with neutral keys A/B. Laya issue #156 reports that its yes/no answers can follow the labels rather than the text. Use a 400-item subsample of each yes/no question.
- Jev: run 3 times to measure run-to-run variance.
- Report paired bootstrap confidence intervals.
  - Choice: accuracy and macro-F1.
  - Yes/no: base rate, balanced accuracy, AUROC, Brier and ECE.
  - Score: exact match, within-one-level accuracy, MAE, weighted kappa or Spearman, and the predicted-level histogram.

**A. CFPB narratives.** Two time slices:
- **S1 (before ChatGPT):** received Mar 2015 - Nov 2022. The earlier "Dec 2011" start was a correction by the verifier: narratives only exist from about March 2015.
- **S2 (fresh):** the Mar-Aug 2026 exports.

Filters: non-empty narrative, English, deduplicated (MinHash similarity 0.8 or more). Tag template letters (matching "1681", "formally dispute" or "I am writing to") and report with and without them.

- **A1, choice** (140 per tier per slice, 20 per class; 560 total). "Which financial product is this complaint about?"
  - credit_reporting: "Credit reports, credit scores, or a credit bureau"
  - debt_collection: "A debt collector or the collection of a debt"
  - credit_card: "A credit card or prepaid card"
  - bank_account: "A checking or savings account"
  - money_transfer: "Sending or receiving money, payment apps, or virtual currency"
  - mortgage: "A home mortgage or home equity loan"
  - consumer_loan: "A vehicle, student, payday, or personal loan"

  Gold is the CFPB `Product` field, mapped across the 2017 and 2023 renames:
  - The three "Credit reporting…" variants map to credit_reporting.
  - "Credit card", "Credit card or prepaid card" and "Prepaid card" map to credit_card.
  - "Checking or savings account" and "Bank account or service" map to bank_account.
  - "Money transfer…", "Money transfers" and "Virtual currency" map to money_transfer.
  - "Vehicle loan or lease", "Consumer Loan", "Student loan" and the "Payday loan…" variants map to consumer_loan.
  - Drop "Debt or credit management" and "Other financial service".

  The question is 133 Laya tokens, so it fits the 192-token question budget.
- **A2, yes/no, debt-collection complaints only** (40 yes and 40 no per tier per slice; 320 total). "Does the consumer say they do not owe the debt that is being collected?"
  - Yes: "the debt is not theirs, was already paid, came from identity theft, or was discharged"
  - No: "the complaint is about something else, such as collection tactics, notices, or threats"

  Gold is yes when Issue == "Attempts to collect debt not owed". The yes wording reuses CFPB's own sub-issue names.

**B. Complaint tweets.** T128 only. Use the 1,974 rows whose domain is not random_reply or random_tweet.
- **B1, yes/no** (1,974 items: 1,232 complaints, 742 not). Use the dataset's own definition verbatim: "A complaint presents a state of affairs which breaches the writer's favorable expectation. Does this message contain a complaint?" Gold is column 3.
- **B2, score** (the 1,232 complaints). "How severe is the complaint in this message?"
  1. "No explicit reproach: states the problem without blaming anyone"
  2. "Disapproval: expresses dissatisfaction or annoyance"
  3. "Accusation: says the company did something wrong"
  4. "Blame: holds the company responsible for the harm"

  Gold is severity 1-4 from complaint_severity_data.csv, joined by row order (the verifier confirmed the rows align 1:1).

**C. Banking77.** T128 only. 385 items, 5 per intent, from the 3,080-row test split.
- Question: "What does the customer want help with?"
- Options: the correct intent plus 11 distractors drawn uniformly with a fixed seed for each item, in shuffled order. The option key is the label name, and the description is the label with underscores replaced by spaces. No hand-written descriptions.
- Why 12 options: I measured a 20-option question at 210-300 Laya tokens, which is over the 192-token budget.
- Add sysone-bench's 96 items as the harness gate.

**D. ENRON-FFP.** First drop the emails that overlap SetFit/enron_spam, using a normalised 50-character match; about 770 remain. Before that removal, 617 emails are in T128 and 56 fall in the 250-450 range, 49 of which fit.
- **D1, yes/no.** "Does the writer of this email sound frustrated or annoyed?"
  - Yes: "Frustrated, annoyed, or impatient"
  - No: "Calm, neutral, or positive"

  The soft gold is the share of the 10 raters who gave Frustration < 0. Score it with Brier, and with AUROC comparing emails where 30% or more of raters marked frustration against emails where none did.
- **D2, score.** "How polite is this email?" with levels Impolite / Neutral / Polite. Gold is the mean of the 10 politeness ratings: -0.5 or below is impolite, 0.5 or above is polite. The main metric is Spearman correlation, because the labels are nearly two-valued.

**E. Airline tweets.** Optional; non-commercial license.
- **E1, score** (150 per class, annotator confidence 0.67 or higher). "How does the customer feel about the airline in this tweet?" with levels Negative / Neutral / Positive.
- **E2, choice** (up to 40 per class). "What is the customer mainly complaining about?"
  - Options: customer service, flight delay, flight cancellation, lost luggage, damaged luggage, poor in-flight experience, booking or reservation problems, flight attendants, long lines.
  - Drop "Can't Tell".

**F. learned_hands_consumer.** Optional; non-commercial license; all 614 posts, tagged by tier.
- Yes/no, with the LegalBench wording verbatim: "Does the post discuss issues people face regarding money, insurance, consumer goods and contracts, taxes, and small claims about quality of service?"
- Measured lengths: 147 posts are 128 tokens or fewer; 161 are 250-450, of which 71 are 320 or fewer; 184 are over 320.

**Size and Jev cost** (input tokens approximated with Laya's tokenizer; Jev's own count may differ by about 20%)

| Part | Decisions | Input tokens | Cost at $0.042 per 1M |
|---|---|---|---|
| Core (A–D, plus the A/B repeats) | about 7,500 | about 1.14M | about $0.05 |
| Optional (E and F) | about 1,400 | about 0.26M | about $0.01 |
| Harness checks (typed-decisions 2,000; sysone 156; AbdelStark 100) | about 2,250 | about 0.71M | about $0.03 |

- One full pass: about 2.1M tokens, about $0.09. Three Jev repeats: about $0.27.
- Cost is negligible. The real limit is latency: about 11k calls at a median of 0.71 s each is about 2.2 hours serially, or about 17 minutes at 8 in parallel.

## 4. Fairness caveats that remain

- **Jev's training data is undisclosed.** TypeSafe says "We make all the data ourselves". Any public set could be in it, especially Banking77, the RAFT/HELM complaint tweets, the airline tweets, LegalBench, and Enron (which is in The Pile). CFPB's Mar-Aug 2026 slice is the only partial time-based hold-out, and it still predates Jev's release by weeks.
- **Laya's full training mix is unpublished.** "Not named" is not the same as clean. It trained on unnamed task families: intent/routing, email triage, sentiment, tone, moderation, "real customer service conversations" and "banking intents".
- **Human agreement caps the achievable score:**
  - Complaint labels: kappa 0.73, so roughly 90% is the ceiling.
  - Severity: kappa 0.64; the best published macro-F1 is 55.7.
  - ENRON-FFP: rater agreement (ICC) 0.51-0.73.
  - RAFT's human baseline on Banking77 is only 0.607 macro-F1.
  - CFPB categories are chosen by the consumer and are sometimes wrong or span several products.
- **LLM-written text is not fully removed.** Post-2022 CFPB narratives include LLM-assisted text, so report slice S1 separately.
- **Architecture:**
  - Laya's 192-token question budget limits how many options it can see, and its roughly 320-token state limit truncates long text.
  - Score questions are Laya's weakest type.
  - Jev over-flags at the 0.5 threshold: 0.729 on Civil Comments against a 0.923 always-no baseline.
  - So report metrics that do not depend on a threshold.
- **We still write some of the rubrics.** Where possible, use the dataset's own guideline wording verbatim. Do not reuse jev-bench's wording, which was written by a project focused on Jev.
- **Licenses:** E and F are non-commercial, and B and D have no license. This is fine for research, but it becomes a grey area if the evaluation feeds a purchase decision. TypeSafe states it does not train on user data.

## 5. Published numbers to sanity-check the harness

**Harness gates (both models):**
- **typed-decisions test** (HF card LocalLLaMA/typed-decisions; Laya's BENCHMARKS.md): Jev 0.727, base Laya 0.362, laya-typed-decisions 0.766, laya-multilingual 0.342-0.352.
- **sysone-bench** (github.com/instax-dutta/sysone-bench, raw outputs included), Jev vs Laya:
  - Banking77, 12 intents, n=96: 0.906 vs 0.802
  - SST-5, n=60: 0.617 vs 0.367

**Banking77, other published Jev numbers** (slices differ):
- 0.870: AbdelStark, BTZSC 72 labels, n=100, revision fef2a2ac, seed 20260917
- 0.763: nibzard, 77-way
- 0.832 [0.779, 0.880]: ickma, n=208
- 0.797: Jevals
- 0.924: simonmesmith, full test set, but with 24 retrieved examples per prediction, so not zero-shot

**Banking77, Laya numbers:**
- 0.425 on all 77 options (0.492 for laya-typed-decisions)
- 54.3% rising to 60.8% with a top-20 shortlist (issue #102)
- BTZSC's modernbert-large-nli, on the same backbone as Laya: 0.240

**jev-bench (Jev only, n=800-2000):**
- SST-5 0.565
- SMS spam 0.965
- Civil Comments 0.729 (AUROC 0.829)
- Measuring Hate Speech 0.527

**Laya-only checks:**
- MASSIVE-en intent with 20 options: 0.783
- SST-5: 0.372 (n=600)
- toxic-chat: jailbreak 0.708, toxicity 0.530

**Non-vendor anchor for complaint tweets:** RAFT twitter_complaints macro-F1 is human 0.897, GPT-3 0.821, BART-MNLI 0.400. These were measured on 3,399 tweets that include the unannotated control rows, so compare against the full RAFT set, not our 1,974-row subset.

**TypeSafe itself publishes no scores on public benchmarks.** Its evals.typesafe.ai Customer Service score of 76.0% uses LLM-consensus gold and cannot be re-run.

**Files:** everything is in `(local scratch)` (`complaints-data.csv`, `complaint_severity.csv`, `ns_data/enron_ffp/enron-FFP.csv`, `samples/cfpb_sample_1000.csv`, `samples/b77_test.jsonl`, `samples/airline_tweets.csv`, `verify_lh.jsonl`, `vendor/td/test_all.json`).