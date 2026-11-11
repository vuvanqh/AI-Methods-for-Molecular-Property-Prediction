# Evidence Retrieval and Augmentation

## Overview

This document describes how the system retrieves experimental evidence from ChEMBL and turns it into a prompt for the language model. It assumes no chemistry background.

The GNN already produces the predictions: for each of the 18 tasks it outputs a yes/no plus a probability. Evidence is **not** used to make or correct those predictions. It exists so the generated interpretation can cite real published measurements instead of producing plausible-sounding but unverifiable prose.

The difference this makes:

> "This molecule likely blocks hERG because of its basic nitrogen." — unverifiable

> "A close analogue has a measured hERG IC50 of 14 nM (Coghlan, *J Med Chem* 2001)." — checkable

## Background: what the data means

### Assays and potency

A bioassay answers one question: **how much of the compound do you need before something happens?**

That amount is a concentration, normally reported in nanomolar (`nM`, 10⁻⁹ mol/litre). The less you need, the stronger the compound. "Potency" is this idea: a low concentration means high potency.

| concentration | reading |
|---|---|
| 1 nM | very potent |
| 1 000 nM (1 µM) | moderate |
| 10 000 nM (10 µM) | weak — the usual cutoff for calling something inactive |

Note the inversion: a *smaller* number means a *stronger* compound.

### Kinds of measurement

The number alone is incomplete, because there are several different quantities a lab can measure:

| measurement | what it is |
|---|---|
| `IC50` | concentration that inhibits 50% of the target's activity |
| `EC50` | concentration producing 50% of the maximum effect (activation rather than inhibition) |
| `Ki` | binding affinity — how tightly the compound sticks to the target |
| `Kd` | dissociation constant; binding only, no functional effect |
| `Potency` | a generic term used by high-throughput screening campaigns |

These are not interchangeable. Comparing an `IC50` to a `Ki` is like comparing blood pressure to cholesterol: both are numbers about the same patient, measuring different things.

### Why we store a logarithm

Potencies span roughly six orders of magnitude, from about 0.1 nM to 100 000 nM. Averaging or ranking such values linearly is dominated by the large ones, so the field uses the negative base-10 logarithm of the molar concentration. ChEMBL publishes this as `pchembl_value`; this codebase calls it `log_potency`.

| log_potency | concentration | |
|---|---|---|
| 9 | 1 nM | very potent |
| 7.85 | 14 nM | the sertindole hERG measurement |
| 6 | 1 000 nM | moderate |
| 5 | 10 000 nM | weak / inactive threshold |

Two consequences: higher now means stronger, and each whole step is a tenfold change in concentration. A difference of 2 is a hundredfold difference in potency.

### The relation, and why it matters most

An assay only tests a limited concentration range. Two outcomes are possible:

- The compound acts somewhere inside the range, a curve is fitted, and an exact number is reported — recorded with relation `=`.
- The compound does nothing even at the highest concentration tested. No number can be fitted. All that is known is that the true value lies *above* the tested maximum — recorded as `> 10000 nM`.

So `>` marks a **negative result**: the compound did not do the thing. On hERG alone, ChEMBL holds 7115 such rows.

This makes the relation the single most important field:

```
IC50  =  10000 nM   →  a weak blocker. It does block hERG.
IC50  >  10000 nM   →  not a blocker. No effect at the strongest dose tested.
```

The same number, opposite conclusions. Dropping the relation inverts the meaning of the row. The relation `<` is the mirror case — already fully active at the lowest dose, so very potent, exact value unknown.

ChEMBL only computes `pchembl_value` when the relation is `=`, which is why `log_potency` is null on every `>` row.

### Targets versus measurements

A **target** is the protein being tested against. A **measurement** is the quantity measured about it. Using a medical check-up as an analogy:

| | check-up | this system |
|---|---|---|
| target | which organ — heart, liver | which protein — hERG channel, CYP3A4 enzyme |
| measurement | which test — blood pressure | which quantity — IC50, Ki |
| value + units | 120 mmHg | 14 nM |

One row therefore reads: *against the hERG potassium channel (target), the IC50 (measurement) was 14 nM (value and units)*.

The target decides whether a row is relevant at all. The GNN predicts hERG blockade, so only rows whose target is hERG can support or contradict that prediction.

### Qualitative outcomes

High-throughput screens such as Tox21 often do not fit a curve. They assign a label instead: `active`, `inactive` or `inconclusive`. Those rows carry a concentration but no `pchembl_value`, and the label is the entire interpretation. The field is called `outcome`.

## What ChEMBL returns

### ChEMBL stores experiments, not properties

This is the key structural point. ChEMBL does not hold "the properties of a molecule" as a record. It holds **published experiments**, one row each:

```
1 activity row = one experiment, from one paper:
    this molecule × this target × this assay → this number
```

There is consequently no single authoritative value for any property. Sertindole against hERG has four separate measurements:

```
IC50  =  14.0 nM   log_potency 7.85   "Inhibition of hERG currents Kv11.1"          2001
IC50  ~   3.0 nM   log_potency null   "... after prolonged depolarizing pulses"     2001
IC50  =   3.0 nM   log_potency 8.52   "... in CHO cells expressing HERG Kv11.1"     2003
IC50  =   2.7 nM   log_potency 8.57   "... by automated patch clamp assay"          2014
```

A fivefold spread, because different laboratories used different cell lines and protocols. The `assay_description` is what distinguishes them.

### Row counts

Counts per *property* are small; counts per *molecule* are large, because a well-studied drug has been tested against many proteins.

```
imatinib (CHEMBL941), human targets          sertindole (CHEMBL12713), human targets
 ├─ KIT          146 experiments              ├─ hERG/KCNH2     6 experiments
 ├─ ABL1         119 experiments              ├─ 5-HT2A         5 experiments
 ├─ K562         115 experiments              ├─ D2             4 experiments
 ├─ ~500 more targets                         ├─ ~45 more targets
 └─ 2716 rows total                           └─ 113 rows total
```

### Endpoints used

| endpoint | returns |
|---|---|
| `molecule.json` | resolves a SMILES string to a ChEMBL id |
| `activity.json` | the measurements — the rows above |
| `document.json` | the publication behind a measurement: title, abstract, DOI, year |

An activity row carries only `document_chembl_id`, a bare identifier. The title and abstract come from a second request, which the code joins locally.

### Fields kept

ChEMBL returns roughly 45 fields per activity; ten are kept.

| ChEMBL field | stored as | notes |
|---|---|---|
| `standard_type` | `measurement` | IC50, Ki, Potency |
| `standard_relation` | `relation` | `=`, `>`, `<`, `~` |
| `standard_value` | `value` | converted from string to float |
| `standard_units` | `units` | normalised by ChEMBL, usually nM |
| `pchembl_value` | `log_potency` | computed by ChEMBL; null unless relation is `=` |
| `activity_comment` | `outcome` | active / inactive / inconclusive |
| `target_pref_name` | `target_name` | |
| `target_chembl_id` | `target_identifier` | used to deduplicate |
| `assay_description` | `assay_description` | distinguishes repeat measurements |
| `molecule_chembl_id` | `molecule_identifier` | |

Two further fields are constructed rather than returned: `provider` (which source produced the row) and `document_url` (built from the DOI, falling back to a ChEMBL link).

ChEMBL returns every number twice — once as published, once normalised:

```
"value": "50.1187",          "units": "uM"      ← as the paper printed it
"standard_value": "50118.7", "standard_units": "nM"   ← normalised by ChEMBL
```

The code always reads the `standard_` variants. Mixing the two would place µM and nM values in the same ordering and rank them a thousandfold wrong.

Discarded: ontology identifiers (`bao_endpoint`, `qudt_units`, `uo_units`), internal keys (`record_id`, `toid`), and other fields that cannot help explain a prediction. Two fields are used but never stored — `data_validity_comment` and `potential_duplicate` are sent as server-side filters so flagged rows are never downloaded.

## What the code does

### Request flow

```
describe_molecule(smiles)              validates input, rejects bad SMILES cheaply
        │
retriever.retrieve(smiles, k)          RAG: k nearest molecules from the local corpus
        │
asyncio.gather(                        GNN and evidence run concurrently
    gnn_client...,                       18 predictions + atom attributions
    gather_evidence(...)                 ChEMBL lookups for all k+1 molecules
)
        │
build_prompt(...)                      predictions + attributions + evidence → text
        │
llm_client.get_summary(prompt)         the written interpretation
```

### Components

| file | responsibility |
|---|---|
| `src/chembl_client.py` | HTTP only: pagination, batching, filters, error translation |
| `rag/augmentation/client_retrieval_base.py` | the strategy interface each provider implements |
| `rag/augmentation/chembl_retrieval.py` | the only module that knows ChEMBL's vocabulary |
| `rag/augmentation/evidence_collection.py` | routes each molecule to the providers it came from |
| `rag/augmentation/prompt_builder.py` | selects rows and renders the prompt |
| `rag/result_types/molecular_evidence.py` | the provider-neutral row |

### Selecting which rows reach the prompt

A molecule can return thousands of rows; the prompt takes at most `MAX_EVIDENCE_PER_MOLECULE`. The filter in `prompt_builder._select` is:

1. Discard rows with neither a value nor an outcome.
2. Rank by potency (`_rank`).
3. Keep only the most potent row per target, so four near-identical hERG readings do not consume four slots.
4. Take the first N.

`_rank` places every row on one scale. Where ChEMBL supplied `log_potency` it is used directly; otherwise the value is converted with `9 - log10(nM)`. Without that fallback, screening rows — which carry a concentration but no `pchembl_value` — would always sort last and never reach the prompt. Since those are exactly the rows backing the Tox21 tasks, the fallback is what keeps that half of the evidence reachable.

### Rendering

Each selected row becomes one line:

```
- KCNH2 (ChEMBL): IC50 = 14 nM (-log10[M] 7.85); assay: Inhibition of hERG currents
  Kv11.1; Recent developments in ... potassium channel modulators (2001)
```

Neighbours are numbered `[1]…[k]`, matching the numbered source links in the frontend. One abstract is included, truncated to `ABSTRACT_MAX_CHARS`. Predictions are read from the GNN response keys rather than a fixed list, so changing the task set requires no change here.

Atom attributions are rendered as small fragments (`N3 in CN`) from the top-scoring atoms of each positive prediction. These are produced from the **original** SMILES rather than the canonical form, because the GNN indexed atoms in the order it received them; canonicalising first would misalign every attribution.

## Design decisions

### Provider-neutral evidence type

`MolecularEvidence` deliberately avoids ChEMBL vocabulary. `pchembl_value` became `log_potency`, `standard_type` became `measurement`, and every row carries a `provider` field. A second source such as PubChem can populate the same object without imitating ChEMBL, and merged evidence remains attributable to its origin.

### Strategy pattern keyed on provider

`molecule_sources` already records which database each corpus molecule came from. `gather_evidence` uses that: a molecule is queried only at the providers it was actually ingested from, using that provider's own identifier. The queried molecule is not in the corpus and has no stored provider, so it is asked of every source by structure.

The registry is a plain list, and the match is `strategy.provider == source.provider`. A dictionary keyed by provider would write the string `"ChEMBL"` in two places, where a mismatch would silently yield zero evidence and no error.

### No hardcoded target map

Restricting the request to the ~15 proteins behind the 18 tasks would be a large efficiency win: imatinib would return roughly 20 rows instead of 2716, in about two seconds instead of fifty.

It was rejected because such a map is hand-written domain knowledge that cannot be derived from the task names — `NR-AR` does not contain "Androgen receptor", and `SR-ARE` does not contain "NRF2". A stale entry fails silently: the affected task receives no evidence at all, with nothing to indicate why. Three of the eighteen mappings were uncertain even when first drafted (`SR-HSE`, `SR-ATAD5`, and `SR-MMP`, which has no protein target).

The accepted cost is that selection ranks by potency rather than relevance, so a molecule's strongest off-target measurements can outrank the task-relevant one. For sertindole the hERG row survived at position 7 of 10. Should this prove limiting, the correct form is a map used as a *preference*, falling back to unfiltered retrieval, so that a stale entry costs speed rather than evidence.

### Degradation instead of failure

Evidence enriches the interpretation; it is not required for a prediction. Three failures are therefore contained:

- A provider that is unreachable loses only its own rows; `_retrieve` catches `ConnectionError` per call.
- A failed document lookup keeps the measurements and drops only their citations.
- A failed language-model call yields an empty summary, which the frontend renders by hiding the interpretation card.

In all three cases the predictions and similar compounds are still returned. ChEMBL has proven intermittently unavailable during development, so this path is exercised in practice.

### Notes on the language model

Reasoning models can exhaust the token budget on internal reasoning and return no content at all; one such model produced 0 characters of content alongside 3827 characters of reasoning. `get_summary` therefore rejects an empty answer rather than returning `None`, which would otherwise fail response validation and turn a degraded summary into a failed request. `llm_max_tokens` is set to 2000 to leave room for both the reasoning and a 250-word answer.

## References

- ChEMBL web services: `https://www.ebi.ac.uk/chembl/api/data`
- Release used during development: ChEMBL_37 (2026-05-01)
- A compound record: `https://www.ebi.ac.uk/chembl/explore/compound/{CHEMBL_ID}`
