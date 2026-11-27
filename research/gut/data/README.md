# Gut study data

| File | Contents |
|---|---|
| `male_rarefied_18000_interpolated_pchip.tsv` | Feature table: one row per amplicon sequence variant (the row key is the DNA sequence), one column per time point (`0`, `1`, ...). Counts are rarefied to 18,000 reads per sample, and missing time points are filled by PCHIP interpolation. |
| `male_taxonomy.tsv` | Taxonomy for the same features: `Feature ID` (the sequence), `Taxon` (`d__...; p__...; ...; f__...; g__...`), `Confidence`. |

**Source and licence: to be filled in** (decision D6 in `docs/code-quality-plan.md`). Record where the male donor dataset comes from, who processed it (rarefaction, interpolation), and whether it may be redistributed.

**Open:** the species codes in `configs/mc_search.json` (`BBBBB`, `BFCLL`, ...) and its `before`/`after` abundances aren't derived from these files by any committed script. Which naming is canonical, and how the abundances are computed, is decision D8 (RES-4 in the plan).
