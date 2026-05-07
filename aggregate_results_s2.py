# -*- coding: utf-8 -*-
"""
aggregate_results_s2.py
=======================
Setting 2 (TopU few-shot) aggregator. Mirrors aggregate_results.py for
Setting 1 line-for-line, with these additions:

- Adds a `tier` column (Tier 1: <50 TopU actives, Tier 2: >=50)
- Tier comes from topU_95_updated.xlsx -> topu_active_count column
- Three summary rows at the bottom: TIER1, TIER2, OVERALL
- Wandb logs per-tier summary scalars in addition to overall

All 7 metrics are reported equally - select your preferred primary metric
downstream.

Reads per-seed CSVs from results/setting2/{model_name}/per_seed/ and produces:

    target | name | class | tier | n_actives | n_decoys | ef_1pct | ef_5pct | ...

where each metric is formatted as "mean +/- std".

Usage:
    python aggregate_results_s2.py --model morgan_rf
    python aggregate_results_s2.py --model gin

Optional:
    --results_dir /path/to/results/setting2
    --seeds      2026 2027 2028
    --xlsx       /path/to/topU_95_updated.xlsx
    --out        /path/to/output.csv
    --no_wandb
"""

import argparse
import csv
import glob
import os
import numpy as np
import wandb

WANDB_ENTITY  = "kumar-surbhi1294-university-of-texas-at-dallas"
WANDB_PROJECT = "LBVS"

# Tier rule (paper Table 2 + Appendix E.3, Table 9)
TIER_CUTOFF = 50

# ---------------------------------------------------------------------------
# Target metadata (identical to aggregate_results.py)
# ---------------------------------------------------------------------------

TARGET_META = {
    "cp2c9":  ("Cytochrome P450 2C9",                                          "Cytochrome P450"),
    "cp3a4":  ("Cytochrome P450 3A4",                                          "Cytochrome P450"),
    "cp2d6":  ("Cytochrome P450 2D6",                                          "Cytochrome P450"),
    "cp1a2":  ("Cytochrome P450 1A2",                                          "Cytochrome P450"),
    "aa2ar":  ("Adenosine A2a receptor",                                       "GPCR"),
    "adrb1":  ("Beta-1 adrenergic receptor",                                   "GPCR"),
    "adrb2":  ("Beta-2 adrenergic receptor",                                   "GPCR"),
    "cxcr4":  ("C-X-C chemokine receptor type 4",                              "GPCR"),
    "drd3":   ("Dopamine D3 receptor",                                         "GPCR"),
    "oprd":   ("Delta opioid receptor",                                        "GPCR"),
    "5ht6r":  ("Serotonin 6 (5-HT6) receptor",                                "GPCR"),
    "aa3r":   ("Adenosine A3 receptor",                                        "GPCR"),
    "hrh3":   ("Histamine H3 receptor",                                        "GPCR"),
    "oprk":   ("Kappa opioid receptor",                                        "GPCR"),
    "oprm":   ("Mu opioid receptor",                                           "GPCR"),
    "cnr2":   ("Cannabinoid CB2 receptor",                                     "GPCR"),
    "drd2":   ("Dopamine D2 receptor",                                         "GPCR"),
    "kcnh2":  ("HERG",                                                         "Ion Channel"),
    "5ht3a":  ("Serotonin 3a (5-HT3a) receptor",                              "Ion Channel"),
    "scn5a":  ("Sodium channel protein type V alpha subunit",                  "Ion Channel"),
    "cac1h":  ("Voltage-gated T-type calcium channel alpha-1H subunit",        "Ion Channel"),
    "trpa1":  ("Transient receptor potential cation channel subfamily A",      "Ion Channel"),
    "abl1":   ("Tyrosine-protein kinase ABL",                                  "Kinase"),
    "akt1":   ("Serine/threonine-protein kinase AKT",                          "Kinase"),
    "akt2":   ("Serine/threonine-protein kinase AKT2",                         "Kinase"),
    "braf":   ("Serine/threonine-protein kinase B-raf",                        "Kinase"),
    "cdk2":   ("Cyclin-dependent kinase 2",                                    "Kinase"),
    "csf1r":  ("Macrophage colony stimulating factor receptor",                "Kinase"),
    "egfr":   ("Epidermal growth factor receptor erbB1",                       "Kinase"),
    "fak1":   ("Focal adhesion kinase 1",                                      "Kinase"),
    "fgfr1":  ("Fibroblast growth factor receptor 1",                          "Kinase"),
    "igf1r":  ("Insulin-like growth factor I receptor",                        "Kinase"),
    "jak2":   ("Tyrosine-protein kinase JAK2",                                 "Kinase"),
    "kit":    ("Stem cell growth factor receptor",                             "Kinase"),
    "kpcb":   ("Protein kinase C beta",                                        "Kinase"),
    "lck":    ("Tyrosine-protein kinase LCK",                                  "Kinase"),
    "mapk2":  ("MAP kinase-activated protein kinase 2",                        "Kinase"),
    "met":    ("Hepatocyte growth factor receptor",                            "Kinase"),
    "mk01":   ("MAP kinase ERK2",                                              "Kinase"),
    "mk10":   ("c-Jun N-terminal kinase 3",                                    "Kinase"),
    "mk14":   ("MAP kinase p38 alpha",                                         "Kinase"),
    "mp2k1":  ("Dual specificity mitogen-activated protein kinase kinase 1",   "Kinase"),
    "plk1":   ("Serine/threonine-protein kinase PLK1",                         "Kinase"),
    "rock1":  ("Rho-associated protein kinase 1",                              "Kinase"),
    "src":    ("Tyrosine-protein kinase SRC",                                  "Kinase"),
    "tgfr1":  ("TGF-beta receptor type I",                                     "Kinase"),
    "vgfr2":  ("Vascular endothelial growth factor receptor 2",                "Kinase"),
    "wee1":   ("Serine/threonine-protein kinase WEE1",                         "Kinase"),
    "andr":   ("Androgen Receptor",                                            "Nuclear Receptor"),
    "esr1":   ("Estrogen receptor alpha",                                      "Nuclear Receptor"),
    "esr2":   ("Estrogen receptor beta",                                       "Nuclear Receptor"),
    "gcr":    ("Glucocorticoid receptor",                                      "Nuclear Receptor"),
    "mcr":    ("Mineralocorticoid receptor",                                   "Nuclear Receptor"),
    "ppara":  ("Peroxisome proliferator-activated receptor alpha",             "Nuclear Receptor"),
    "ppard":  ("Peroxisome proliferator-activated receptor delta",             "Nuclear Receptor"),
    "pparg":  ("Peroxisome proliferator-activated receptor gamma",             "Nuclear Receptor"),
    "prgr":   ("Progesterone receptor",                                        "Nuclear Receptor"),
    "rxra":   ("Retinoid X receptor alpha",                                    "Nuclear Receptor"),
    "thb":    ("Thyroid hormone receptor beta-1",                              "Nuclear Receptor"),
    "aces":   ("Acetylcholinesterase",                                         "Other Enzymes"),
    "aofb":   ("Monoamine oxidase B",                                          "Other Enzymes"),
    "ampc":   ("Beta-lactamase",                                               "Other Enzymes"),
    "cah2":   ("Carbonic anhydrase II",                                        "Other Enzymes"),
    "dhi1":   ("11-beta-hydroxysteroid dehydrogenase 1",                       "Other Enzymes"),
    "dyr":    ("Dihydrofolate reductase",                                      "Other Enzymes"),
    "fnta":   ("Protein farnesyltransferase alpha subunit",                    "Other Enzymes"),
    "glcm":   ("Beta-glucocerebrosidase",                                      "Other Enzymes"),
    "hdac2":  ("Histone deacetylase 2",                                        "Other Enzymes"),
    "hdac8":  ("Histone deacetylase 8",                                        "Other Enzymes"),
    "hmdh":   ("HMG-CoA reductase",                                            "Other Enzymes"),
    "hivint": ("Human immunodeficiency virus type 1 integrase",                "Other Enzymes"),
    "hivrt":  ("HIV type 1 reverse transcriptase",                             "Other Enzymes"),
    "hxk4":   ("Hexokinase type IV",                                           "Other Enzymes"),
    "inha":   ("Enoyl-[acyl-carrier-protein] reductase",                       "Other Enzymes"),
    "nos1":   ("Nitric-oxide synthase, brain",                                 "Other Enzymes"),
    "parp1":  ("Poly [ADP-ribose] polymerase-1",                               "Other Enzymes"),
    "pde5a":  ("Phosphodiesterase 5A",                                         "Other Enzymes"),
    "pgh1":   ("Cyclooxygenase-1",                                             "Other Enzymes"),
    "pgh2":   ("Cyclooxygenase-2",                                             "Other Enzymes"),
    "ptn1":   ("Protein-tyrosine phosphatase 1B",                              "Other Enzymes"),
    "pygm":   ("Muscle glycogen phosphorylase",                                "Other Enzymes"),
    "pyrd":   ("Dihydroorotate dehydrogenase",                                 "Other Enzymes"),
    "tysy":   ("Thymidylate synthase",                                         "Other Enzymes"),
    "ace":    ("Angiotensin-converting enzyme",                                "Protease"),
    "ada17":  ("ADAM17",                                                       "Protease"),
    "bace1":  ("Beta-secretase 1",                                             "Protease"),
    "dpp4":   ("Dipeptidyl peptidase IV",                                      "Protease"),
    "fa10":   ("Coagulation factor X",                                         "Protease"),
    "hivpr":  ("HIV type 1 protease",                                          "Protease"),
    "mmp13":  ("Matrix metalloproteinase 13",                                  "Protease"),
    "thrb":   ("Thrombin",                                                     "Protease"),
    "try1":   ("Trypsin I",                                                    "Protease"),
    "urok":   ("Urokinase-type plasminogen activator",                         "Protease"),
    "hs90a":  ("Heat shock protein HSP 90-alpha",                              "Miscellaneous"),
    "xiap":   ("Inhibitor of apoptosis protein 3",                             "Miscellaneous"),
}


METRICS = ["ef_1pct", "ef_5pct", "ef_10pct", "prauc", "rocauc", "bedroc", "bedroc_rdkit", "logauc"]

# Default search paths for the metadata xlsx
DEFAULT_XLSX_PATHS = [
    "./topU_95_updated.xlsx",
    "/groups/bcoskunuzer/sxk230046/LBVS/topu_dataset/topU_95_updated.xlsx",
    "./topU_95_updated.xlsx",
    "/mnt/project/topU_95_updated.xlsx",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))[0]


def fmt(mean, std):
    return f"{mean:.3f} \u00b1 {std:.3f}"


def load_tier_map(xlsx_path):
    """
    Read topU_95_updated.xlsx and build {target -> tier} where tier is 1 or 2
    based on topu_active_count vs TIER_CUTOFF. Returns {} if unreadable.
    """
    try:
        import pandas as pd
    except ImportError:
        print("WARNING: pandas not installed; tier column will be 'Unknown'.")
        return {}

    if not xlsx_path or not os.path.exists(xlsx_path):
        return {}

    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"WARNING: failed to read {xlsx_path}: {e}")
        return {}

    if "target" not in df.columns or "topu_active_count" not in df.columns:
        print(f"WARNING: {xlsx_path} missing required columns; tier will be 'Unknown'.")
        return {}

    tier_map = {}
    for _, row in df.iterrows():
        t = str(row["target"]).strip()
        try:
            n_act = int(row["topu_active_count"])
        except Exception:
            continue
        tier_map[t] = 1 if n_act < TIER_CUTOFF else 2
    return tier_map


def resolve_xlsx(args_xlsx):
    """If --xlsx was given use it; else try default paths in order."""
    if args_xlsx:
        return args_xlsx
    for p in DEFAULT_XLSX_PATHS:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--results_dir", default="/groups/bcoskunuzer/sxk230046/LBVS/results/setting2")
    parser.add_argument("--seeds",       nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--xlsx",        default=None,
                        help="Path to topU_95_updated.xlsx. Auto-detected if omitted.")
    parser.add_argument("--out",         default=None)
    parser.add_argument("--no_wandb",    action="store_true", help="Disable wandb logging.")
    args = parser.parse_args()

    # Load tier map from xlsx
    xlsx_path = resolve_xlsx(args.xlsx)
    if xlsx_path:
        print(f"Loading tier map from: {xlsx_path}")
    else:
        print("WARNING: topU_95_updated.xlsx not found in any default location. "
              "Pass --xlsx PATH or tier will be 'Unknown'.")
    tier_map = load_tier_map(xlsx_path) if xlsx_path else {}

    per_seed_dir = os.path.join(args.results_dir, args.model, "per_seed")
    out_path     = args.out or os.path.join(args.results_dir, args.model, "aggregate", "summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_files = glob.glob(os.path.join(per_seed_dir, f"*_seed{args.seeds[0]}.csv"))
    targets   = sorted([
        os.path.basename(f).replace(f"_seed{args.seeds[0]}.csv", "")
        for f in all_files
    ])

    if not targets:
        print(f"No completed targets found in {per_seed_dir}")
        return

    print(f"Found {len(targets)} completed targets for model '{args.model}'")

    rows         = []
    missing      = []
    all_means    = {m: [] for m in METRICS}     # all targets
    tier1_means  = {m: [] for m in METRICS}     # Tier 1 only
    tier2_means  = {m: [] for m in METRICS}     # Tier 2 only
    tier1_count  = 0
    tier2_count  = 0
    tier1_actives = tier1_decoys = 0
    tier2_actives = tier2_decoys = 0

    for target in targets:
        seed_data = []
        skip = False

        for seed in args.seeds:
            path = os.path.join(per_seed_dir, f"{target}_seed{seed}.csv")
            if not os.path.exists(path):
                missing.append(f"{target}_seed{seed}")
                skip = True
                break
            seed_data.append(load_csv(path))

        if skip:
            continue

        n_actives = int(seed_data[0]["n_actives"])
        n_decoys  = int(seed_data[0]["n_decoys"])
        name, cls = TARGET_META.get(target, ("Unknown", "Unknown"))
        tier      = tier_map.get(target, "Unknown")

        row = {
            "target":    target,
            "name":      name,
            "class":     cls,
            "tier":      tier,
            "n_actives": n_actives,
            "n_decoys":  n_decoys,
        }

        for m in METRICS:
            vals = [float(d[m]) for d in seed_data]
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=0))
            row[m] = fmt(mean, std)
            all_means[m].append(mean)
            if tier == 1:
                tier1_means[m].append(mean)
            elif tier == 2:
                tier2_means[m].append(mean)

        if tier == 1:
            tier1_count   += 1
            tier1_actives += n_actives
            tier1_decoys  += n_decoys
        elif tier == 2:
            tier2_count   += 1
            tier2_actives += n_actives
            tier2_decoys  += n_decoys

        rows.append(row)

    # -- Summary rows: TIER1, TIER2, OVERALL ----------------------------------
    if rows:
        if tier1_count > 0:
            tier1_row = {
                "target":    "TIER1",
                "name":      f"Tier 1 (<{TIER_CUTOFF} actives)",
                "class":     "TIER1",
                "tier":      1,
                "n_actives": tier1_actives,
                "n_decoys":  tier1_decoys,
            }
            for m in METRICS:
                vals = tier1_means[m]
                tier1_row[m] = fmt(float(np.mean(vals)), float(np.std(vals, ddof=0)))
            rows.append(tier1_row)

        if tier2_count > 0:
            tier2_row = {
                "target":    "TIER2",
                "name":      f"Tier 2 (>={TIER_CUTOFF} actives)",
                "class":     "TIER2",
                "tier":      2,
                "n_actives": tier2_actives,
                "n_decoys":  tier2_decoys,
            }
            for m in METRICS:
                vals = tier2_means[m]
                tier2_row[m] = fmt(float(np.mean(vals)), float(np.std(vals, ddof=0)))
            rows.append(tier2_row)

        overall = {
            "target":    "OVERALL",
            "name":      "",
            "class":     "ALL",
            "tier":      "ALL",
            "n_actives": sum(r["n_actives"] for r in rows
                              if r["target"] not in ("TIER1", "TIER2")),
            "n_decoys":  sum(r["n_decoys"]  for r in rows
                              if r["target"] not in ("TIER1", "TIER2")),
        }
        for m in METRICS:
            means = all_means[m]
            overall[m] = fmt(float(np.mean(means)), float(np.std(means, ddof=0)))
        rows.append(overall)

    fieldnames = ["target", "name", "class", "tier", "n_actives", "n_decoys"] + METRICS

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_data_rows = len([r for r in rows if r["target"] not in ("TIER1", "TIER2", "OVERALL")])
    print(f"Summary saved to: {out_path}")
    print(f"Targets included: {n_data_rows} (Tier 1: {tier1_count}, Tier 2: {tier2_count})")

    if missing:
        print(f"WARNING: {len(missing)} missing seed CSVs skipped: {missing}")

    # -- Log to wandb ---------------------------------------------------------
    if not args.no_wandb and rows:
        data_rows = [r for r in rows if r["target"] not in ("TIER1", "TIER2", "OVERALL")]

        run = wandb.init(
            entity  = WANDB_ENTITY,
            project = WANDB_PROJECT,
            group   = "setting2_aggregate",
            name    = f"{args.model}_setting2_aggregate",
            job_type= "aggregate",
            config  = {
                "model":     args.model,
                "setting":   2,
                "n_targets": len(data_rows),
                "n_tier1":   tier1_count,
                "n_tier2":   tier2_count,
                "seeds":     args.seeds,
            },
        )

        # Overall summary scalars - one per metric (mirrors Setting 1)
        for m in METRICS:
            mean_val = float(np.mean(all_means[m]))
            run.summary[f"overall/{m}_mean"] = mean_val

        # Tier-level summary scalars - new for Setting 2
        if tier1_count > 0:
            for m in METRICS:
                run.summary[f"tier1/{m}_mean"] = float(np.mean(tier1_means[m]))
        if tier2_count > 0:
            for m in METRICS:
                run.summary[f"tier2/{m}_mean"] = float(np.mean(tier2_means[m]))

        # Full per-target table
        columns = ["target", "name", "class", "tier", "n_actives", "n_decoys"] + METRICS
        table   = wandb.Table(columns=columns)
        for r in data_rows:
            table.add_data(*[r[c] for c in columns])
        run.log({"per_target_results": table})

        # Per-class mean EF@1% bar chart (identical to Setting 1)
        classes   = sorted(set(r["class"] for r in data_rows))
        class_ef1 = []
        for cls in classes:
            cls_rows  = [r for r in data_rows if r["class"] == cls]
            cls_means = [float(r["ef_1pct"].split()[0]) for r in cls_rows]
            class_ef1.append([cls, float(np.mean(cls_means))])

        class_table = wandb.Table(columns=["class", "mean_ef1pct"], data=class_ef1)
        run.log({"ef1pct_by_class": wandb.plot.bar(
            class_table, "class", "mean_ef1pct",
            title="Mean EF@1% by Protein Class")})

        # Per-tier mean EF@1% bar chart - new for Setting 2
        tier_ef1 = []
        if tier1_count > 0:
            tier_ef1.append(["Tier 1", float(np.mean(tier1_means["ef_1pct"]))])
        if tier2_count > 0:
            tier_ef1.append(["Tier 2", float(np.mean(tier2_means["ef_1pct"]))])
        if tier_ef1:
            tier_table = wandb.Table(columns=["tier", "mean_ef1pct"], data=tier_ef1)
            run.log({"ef1pct_by_tier": wandb.plot.bar(
                tier_table, "tier", "mean_ef1pct",
                title="Mean EF@1% by Tier")})

        run.finish()
        print(f"Results logged to wandb run: {args.model}_setting2_aggregate")


if __name__ == "__main__":
    main()
