# -*- coding: utf-8 -*-
"""
aggregate_results.py
====================
Reads per-seed CSVs from results/setting1/{model_name}/per_seed/
and produces a single summary CSV with format:

    target | name | class | n_actives | n_decoys | ef_1pct | ef_5pct | ...

where each metric is formatted as "mean +/- std"

Usage:
    python aggregate_results.py --model morgan_rf
    python aggregate_results.py --model tanimoto_nn
    python aggregate_results.py --model gin

Optional:
    --results_dir /path/to/results/setting1
    --seeds 2026 2027 2028
    --out   /path/to/output.csv
"""

import argparse
import csv
import glob
import os
import numpy as np
import wandb

WANDB_ENTITY  = "kumar-surbhi1294-university-of-texas-at-dallas"
WANDB_PROJECT = "LBVS"

# ---------------------------------------------------------------------------
# Target metadata from topU_95.xlsx
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


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))[0]


def fmt(mean, std):
    return f"{mean:.3f} \u00b1 {std:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--results_dir", default="/groups/bcoskunuzer/sxk230046/LBVS/results/setting1")
    parser.add_argument("--seeds",       nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--out",         default=None)
    parser.add_argument("--no_wandb",    action="store_true", help="Disable wandb logging.")
    args = parser.parse_args()

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

    rows      = []
    missing   = []
    all_means = {m: [] for m in METRICS}

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

        row = {
            "target":    target,
            "name":      name,
            "class":     cls,
            "n_actives": n_actives,
            "n_decoys":  n_decoys,
        }

        for m in METRICS:
            vals = [float(d[m]) for d in seed_data]
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=0))
            row[m] = fmt(mean, std)
            all_means[m].append(mean)

        rows.append(row)

    # Overall row
    if rows:
        overall = {
            "target":    "OVERALL",
            "name":      "",
            "class":     "ALL",
            "n_actives": sum(r["n_actives"] for r in rows),
            "n_decoys":  sum(r["n_decoys"]  for r in rows),
        }
        for m in METRICS:
            means = all_means[m]
            overall[m] = fmt(float(np.mean(means)), float(np.std(means, ddof=0)))
        rows.append(overall)

    fieldnames = ["target", "name", "class", "n_actives", "n_decoys"] + METRICS

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary saved to: {out_path}")
    print(f"Targets included: {len(rows) - 1}")

    if missing:
        print(f"WARNING: {len(missing)} missing seed CSVs skipped: {missing}")

    # -- Log to wandb ---------------------------------------------------------
    if not args.no_wandb and rows:
        data_rows = [r for r in rows if r["target"] != "OVERALL"]
        overall   = next(r for r in rows if r["target"] == "OVERALL")

        run = wandb.init(
            entity  = WANDB_ENTITY,
            project = WANDB_PROJECT,
            group   = "aggregate",
            name    = f"{args.model}_aggregate",
            job_type= "aggregate",
            config  = {"model": args.model, "n_targets": len(data_rows), "seeds": args.seeds},
        )

        # Log overall summary scalars - easy to compare across models
        for m in METRICS:
            mean_val = float(np.mean(all_means[m]))
            run.summary[f"overall/{m}_mean"] = mean_val

        # Log full per-target table
        columns = ["target", "name", "class", "n_actives", "n_decoys"] + METRICS
        table   = wandb.Table(columns=columns)
        for r in data_rows:
            table.add_data(*[r[c] for c in columns])
        run.log({"per_target_results": table})

        # Log per-class mean EF@1% bar chart
        classes     = sorted(set(r["class"] for r in data_rows))
        class_ef1   = []
        for cls in classes:
            cls_rows = [r for r in data_rows if r["class"] == cls]
            cls_means = []
            for r in cls_rows:
                val_str = r["ef_1pct"].split()[0]
                cls_means.append(float(val_str))
            class_ef1.append([cls, float(np.mean(cls_means))])

        class_table = wandb.Table(columns=["class", "mean_ef1pct"], data=class_ef1)
        run.log({"ef1pct_by_class": wandb.plot.bar(class_table, "class", "mean_ef1pct",
                                                     title="Mean EF@1% by Protein Class")})

        run.finish()
        print(f"Results logged to wandb run: {args.model}_aggregate")


if __name__ == "__main__":
    main()