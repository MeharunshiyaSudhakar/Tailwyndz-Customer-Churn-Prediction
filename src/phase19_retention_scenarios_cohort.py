"""Retention scenarios on the PRIMARY (signup-cohort) test set.

Replaces the earlier random-split scenario as the main retention analysis. It
asks: if the business contacted a chosen group of members, how many of the
members who really churned would be reached, and how many could be saved under
stated effectiveness assumptions?

Strategies compared on the same 485 test-cohort members ("reachable" = still
buying within the last month of the feature window; members who already stopped
buying cannot be "retained", they would need a win-back offer):
    1. Blanket                  - contact everyone (current practice)
    2. Model top 20%, any member - includes already-lapsed members (reference only)
    3. Model top 20%, reachable  - same contact budget, reachable members only
    4. Support outreach          - proactive call to reachable members with >=1
                                   support ticket in the late window (Nov 2023 - Mar 2024)
    5. Targeted coupon           - reachable members flagged by the model
    6. Support outreach + flagged - reachable, >=1 ticket AND flagged by the model

Impact = assumed effectiveness x churners in the targeted group. Effectiveness
is an ASSUMPTION (no campaign data exists), so a grid is shown. Costs are
expressed relative to the value of one retained member (cost ratio r), so no
invented currency figures are needed.

A model what-if (re-scoring after removing support tickets) is also reported;
it shows what the model has learned, not what a real intervention would do.

Run after phase18_cohort_validation.py.
Outputs: reports/phase19/
"""
from pathlib import Path

import numpy as np
import pandas as pd

from phase18_cohort_validation import (
    EXCLUDE, FEATURE_FILE, SEED, TARGET, TEST_COHORTS, TRAIN_COHORTS,
    VAL_COHORTS, estimators, fit,
)

ROOT = Path(__file__).resolve().parents[1]
PRED_FILE = ROOT / "reports/phase18/phase18_test_predictions.csv"
OUT = ROOT / "reports/phase19"
EFFECTIVENESS = [0.05, 0.10, 0.20, 0.30]
COST_RATIOS = [0.02, 0.05, 0.10]       # contact cost / value of one retained member
RECOMMENDED = "6. Support outreach + model-flagged (reachable)"
LATE_TICKETS = "LATE_PERIOD_SUPPORT_TICKETS"
RECENCY = "MONTHS_SINCE_LAST_OBSERVED_ACTIVITY"


def boot_ci(values, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if len(values) == 0:
        return (np.nan, np.nan)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(FEATURE_FILE, parse_dates=["SIGNUP_DATE"])
    pred = pd.read_csv(PRED_FILE)
    d = pred.merge(feats, on="CUSTOMER_ID", how="left", suffixes=("", "_F"), validate="1:1")
    d["TICKETS"] = d[LATE_TICKETS].fillna(0)
    d["STILL_ACTIVE"] = d[RECENCY].fillna(0) <= 1
    y = d["ACTUAL_CHURNED"].values
    base_rate, n, total_churners = y.mean(), len(d), int(y.sum())

    risk_rank = d["PREDICTED_CHURN_PROBABILITY"].rank(method="first", ascending=False)
    budget = int(round(0.20 * n))
    reach_rank = d["PREDICTED_CHURN_PROBABILITY"].where(d["STILL_ACTIVE"]).rank(method="first", ascending=False)
    flagged = d["PREDICTED_CHURNED"] == 1
    strategies = {
        "1. Blanket (everyone)": np.ones(n, dtype=bool),
        "2. Model top 20%, any member (incl. already lapsed)": (risk_rank <= budget).values,
        "3. Model top 20%, reachable members only": (reach_rank <= budget).values,
        "4. Support outreach (reachable, >=1 late-window ticket)": ((d["TICKETS"] >= 1) & d["STILL_ACTIVE"]).values,
        "5. Targeted coupon (reachable, model-flagged)": (flagged & d["STILL_ACTIVE"]).values,
        "6. Support outreach + model-flagged (reachable)": ((d["TICKETS"] >= 1) & flagged & d["STILL_ACTIVE"]).values,
    }
    lapsed_in_top = int((~d["STILL_ACTIVE"] & (risk_rank <= budget)).sum())

    # Support-ticket relationship (test cohort) for context.
    # Still-active members only: members who already stopped buying raise no tickets,
    # which would otherwise make "0 tickets" look riskier than it is.
    act = d[d["STILL_ACTIVE"]]
    bins, labels = [-1, 0, 1, 2, 99], ["0", "1", "2", "3+"]
    rel = act.groupby(pd.cut(act["TICKETS"], bins, labels=labels), observed=True).agg(
        TEST_MEMBERS=("ACTUAL_CHURNED", "size"), TEST_CHURN_RATE=("ACTUAL_CHURNED", "mean")).reset_index()
    rel.columns = ["LATE_WINDOW_TICKETS"] + list(rel.columns[1:])
    tr = feats[(feats.INSUFFICIENT_HISTORY_FLAG == 0)]
    tr = tr[tr["SIGNUP_DATE"].dt.to_period("Q").astype(str).isin(TRAIN_COHORTS)]
    tr = tr[tr[RECENCY].fillna(0) <= 1]
    trg = tr.groupby(pd.cut(tr[LATE_TICKETS].fillna(0), bins, labels=labels), observed=True)[TARGET].agg(["size", "mean"])
    rel["TRAIN_MEMBERS"] = trg["size"].values
    rel["TRAIN_CHURN_RATE"] = trg["mean"].values
    rel.to_csv(OUT / "phase19_support_ticket_relationship.csv", index=False)

    # Strategy comparison + effectiveness / cost grid.
    comp_rows, grid_rows = [], []
    for name, mask in strategies.items():
        contacts, churners = int(mask.sum()), int(y[mask].sum())
        rate = churners / contacts if contacts else np.nan
        lo, hi = boot_ci(y[mask])
        comp_rows.append({
            "STRATEGY": name, "CONTACTS": contacts, "SHARE_OF_MEMBERS": contacts / n,
            "CHURNERS_IN_GROUP": churners, "GROUP_CHURN_RATE": rate,
            "RATE_CI_LOW": lo, "RATE_CI_HIGH": hi,
            "LIFT_VS_BASE_RATE": rate / base_rate if contacts else np.nan,
            "SHARE_OF_ALL_CHURNERS_REACHED": churners / total_churners})
        for e in EFFECTIVENESS:
            prevented = e * churners
            row = {"STRATEGY": name, "ASSUMED_EFFECTIVENESS": e, "CONTACTS": contacts,
                   "CHURN_PREVENTED": prevented,
                   "CONTACTS_PER_CHURN_PREVENTED": contacts / prevented if prevented else np.nan,
                   "CHURN_AFTER_INTERVENTION_ALL_MEMBERS": total_churners - prevented}
            for r in COST_RATIOS:
                row[f"NET_VALUE_r{r:.2f}"] = prevented - r * contacts   # in retained-member values
            grid_rows.append(row)
        comp_rows[-1]["BREAK_EVEN_EFFECTIVENESS_r0.05"] = 0.05 * contacts / churners if churners else np.nan
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT / "phase19_strategy_comparison.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(OUT / "phase19_effectiveness_cost_grid.csv", index=False)

    # Target list for the recommended strategy (5).
    rec_mask = strategies[RECOMMENDED]
    target = d.loc[rec_mask, ["CUSTOMER_ID", "PREDICTED_CHURN_PROBABILITY", "TICKETS",
                              "TOP_DRIVER_1", "TOP_DRIVER_2", "TOP_DRIVER_3"]].rename(
        columns={"TICKETS": "LATE_WINDOW_SUPPORT_TICKETS"})
    target.sort_values("PREDICTED_CHURN_PROBABILITY", ascending=False).to_csv(
        OUT / "phase19_recommended_target_list.csv", index=False)

    # Model what-if: re-score after removing late-window support tickets.
    cols = [c for c in feats.columns if c not in EXCLUDE | {"COHORT"}]
    feats["COHORT"] = feats["SIGNUP_DATE"].dt.to_period("Q").astype(str)
    el = feats[feats.INSUFFICIENT_HISTORY_FLAG == 0]
    train = el[el.COHORT.isin(TRAIN_COHORTS)]
    test = el[el.COHORT.isin(TEST_COHORTS)].set_index("CUSTOMER_ID").loc[d["CUSTOMER_ID"]].reset_index()
    chosen = d["MODEL"].iloc[0]
    pipe = fit(estimators()[chosen], train[cols], train[TARGET].values)
    thr = d.loc[flagged, "PREDICTED_CHURN_PROBABILITY"].min() if flagged.any() else 0.5
    base_p = pipe.predict_proba(test[cols])[:, 1]
    assert np.allclose(base_p, d["PREDICTED_CHURN_PROBABILITY"].values, atol=1e-5), "model re-fit differs from phase 18"
    whatif_rows = []
    has_tk = (test[LATE_TICKETS].fillna(0) >= 1).values
    for label, factor in (("Halve late-window support tickets", 0.5), ("Remove all late-window support tickets", 0.0)):
        mod = test[cols].copy()
        new_late = np.floor(mod[LATE_TICKETS] * factor)
        removed = (mod[LATE_TICKETS] - new_late).fillna(0)
        mod[LATE_TICKETS] = new_late
        mod["TOTAL_SUPPORT_TICKETS_PRE_APR2024"] = (mod["TOTAL_SUPPORT_TICKETS_PRE_APR2024"] - removed).clip(lower=0)
        new_p = pipe.predict_proba(mod)[:, 1]
        whatif_rows.append({
            "WHAT_IF": label, "MEMBERS_WITH_TICKETS": int(has_tk.sum()),
            "MEAN_RISK_BEFORE_AFFECTED": base_p[has_tk].mean(), "MEAN_RISK_AFTER_AFFECTED": new_p[has_tk].mean(),
            "MEAN_RISK_CHANGE_AFFECTED": (new_p - base_p)[has_tk].mean(),
            "SUM_RISK_BEFORE_ALL": base_p.sum(), "SUM_RISK_AFTER_ALL": new_p.sum(),
            "FLAGGED_BEFORE": int((base_p >= thr).sum()), "FLAGGED_AFTER": int((new_p >= thr).sum())})
    whatif = pd.DataFrame(whatif_rows)
    whatif.to_csv(OUT / "phase19_model_whatif.csv", index=False)

    # Report.
    L = ["PHASE 19 - RETENTION SCENARIOS ON THE COHORT-SPLIT TEST SET", "=" * 66,
         f"Test cohort (signups {TEST_COHORTS[0]}-{TEST_COHORTS[-1]}): {n} members, {total_churners} churners, base churn rate {base_rate:.1%}.",
         f"Model: {chosen} (from phase 18). Reachable = still buying within the last month of the feature window.", "",
         f"Note: {lapsed_in_top} of the {budget} highest-risk members had already stopped buying (all churned); they inflate strategy 2.", "",
         "SUPPORT TICKETS AND CHURN, still-active members (late window Nov 2023 - Mar 2024)", "-" * 66,
         rel.round(3).to_string(index=False), "",
         "STRATEGY COMPARISON (observed churn among members each strategy would contact)", "-" * 66]
    for _, r in comp.iterrows():
        L.append(f"{r.STRATEGY}: contacts={r.CONTACTS} ({r.SHARE_OF_MEMBERS:.0%}), churners={r.CHURNERS_IN_GROUP}, "
                 f"churn rate={r.GROUP_CHURN_RATE:.1%} (95% CI {r.RATE_CI_LOW:.1%}-{r.RATE_CI_HIGH:.1%}), lift={r.LIFT_VS_BASE_RATE:.1f}x, "
                 f"reaches {r.SHARE_OF_ALL_CHURNERS_REACHED:.0%} of all churners")
    L += ["", "CHURN PREVENTED UNDER ASSUMED EFFECTIVENESS (effectiveness = share of would-be churners saved)", "-" * 66]
    g = pd.DataFrame(grid_rows)
    for name in strategies:
        s = g[g.STRATEGY == name]
        L.append(name + ": " + "; ".join(f"{int(r.ASSUMED_EFFECTIVENESS*100)}% -> {r.CHURN_PREVENTED:.1f} saved" for _, r in s.iterrows()))
    L += ["", "NET VALUE in units of 'one retained member' at contact-cost ratio r (positive = worth doing)", "-" * 66]
    for r_ in COST_RATIOS:
        L.append(f"r = {r_:.2f} (cost per contact is {r_:.0%} of a retained member's value), effectiveness 20%:")
        for name in strategies:
            row = g[(g.STRATEGY == name) & (g.ASSUMED_EFFECTIVENESS == 0.20)].iloc[0]
            L.append(f"   {name}: net = {row[f'NET_VALUE_r{r_:.2f}']:.1f}")
    L += ["", "MODEL WHAT-IF (association only, not a causal effect)", "-" * 66, whatif.round(3).T.to_string(), "",
          "LIMITATIONS",
          "- Effectiveness values are assumptions; no campaign has been run. A randomised pilot is needed.",
          "- Groups are small (see confidence intervals); results are indicative.",
          "- Churners in the group are counted from known outcomes, so this is a retrospective simulation.",
          "- Already-lapsed members (no purchase for 2+ months) churn almost certainly; they need a win-back offer, not prevention."]
    (OUT / "phase19_retention_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
