
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parent.parent
PPT_PATH = ROOT / "presentation" / "final_presentation.pptx"
REPORT_PATH = ROOT / "reports" / "phase14" / "final_report.md"

def add_slide(prs, title, bullets, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title

    body = slide.placeholders[1].text_frame
    body.clear()

    if subtitle:
        p = body.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(18)
        p.font.bold = True
        p.space_after = Pt(12)

    for i, text in enumerate(bullets):
        p = body.add_paragraph() if (subtitle or i > 0) else body.paragraphs[0]
        p.text = text
        p.level = 0
        p.font.size = Pt(17)
        p.space_after = Pt(9)

    return slide

def main():
    if not PPT_PATH.exists():
        raise FileNotFoundError(f"Presentation not found: {PPT_PATH}")

    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Final report not found: {REPORT_PATH}")

    report = REPORT_PATH.read_text(encoding="utf-8")
    required = [
        "Phase 15",
        "Phase 16",
        "Phase 17",
        "0.9036",
        "0.9649",
    ]
    missing = [term for term in required if term not in report]
    if missing:
        raise ValueError(
            "The report is missing expected validation content: "
            + ", ".join(missing)
        )

    # Open the existing deck so all original slides are preserved.
    prs = Presentation(str(PPT_PATH))
    original_count = len(prs.slides)

    add_slide(
        prs,
        "Additional Validation: Phases 15–17",
        [
            "Purpose: test sensitivity to recency features and assess variation across repeated stratified splits.",
            "The results remain exploratory; stronger scores alone do not establish deployment readiness.",
            "All original slides are retained in this updated deck.",
        ],
    )

    add_slide(
        prs,
        "Phase 15 — Recency Sensitivity",
        [
            "Balanced model with existing recency: F1 = 0.9008; ROC-AUC = 0.9669.",
            "Without recency: F1 = 0.7671; ROC-AUC = 0.9121.",
            "Recency contributes substantial predictive information, but its relationship to the churn label requires further audit.",
            "Signup-cohort check: F1 = 0.8794; ROC-AUC = 0.9816. This is not a future-time holdout.",
        ],
    )

    add_slide(
        prs,
        "Phase 16 — Positive-Transaction Recency",
        [
            "Alternative: months since the last positive-transaction month in the feature window.",
            "An explicit indicator represents customers with no positive-transaction month.",
            "Accuracy = 0.9484; balanced accuracy = 0.9373.",
            "Precision = 0.9167; recall = 0.9098; F1 = 0.9132.",
            "ROC-AUC = 0.9674; PR-AUC = 0.9503.",
            "The alternative still needs auditing against the target definition.",
        ],
    )

    add_slide(
        prs,
        "Phase 17 — Repeated Stratified Validation",
        [
            "Five stratified 80/20 splits, using seeds 42–46; identical splits compared across models.",
            "Positive-transaction recency: F1 = 0.9036 ± 0.0158.",
            "Existing observed-activity recency: F1 = 0.8913 ± 0.0205.",
            "No recency: F1 = 0.7858 ± 0.0305.",
            "Positive-transaction recency: ROC-AUC = 0.9772 ± 0.0091; PR-AUC = 0.9649 ± 0.0120.",
            "Repeated splits are not independent replications and do not establish statistical significance.",
        ],
    )

    add_slide(
        prs,
        "Interpretation and Limitations",
        [
            "Strong recency signals may partly reflect a target-proxy relationship; this is a concern to investigate, not proof of leakage.",
            "The churn target covers April–June 2024; predictors use activity through March 2024.",
            "A cutoff check alone does not prove independence from target construction.",
            "Signup-cohort checks and repeated random splits do not replace future-period validation.",
            "Results do not establish causal impact or deployment-ready probabilities.",
        ],
    )

    add_slide(
        prs,
        "Next Steps Before Operational Use",
        [
            "Confirm churn-label generation and missing-month semantics with the dataset owner.",
            "Audit the lineage and timing of recency features.",
            "Preserve original labels and document unresolved data-quality concerns.",
            "Evaluate on a defensible time-based holdout after resolving label and feature timing.",
            "Reassess calibration and conduct a randomized retention pilot before operational use.",
        ],
    )

    # Save only after all slides have been added successfully.
    prs.save(str(PPT_PATH))

    # Reopen and verify the saved deck.
    verified = Presentation(str(PPT_PATH))
    expected_count = original_count + 6
    if len(verified.slides) != expected_count:
        raise RuntimeError(
            f"Slide-count verification failed: expected {expected_count}, "
            f"found {len(verified.slides)}"
        )

    print(f"Original slides preserved: {original_count}")
    print(f"New validation slides added: 6")
    print(f"Verified total slide count: {len(verified.slides)}")
    print(f"Updated presentation: {PPT_PATH}")
    print("Backup remains separate and unchanged.")

if __name__ == "__main__":
    main()
