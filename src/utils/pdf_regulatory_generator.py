"""
Regulatory Compliance PDF Generator
Generates a structured, multi-clause PDF containing authentic FINRA Rule 3310,
FinCEN advisories, and ADGM AML guidelines for vector ingestion and semantic chunking.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable


def create_regulatory_pdf(output_pdf_path: str):
    """Generates a structured regulatory PDF for compliance grounding."""
    output_path = Path(output_pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles matching legal formatting
    title_style = ParagraphStyle(
        "LegalTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=0,
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "LegalH1",
        parent=styles["Heading1"],
        fontSize=13,
        leading=16,
        spaceBefore=14,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "LegalBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )

    story = []

    # Title Banner
    story.append(
        Paragraph("FINRA & FinCEN Regulatory Compliance Manual", title_style)
    )
    story.append(
        Paragraph(
            "<b>Document ID:</b> REG-2026-AML-01 | <b>Effective Date:</b> January 2026",
            body_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, spaceAfter=12))

    # Section 1: FINRA Rule 3310
    story.append(
        Paragraph(
            "1. FINRA Rule 3310: Anti-Money Laundering Compliance Program",
            h1_style,
        )
    )
    story.append(
        Paragraph(
            "Each member firm must develop and implement a written Anti-Money Laundering (AML) program "
            "reasonably designed to achieve and monitor compliance with the requirements of the Bank Secrecy Act (BSA). "
            "The program must be approved in writing by a member of senior management.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>1.1 Transaction Monitoring & Red Flags:</b> Member firms must establish procedures to detect and "
            "report suspicious activity across all customer accounts. Red flags include unusual wire transfer activity, "
            "frequent transactions just under the $10,000 Currency Transaction Reporting (CTR) threshold, "
            "and transfers involving high-risk jurisdictions without clean commercial rationale.",
            body_style,
        )
    )

    # Section 2: FinCEN Guidance on Structuring
    story.append(
        Paragraph(
            "2. FinCEN Advisory: Detection & Prevention of Structuring (31 U.S.C. 5324)",
            h1_style,
        )
    )
    story.append(
        Paragraph(
            "Structuring occurs when an individual breaks down a large sum of cash or wire transfers into multiple "
            "consecutive transactions below the $10,000 threshold specifically to evade mandatory Currency Transaction "
            "Reports (CTRs) or Suspicious Activity Reports (SARs).",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>2.1 Smurfing & Fan-In / Fan-Out Patterns:</b> Financial institutions must identify 'Smurfing' patterns "
            "where multiple distinct individuals deposit small sums into a central account (Fan-In), followed by rapid "
            "transfers to offshore or high-risk accounts (Fan-Out). Such patterns require filing a SAR within 30 calendar days.",
            body_style,
        )
    )

    # Section 3: High-Risk Jurisdictions & Cross-Border Rules
    story.append(
        Paragraph(
            "3. Cross-Border Wire Guidance & High-Risk Jurisdictions",
            h1_style,
        )
    )
    story.append(
        Paragraph(
            "International transfers involving tax havens, high-risk offshore jurisdictions, or non-cooperative "
            "countries must include complete SWIFT ISO 20022 message blocks, including ordering customer identity (Field :50K:) "
            "and beneficiary customer identity (Field :59:). Incomplete or obfuscated payment memos require immediate freeze and review.",
            body_style,
        )
    )

    doc.build(story)
    print(f"✅ Generated regulatory PDF at: {output_path.resolve()}")


if __name__ == "__main__":
    create_regulatory_pdf("data/raw/finra_rule_3310.pdf")