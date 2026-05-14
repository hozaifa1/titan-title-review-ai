"""Produce a simulated operator-edited copy of a TitleReviewSummary draft.

Mirrors the kind of edits a senior partner would make: tightens wording,
adds Book/Page citations in chain/liens sections, escalates green flags
to yellow when gaps exist, and adds explicit open questions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from titan.schemas import CitedSentence, TitleReviewSummary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = TitleReviewSummary.model_validate_json(args.baseline.read_text(encoding="utf-8"))

    # s1 — wording polish on vesting summary
    if summary.s1_vesting_and_estate.summary:
        s = summary.s1_vesting_and_estate.summary[0]
        summary.s1_vesting_and_estate.summary[0] = CitedSentence(
            text=(
                "Title is vested of record in the parties identified in Schedule A; "
                "confirm marital/entity capacity against the vesting deed prior to closing."
            ),
            citations=s.citations,
            confidence="medium",
        )

    # s3 — citation fix: add Book/Page anchoring
    if summary.s3_chain_of_title.summary:
        s = summary.s3_chain_of_title.summary[0]
        summary.s3_chain_of_title.summary[0] = CitedSentence(
            text=(
                "The chain of title is supported by recorded instruments; cite Deed Book "
                "and Page (and Instrument Number where present) for every link in the chain."
            ),
            citations=s.citations,
            confidence="high",
        )

    # s4 — fact correction on liens + risk escalation
    if summary.s4_open_encumbrances_and_liens.summary:
        s = summary.s4_open_encumbrances_and_liens.summary[0]
        summary.s4_open_encumbrances_and_liens.summary[0] = CitedSentence(
            text=(
                "Identify each open lien with creditor, debtor, original amount, recording "
                "Book and Page, and current status; treat unreleased records as open until proven otherwise."
            ),
            citations=s.citations,
            confidence="high",
        )
    if "green" in summary.s4_open_encumbrances_and_liens.flags:
        summary.s4_open_encumbrances_and_liens.flags = ["yellow"]

    # s6 — house style for Schedule B-I
    if summary.s6_requirements_schedule_b_i.summary:
        s = summary.s6_requirements_schedule_b_i.summary[0]
        summary.s6_requirements_schedule_b_i.summary[0] = CitedSentence(
            text=(
                "Schedule B-I requirements must list payoff, release, execution, and "
                "recordation steps and identify the responsible party (seller/buyer/lender)."
            ),
            citations=s.citations,
            confidence="high",
        )

    # s7 — house format for exceptions
    if summary.s7_exceptions_schedule_b_ii.summary:
        s = summary.s7_exceptions_schedule_b_ii.summary[0]
        summary.s7_exceptions_schedule_b_ii.summary[0] = CitedSentence(
            text=(
                "Schedule B-II must enumerate standard and special exceptions and note "
                "whether each is removable upon survey or affidavit."
            ),
            citations=s.citations,
            confidence="high",
        )

    # s8 — addition: a meaningful gap about tax delinquency
    if "Confirm no delinquent property taxes via current tax certificate." not in summary.s8_taxes_and_survey_matters.gaps:
        summary.s8_taxes_and_survey_matters.gaps.append(
            "Confirm no delinquent property taxes via current tax certificate."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"baseline": str(args.baseline), "edited": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
