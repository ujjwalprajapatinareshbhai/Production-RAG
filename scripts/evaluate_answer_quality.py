import json
import re
import time
from pathlib import Path

from app.answer import answer_question


QUESTIONS = [
    {
        "id": "defect_tracking",
        "question": "What problem did Company X have with defect tracking?",
        "answerable": True,
        "reference_facts": [
            "No defect-tracking system was implemented.",
            "Developers tried to remember defects or wrote them on paper.",
        ],
        # Each inner list represents one required concept.
        # Phrases within a concept are alternatives.
        "required_concepts": [
            [
                [
                    "no defect tracking system",
                    "no defect-tracking system",
                    "no implemented defect tracking system",
                    "no defect tracking system was implemented",
                    "no defect tracking system implemented",
                    "does not have a defect tracking system implemented",
                    "does not have a defect tracking system",
                    "defect tracking system implemented to monitor detected or solved defects",
                    "no implemented system to keep track of defects",
                    "no system to keep track of defects",
                    "lacks a defect tracking system",
                    "did not have a defect tracking system",
                    "did not implement a defect tracking system",
                ],
            ],
            [
                [
                    "remember defects",
                    "try to remember defects",
                    "tried to remember defects",
                    "attempt to remember it",
                    "attempt to remember them",
                    "either try to remember defects",
                    "rely on memory",
                    "rely on their memory",
                    "rely on memory to track defects",
                ],
                [
                    "write them on paper",
                    "wrote them on paper",
                    "scribble them on paper",
                    "scribbled them on paper",
                    "write them on a piece of paper",
                    "wrote them on a piece of paper",
                    "scribble them on pieces of paper",
                    "scribbled them on pieces of paper",
                    "scribble it on a piece of paper",
                    "scribble them on a piece of paper",
                    "write defects on paper",
                    "scribble defects on pieces of paper",
                    "scribble defects on paper",
                    "scribble them on pieces of paper",
                    "either try to remember defects or scribble them on pieces of paper",
                ],
            ],
        ],
        "expected_pages": [37],
    },
    {
        "id": "development_lifecycle",
        "question": "Which software development lifecycle did Company X use?",
        "answerable": True,
        "reference_facts": [
            "Company X used the sequential Waterfall model.",
            "Testing occurred near the end of the process.",
        ],
        "required_concepts": [
            [
                [
                    "sequential waterfall model",
                    "waterfall model",
                    "sequential waterfall approach",
                ],
            ],
            [
                [
                    "testing phase",
                    "software testing",
                    "testing",
                ],
                [
                    "at the end of the process",
                    "near the end of the process",
                    "toward the end of the process",
                    "at the endof the process",
                    "end of the development process",
                ],
            ],
        ],
        "expected_pages": [6, 11, 32],
    },
    {
        "id": "annual_revenue",
        "question": "What was Company X's annual revenue in 2009?",
        "answerable": False,
        "reference_facts": [],
        "required_concepts": [],
        "expected_pages": [],
    },
]


# Supports citation groups such as [S1] and [S1, S2].
CITATION_GROUP_PATTERN = re.compile(r"\[([^\]]+)\]")
SOURCE_ID_PATTERN = re.compile(r"\bS(\d+)\b")


def extract_citation_ids(answer: str) -> set[str]:
    """Extract source IDs from citations such as [S1] and [S1, S2]."""
    citation_ids = set()

    for group in CITATION_GROUP_PATTERN.findall(answer):
        for number in SOURCE_ID_PATTERN.findall(group):
            citation_ids.add(f"S{number}")

    return citation_ids


def get_valid_source_ids(sources: list) -> set[str]:
    """Extract recognizable source IDs from source metadata."""
    valid_ids = set()

    for source in sources:
        if not isinstance(source, dict):
            continue

        source_id = source.get("source_id")

        if isinstance(source_id, str):
            if re.fullmatch(r"S\d+", source_id):
                valid_ids.add(source_id)

    return valid_ids


def get_returned_pages(sources: list) -> set[int]:
    """Collect unique, valid page numbers from returned sources."""
    returned_pages = set()

    for source in sources:
        if not isinstance(source, dict):
            continue

        pages = source.get("pages", [])

        if not isinstance(pages, list):
            continue

        for page in pages:
            # bool is a subclass of int, so explicitly exclude it.
            if isinstance(page, int) and not isinstance(page, bool):
                returned_pages.add(page)

    return returned_pages


def calculate_page_recall(
    expected_pages: list[int],
    returned_pages: set[int],
) -> dict | None:
    """
    Calculate recall against the explicitly listed expected pages.

    This measures expected-page coverage, not full retrieval quality.
    Returns None when no expected pages were defined.
    """
    expected = set(expected_pages)

    if not expected:
        return None

    matched_pages = expected.intersection(returned_pages)
    missing_pages = expected - returned_pages

    return {
        "matched_pages": sorted(matched_pages),
        "missing_pages": sorted(missing_pages),
        "recall": round(len(matched_pages) / len(expected), 3),
    }


def normalize_text(text: str) -> str:
    """Normalize text for simple phrase matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def calculate_reference_fact_coverage(
    reference_facts: list[str],
    answer: str,
    required_concepts: list[list[list[str]]] | None = None,
) -> dict:
    """
    Estimate reference-fact coverage using required concepts.

    Structure of required_concepts:
        One entry per reference fact.
        Each fact contains concept groups that are all required.
        Each concept group contains alternative phrases; at least one
        phrase in each group must appear in the answer.

    A fact matches only if every required concept group matches.

    This is a phrase-based screening metric, not semantic verification.
    """
    if not reference_facts:
        return {
            "matched_facts": [],
            "missing_facts": [],
            "coverage": None,
        }

    if required_concepts is None:
        return {
            "matched_facts": [],
            "missing_facts": reference_facts,
            "coverage": None,
        }

    if len(required_concepts) != len(reference_facts):
        raise ValueError(
            "Each reference fact must have a matching concept list."
        )

    normalized_answer = normalize_text(answer)
    matched_facts = []
    missing_facts = []

    for fact, concept_groups in zip(
        reference_facts,
        required_concepts,
    ):
        # Every group is required, but any alternative within a group
        # can satisfy that group.
        fact_matches = bool(concept_groups) and all(
            any(
                normalize_text(phrase) in normalized_answer
                for phrase in alternatives
                if normalize_text(phrase)
            )
            for alternatives in concept_groups
        )

        if fact_matches:
            matched_facts.append(fact)
        else:
            missing_facts.append(fact)

    return {
        "matched_facts": matched_facts,
        "missing_facts": missing_facts,
        "coverage": round(
            len(matched_facts) / len(reference_facts),
            3,
        ),
    }


def evaluate_question(item: dict) -> dict:
    """Run one question and collect answer-quality metrics."""
    started = time.perf_counter()

    try:
        result = answer_question(
            query=item["question"],
            limit=5,
            candidate_limit=20,
        )

        latency_seconds = time.perf_counter() - started

        if not isinstance(result, dict):
            raise TypeError(
                "Expected answer_question to return a dict, "
                f"got {type(result).__name__}"
            )

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        insufficient = result.get("insufficient_evidence")

        if not isinstance(answer, str):
            answer = ""

        if not isinstance(sources, list):
            sources = []

        citations = extract_citation_ids(answer)
        valid_source_ids = get_valid_source_ids(sources)

        # Check whether the answerability decision matches the reference.
        answerability_check = (
            isinstance(insufficient, bool)
            and (
                (item["answerable"] and not insufficient)
                or (not item["answerable"] and insufficient)
            )
        )

        # This checks ID validity, not whether citations support claims.
        citation_check = None

        if citations and valid_source_ids:
            citation_check = citations.issubset(valid_source_ids)

        returned_pages = get_returned_pages(sources)

        reference_fact_coverage = calculate_reference_fact_coverage(
            reference_facts=item["reference_facts"],
            answer=answer,
            required_concepts=item.get("required_concepts"),
        )

        page_recall = calculate_page_recall(
            expected_pages=item["expected_pages"],
            returned_pages=returned_pages,
        )

        return {
            "id": item["id"],
            "question": item["question"],
            "expected_answerable": item["answerable"],
            "reference_facts": item["reference_facts"],
            "expected_pages": item["expected_pages"],
            "returned_pages": sorted(returned_pages),
            "answer": answer,
            "insufficient_evidence": insufficient,
            "sources": sources,
            "citation_ids": sorted(citations),
            "valid_source_ids": sorted(valid_source_ids),
            "answerability_check": answerability_check,
            "citation_check": citation_check,
            "page_recall": page_recall,
            "reference_fact_coverage": reference_fact_coverage,
            "latency_seconds": round(latency_seconds, 2),
            "error": None,
            # Requires manual review against the source evidence.
            "manual_correctness": None,
        }

    except Exception as exc:
        return {
            "id": item["id"],
            "question": item["question"],
            "expected_answerable": item["answerable"],
            "reference_facts": item["reference_facts"],
            "expected_pages": item["expected_pages"],
            "returned_pages": [],
            "answer": None,
            "insufficient_evidence": None,
            "sources": [],
            "citation_ids": [],
            "valid_source_ids": [],
            "answerability_check": False,
            "citation_check": None,
            "page_recall": None,
            "reference_fact_coverage": {
                "matched_facts": [],
                "missing_facts": item["reference_facts"],
                "coverage": None,
            },
            "latency_seconds": round(
                time.perf_counter() - started, 2
            ),
            "error": f"{type(exc).__name__}: {exc}",
            "manual_correctness": None,
        }


def percentage(passed: int, total: int) -> str:
    """Format a metric as count and percentage."""
    if total == 0:
        return "N/A (0 applicable cases)"

    return f"{passed}/{total} ({passed / total:.1%})"


def main() -> None:
    results = []

    for item in QUESTIONS:
        print(f"\nEvaluating: {item['id']}")
        print(f"Question: {item['question']}")

        result = evaluate_question(item)
        results.append(result)

        print(f"Answer: {result['answer']}")
        print(
            "Answerability check: "
            f"{result['answerability_check']}"
        )
        print(f"Citation check: {result['citation_check']}")

        page_recall = result["page_recall"]

        if page_recall is None:
            print("Page recall: N/A (no expected pages)")
        else:
            print(
                f"Page recall: {page_recall['recall']:.1%} "
                f"(matched: {page_recall['matched_pages']}, "
                f"missing: {page_recall['missing_pages']})"
            )

        fact_coverage = result["reference_fact_coverage"]
        coverage = fact_coverage["coverage"]

        if coverage is None:
            print("Reference-fact coverage: N/A (no reference facts)")
        else:
            print(
                f"Reference-fact coverage: {coverage:.1%} "
                f"({len(fact_coverage['matched_facts'])}/"
                f"{len(item['reference_facts'])} facts matched)"
            )
            if fact_coverage["missing_facts"]:
                print(
                    "Missing reference facts: "
                    f"{fact_coverage['missing_facts']}"
                )

        if result["error"]:
            print(f"Error: {result['error']}")

        print(f"Latency: {result['latency_seconds']} seconds")

    output_path = Path("data/answer_quality_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    successful = [
        result for result in results
        if result["error"] is None
    ]

    answerability_passes = sum(
        result["answerability_check"]
        for result in successful
    )

    citation_applicable = [
        result for result in successful
        if result["citation_check"] is not None
    ]

    citation_passes = sum(
        result["citation_check"]
        for result in citation_applicable
    )

    page_applicable = [
        result for result in successful
        if result["page_recall"] is not None
    ]

    page_recall_values = [
        result["page_recall"]["recall"]
        for result in page_applicable
    ]

    average_page_recall = (
        sum(page_recall_values) / len(page_recall_values)
        if page_recall_values
        else None
    )

    coverage_applicable = [
        result for result in successful
        if result["reference_fact_coverage"]["coverage"] is not None
    ]

    coverage_values = [
        result["reference_fact_coverage"]["coverage"]
        for result in coverage_applicable
    ]

    average_fact_coverage = (
        sum(coverage_values) / len(coverage_values)
        if coverage_values
        else None
    )

    print("\n" + "=" * 60)
    print("ANSWER-QUALITY EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Questions: {len(results)}")
    print(f"Successful calls: {len(successful)}")
    print(f"Failed calls: {len(results) - len(successful)}")

    print(
        "Answerability decisions: "
        + percentage(answerability_passes, len(successful))
    )

    print(
        "Citation validity: "
        + percentage(citation_passes, len(citation_applicable))
    )

    if average_page_recall is None:
        print("Average expected-page recall: N/A")
    else:
        print(
            "Average expected-page recall: "
            f"{average_page_recall:.1%} "
            f"({len(page_applicable)} questions)"
        )

    if average_fact_coverage is None:
        print("Average reference-fact coverage: N/A")
    else:
        print(
            "Average reference-fact coverage: "
            f"{average_fact_coverage:.1%} "
            f"({len(coverage_applicable)} questions)"
        )

    if successful:
        average_latency = sum(
            result["latency_seconds"]
            for result in successful
        ) / len(successful)

        print(f"Average latency: {average_latency:.2f} seconds")

    print(
        "Note: reference-fact coverage checks required concepts using "
        "handwritten alternative phrases. It is a screening metric, "
        "not proof of semantic correctness or citation support."
    )
    print(
        "Manual factual correctness remains available for review "
        "against the retrieved source text."
    )

    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()