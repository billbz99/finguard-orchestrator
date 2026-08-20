# tests/eval_suite.py

import pytest
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

load_dotenv()


def test_aml_generation_quality():
    """Validates AML compliance output quality against a gold standard scenario."""
    
    test_case = LLMTestCase(
        input="Analyze transaction references TXN-093 for structuring compliance violations.",
        actual_output=(
            "Transaction TXN-093 splits a $34,000 transfer into four distinct $8,500 batches "
            "over 48 hours to evade the $10,000 Currency Transaction Reporting requirements. "
            "This violates FINRA Rule 3310(a)."
        ),
        # Changed from retrieved_context to retrieval_context
        retrieval_context=[
            (
                "FINRA Rule 3310(a) mandates Member firms must implement transactional monitoring "
                "designed to detect structuring, defined as dividing single transactions into "
                "consecutive segments below $10,000 reporting thresholds."
            )
        ],
    )

    # 1. Instantiate evaluation metrics with target thresholds
    faithfulness_metric = FaithfulnessMetric(threshold=0.85, model="gpt-4o")
    relevancy_metric = AnswerRelevancyMetric(threshold=0.80, model="gpt-4o")

    # 2. Run assertions
    assert_test(test_case, [faithfulness_metric, relevancy_metric])


if __name__ == "__main__":
    pytest.main(["-v", "tests/eval_suite.py"])