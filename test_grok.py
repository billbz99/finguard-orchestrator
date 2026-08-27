from src.graph.schemas import TransactionExtraction
from src.llm.client import get_llm


llm = get_llm()

structured_llm = llm.with_structured_output(TransactionExtraction)

result = structured_llm.invoke(
    """
    Extract AML-relevant information from this audit request.

    Only extract information explicitly present in the request.
    Do not invent missing values.

    Request:
    Audit wire TXN-555123 for possible structuring under FINRA Rule 3310.
    """
)

print(result)

print()
print(result.model_dump())