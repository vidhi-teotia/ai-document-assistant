
import pandas as pd

from src.config.settings import load_settings

from src.agents.document_tools import (
    get_claim_record_by_id,
    get_claim_record_by_patient,
    filter_claim_by_status,
    get_data_quality_status,
    summarize_pipeline_outputs,
    ask_claim_documents,
    query_claim_dataset
)

settings = load_settings()


# ==================================================
# TEST 1: GET CLAIM RECORD BY ID
# ==================================================

print("\n========== TEST 1: CLAIM LOOKUP ==========")

result = get_claim_record_by_id(
    settings=settings,
    claim_id="CLM2024001847"
)

print(result)


# ==================================================
# TEST 2: GET CLAIM RECORD BY PATIENT
# ==================================================

print("\n========== TEST 2: PATIENT LOOKUP ==========")

csv_path = settings.output_data_dir / "claim_dataset.csv"
df = pd.read_csv(csv_path)

if "patient_name" in df.columns and not df["patient_name"].dropna().empty:

    patient_name = df["patient_name"].dropna().iloc[0]

    result = get_claim_record_by_patient(
        settings=settings,
        patient_name=patient_name
    )

    print(result)

else:
    print("No patient names available in CSV.")


# ==================================================
# TEST 3: FILTER CLAIMS BY STATUS
# ==================================================

print("\n========== TEST 3: STATUS FILTER ==========")

result = filter_claim_by_status(
    settings=settings,
    status="Approved"
)

print(result)


# ==================================================
# TEST 4: CHECK DATA QUALITY
# ==================================================

print("\n========== TEST 4: DATA QUALITY ==========")

result = get_data_quality_status(settings)

print(result)


# ==================================================
# TEST 5: SUMMARIZE PIPELINE OUTPUTS
# ==================================================

print("\n========== TEST 5: PIPELINE SUMMARY ==========")

result = summarize_pipeline_outputs(settings)

print(result)


# ==================================================
# TEST 6: SEARCH CLAIM DOCUMENTS USING RAG
# ==================================================

print("\n========== TEST 6: DOCUMENT SEARCH ==========")

result = ask_claim_documents(
    settings=settings,
    question="What is the approved amount for claim CLM2024001847?"
)

print(result)


# ==================================================
# TEST 7: QUERY CLAIM DATASET
# ==================================================

print("\n========== TEST 7: CLAIM DATASET QUERY ==========")

query_plan = {
    "columns": ["claim_id", "patient_name", "claim_status"],
    "filters": [],
    "sort_by": None,
    "ascending": True,
    "limit": 5
}

result = query_claim_dataset(
    settings=settings,
    query_plan=query_plan
)

print(result)


print("\n========== ALL INDIVIDUAL TOOL TESTS COMPLETED ==========")
