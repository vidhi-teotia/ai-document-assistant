
from pathlib import Path
import pandas as pd

from src.config.settings import load_settings
from src.rag.qa_pipeline import run_rag_question


# Initialize project settings once
settings = load_settings()


# ==================================================
# STAGE 1: VERIFY PDF TEXT EXTRACTION
# ==================================================

print("\n========== STAGE 1: PDF TEXT EXTRACTION ==========")

extracted_dir = settings.processed_data_dir / "extracted_text"

if extracted_dir.exists():
    claim_folders = [f for f in extracted_dir.iterdir() if f.is_dir()]

    print("Total claim folders:", len(claim_folders))

    for folder in claim_folders:
        print(folder.name)
else:
    print("Extracted text directory not found.")


# ==================================================
# STAGE 2: VERIFY CLEANING AND CHUNKING
# ==================================================

print("\n========== STAGE 2: CLEANING AND CHUNKING ==========")

cleaned_dir = settings.processed_data_dir / "cleaned_claims"

if cleaned_dir.exists():
    cleaned_files = list(cleaned_dir.glob("*.txt"))

    print("\nCleaned claims:", len(cleaned_files))

    for file in cleaned_files:
        print(file.name)
else:
    print("Cleaned claims directory not found.")


chunks_dir = settings.processed_data_dir / "chunks"

if chunks_dir.exists():
    chunk_files = list(chunks_dir.glob("*_chunks.json"))

    print("\nChunk files:", len(chunk_files))

    for file in chunk_files:
        print(file.name)
else:
    print("Chunks directory not found.")


# ==================================================
# STAGE 3: VERIFY EMBEDDINGS AND CHROMADB
# ==================================================

print("\n========== STAGE 3: EMBEDDINGS AND CHROMADB ==========")

vector_store_dir = settings.processed_data_dir / "vector_store"

if vector_store_dir.exists():
    print("Vector store exists:", vector_store_dir)

    for file in vector_store_dir.iterdir():
        print(file.name)
else:
    print("Vector store directory not found.")


# ==================================================
# STAGE 4: VERIFY STRUCTURED CLAIM EXTRACTION
# ==================================================

print("\n========== STAGE 4: STRUCTURED CLAIM EXTRACTION ==========")

records_path = settings.output_data_dir / "claim_records.json"

print("Claim records file exists:", records_path.exists())

if records_path.exists():
    print("File size:", records_path.stat().st_size, "bytes")
else:
    print("claim_records.json not found.")


# ==================================================
# STAGE 5: VERIFY CSV DATASET
# ==================================================

print("\n========== STAGE 5: CSV DATASET ==========")

csv_path = settings.output_data_dir / "claim_dataset.csv"

if csv_path.exists():
    df = pd.read_csv(csv_path)

    print("Total claim records:", len(df))
    print("\nClaim details:")

    available_columns = [
        col for col in ["claim_id", "patient_name"]
        if col in df.columns
    ]

    if available_columns:
        print(df[available_columns].to_string(index=False))
    else:
        print("Expected claim columns not found.")

else:
    print("claim_dataset.csv not found.")


# ==================================================
# STAGE 6: TEST RAG PIPELINE
# ==================================================

print("\n========== STAGE 6: RAG TESTING ==========")

questions = [
    "What is the total claim amount and approved amount for claim CLM2024001847?",
    "What is the diagnosis mentioned in claim CLM2024001847?",
    "Which hospital treated the patient for claim CLM2024001847?",
    "What is the claim status of CLM2024002193?"
]

for question in questions:
    print("\nQuestion:", question)

    try:
        answer = run_rag_question(
            settings=settings,
            question=question
        )

        print("Answer:", answer)

    except Exception as e:
        print("Error:", str(e))

    print("-" * 60)


print("\n========== ALL VERIFICATION AND RAG TESTS COMPLETED ==========")
