from src.config.settings import load_settings
from src.extraction.claim_extractor import (
    extract_claim_records,
    save_claim_records,
    save_claim_records_csv,
)

settings = load_settings()

records = extract_claim_records(settings)

save_claim_records(settings.output_data_dir, records)
save_claim_records_csv(settings.output_data_dir, records)

print("\n========== EXTRACTION RESULTS ==========")
print("Total records:", len(records))

for record in records:
    print("\nClaim ID:", record.claim_id)
    print("Patient:", record.patient_name)
    print("Diagnosis:", record.diagnosis)
    print("Claim Amount:", record.total_claim_amount)
    print("Approved Amount:", record.approved_amount)
    print("Status:", record.claim_status)