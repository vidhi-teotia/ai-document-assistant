from src.config.settings import load_settings
from  src.extraction.claim_extractor import (
    build_extraction_prompt,
)
from langchain_ollama import ChatOllama

settings = load_settings()

cleaned_dir = settings.processed_data_dir / "cleaned_claims"
file = sorted(cleaned_dir.glob("*.txt"))[0]

claim_text = file.read_text(encoding="utf-8")

model = ChatOllama(
    model=settings.chat_model,
    temperature=0,
    format="json",
)

response = model.invoke(build_extraction_prompt(claim_text))

print("File:", file.name)
print("\nRAW OLLAMA RESPONSE:")
print(response.content)
