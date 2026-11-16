
import json
from pathlib import Path
import pandas as pd

from src.config.settings import AppSettings
from src.rag.qa_pipeline import run_rag_question


MAX_QUERY_LIMIT = 20
DEFAULT_QUERY_LIMIT = 10


# Convert pandas missing values into JSON-friendly Python values.
def normalize_record_value(value):
    if pd.isna(value):
        return None
    return value


# Normalize one CSV claim record before returning it from an agent tool.
def normalize_claim_record(record: dict) -> dict:
    return {
        column_name: normalize_record_value(value)
        for column_name, value in record.items()
    }


# Load JSON file.
def load_json_file(file_path: Path) -> dict | list:
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"JSON file is empty: {file_path}")

    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON file: {file_path}"
        ) from error


# Load claim dataset.
def load_claim_dataset(settings: AppSettings) -> pd.DataFrame | None:
    dataset_path = settings.output_data_dir / "claim_dataset.csv"

    if not dataset_path.exists():
        return None

    dataframe = pd.read_csv(dataset_path)

    return dataframe


# Summarize pipeline outputs.
def summarize_pipeline_outputs(settings: AppSettings) -> dict:
    summary_path = settings.output_data_dir / "processing_summary.json"

    if not summary_path.exists():
        return {"error": "processing_summary.json was not found"}

    return load_json_file(summary_path)


# Get data quality report status.
def get_data_quality_status(settings: AppSettings) -> dict:
    report_path = settings.output_data_dir / "data_quality_report.json"

    if not report_path.exists():
        return {"error": "data_quality_report.json was not found"}

    return load_json_file(report_path)


# Filter claims by status.
def filter_claim_by_status(settings: AppSettings, status: str) -> list[dict]:
    dataframe = load_claim_dataset(settings=settings)

    if dataframe is None:
        return [{"error": "claim_dataset.csv was not found"}]

    if "claim_status" not in dataframe.columns:
        return [{"error": "claim_status column was not found"}]

    filtered = dataframe[
        dataframe["claim_status"].astype("string").str.strip().str.lower()
        == str(status).strip().lower()
    ]

    return [
        normalize_claim_record(record)
        for record in filtered.to_dict(orient="records")
    ]


# Get claim fields.
def get_claim_fields(settings: AppSettings, fields: list[str]) -> list[dict]:
    dataframe = load_claim_dataset(settings=settings)

    if dataframe is None:
        return [{"error": "claim_dataset.csv was not found"}]

    available_fields = set(dataframe.columns)
    selected_fields = [
        field for field in fields if field in available_fields
    ]

    if not selected_fields:
        return [
            {
                "error": "No requested fields were found in claim_dataset.csv",
                "available_fields": list(dataframe.columns),
            }
        ]

    return [
        normalize_claim_record(record)
        for record in dataframe[selected_fields].to_dict(orient="records")
    ]


# Get claim record by ID.
def get_claim_record_by_id(settings: AppSettings, claim_id: str) -> dict:
    dataframe = load_claim_dataset(settings=settings)

    if dataframe is None:
        return {"error": "claim_dataset.csv was not found"}

    if "claim_id" not in dataframe.columns:
        return {"error": "claim_id column was not found in claim_dataset.csv"}

    filtered = dataframe[
        dataframe["claim_id"].astype("string").str.strip().str.lower()
        == str(claim_id).strip().lower()
    ]

    if filtered.empty:
        return {"error": f"No claim record found for claim_id={claim_id}"}

    return normalize_claim_record(filtered.iloc[0].to_dict())


# Get claim record by patient.
def get_claim_record_by_patient(
    settings: AppSettings,
    patient_name: str
) -> list[dict] | dict:

    dataframe = load_claim_dataset(settings=settings)

    if dataframe is None:
        return {"error": "claim_dataset.csv was not found"}

    if "patient_name" not in dataframe.columns:
        return {"error": "patient_name column was not found"}

    filtered = dataframe[
        dataframe["patient_name"].astype("string").str.contains(
            str(patient_name),
            case=False,
            na=False,
            regex=False,
        )
    ]

    if filtered.empty:
        return {
            "error": f"No claim record found for patient_name={patient_name}"
        }

    return [
        normalize_claim_record(record)
        for record in filtered.to_dict(orient="records")
    ]


# Normalize query operators.
def normalize_query_operator(operator: str) -> str:
    operator_map = {
        "=": "equals",
        "==": "equals",
        "eq": "equals",
        "equals": "equals",
        "not_equals": "not_equals",
        "!=": "not_equals",
        ">": "greater_than",
        "gt": "greater_than",
        "greater_than": "greater_than",
        "<": "less_than",
        "lt": "less_than",
        "less_than": "less_than",
        ">=": "greater_than_or_equal",
        "gte": "greater_than_or_equal",
        "greater_than_or_equal": "greater_than_or_equal",
        "<=": "less_than_or_equal",
        "lte": "less_than_or_equal",
        "less_than_or_equal": "less_than_or_equal",
        "contains": "contains",
    }

    return operator_map.get(operator, "equals")


# Apply one validated filter rule to the claims dataframe.
def apply_query_filter(
    dataframe: pd.DataFrame,
    filter_rule: dict
) -> pd.DataFrame:

    column = filter_rule["column"]
    operator = normalize_query_operator(
        filter_rule.get("operator", "equals")
    )
    value = filter_rule.get("value")
    value_column = filter_rule.get("value_column")

    if column not in dataframe.columns:
        return dataframe

    if value_column not in dataframe.columns:
        value_column = None

    series = dataframe[column]

    comparison_value = (
        dataframe[value_column]
        if value_column
        else value
    )

    if operator == "contains":
        return dataframe[
            series.astype("string").str.contains(
                str(comparison_value),
                case=False,
                na=False,
                regex=False,
            )
        ]

    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")

        if value_column:
            numeric_value = pd.to_numeric(
                comparison_value, errors="coerce"
            )
        else:
            numeric_value = pd.to_numeric(value, errors="coerce")

        if operator == "greater_than":
            return dataframe[numeric_series > numeric_value]

        if operator == "less_than":
            return dataframe[numeric_series < numeric_value]

        if operator == "greater_than_or_equal":
            return dataframe[numeric_series >= numeric_value]

        if operator == "less_than_or_equal":
            return dataframe[numeric_series <= numeric_value]

        if operator == "not_equals":
            return dataframe[numeric_series != numeric_value]

        return dataframe[numeric_series == numeric_value]

    normalized_series = series.astype("string").str.strip().str.lower()

    if value_column:
        normalized_value = (
            comparison_value.astype("string").str.strip().str.lower()
        )
    else:
        normalized_value = str(value).strip().lower()

    if operator == "not_equals":
        return dataframe[normalized_series != normalized_value]

    return dataframe[normalized_series == normalized_value]


# Execute dynamic query against the claims dataset.
def query_claim_dataset(settings: AppSettings, query_plan: dict) -> dict:
    """
    Execute a validated dynamic query plan against the claims CSV.

    The LLM creates a query plan and Python safely applies it.
    """

    dataframe = load_claim_dataset(settings=settings)

    if dataframe is None:
        return {"error": "claim_dataset.csv was not found"}

    available_columns = list(dataframe.columns)
    guarded_query_plan = query_plan.copy()

    selected_columns = (
        guarded_query_plan.get("select") or available_columns
    )

    selected_columns = [
        column
        for column in selected_columns
        if column in available_columns
    ]

    if not selected_columns:
        selected_columns = available_columns

    filtered_dataframe = dataframe.copy()

    for filter_rule in guarded_query_plan.get("filters", []):
        if (
            isinstance(filter_rule, dict)
            and filter_rule.get("column") in available_columns
        ):
            filtered_dataframe = apply_query_filter(
                dataframe=filtered_dataframe,
                filter_rule=filter_rule,
            )

    sort_by = guarded_query_plan.get("sort_by")

    if sort_by in available_columns:
        sort_order = str(
            guarded_query_plan.get("sort_order", "asc")
        ).lower()

        filtered_dataframe = filtered_dataframe.sort_values(
            by=sort_by,
            ascending=sort_order != "desc",
            na_position="last",
        )

    limit = guarded_query_plan.get("limit")

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        limit = DEFAULT_QUERY_LIMIT

    limit = min(limit, MAX_QUERY_LIMIT)

    guarded_query_plan["limit"] = limit

    filtered_dataframe = filtered_dataframe.head(limit)

    records = [
        normalize_claim_record(record)
        for record in filtered_dataframe[selected_columns].to_dict(
            orient="records"
        )
    ]

    return {
        "query_plan": guarded_query_plan,
        "row_count": len(records),
        "max_query_limit": MAX_QUERY_LIMIT,
        "records": records,
    }


# Ask a natural-language question against the claim documents.
def ask_claim_documents(settings: AppSettings, question: str) -> dict | None:
    """
    This tool delegates to the RAG question-answering pipeline.
    """

    return run_rag_question(
        settings=settings,
        question=question,
    )
