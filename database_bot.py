from langchain_community.llms import Ollama
import sqlite3

DB = "tesla_motors.db"
SCHEMA = """
Table: tesla_motors_data
Columns:
- id (integer)
- model (text)
- variant (text)
- color (text)
- fuel_type (text)
- transmission (text)
- price (real)
- manufacture_date (text)
- sale_date (text)
- state (text)
- mileage (real)
- engine_capacity (real)
- battery_capacity (real)
- charging_time (real)
- seating_capacity (integer)
- ground_clearance (real)
- boot_space (integer)
"""


SQL_INSTRUCTIONS = """
You are a SQLite SQL generator.

Rules:
- Output ONLY a valid SQLite SQL query
- Do NOT explain anything
- Do NOT use markdown
- Do NOT add comments
- Do NOT guess table or column names
- Use only the schema provided
- Never modify data (no INSERT, UPDATE, DELETE, DROP)

If the question cannot be answered using the schema,
output exactly: INVALID QUERY
"""

FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop",
    "alter", "create", "truncate"
]

def build_sql_prompt(question: str) -> str:
    return f"""
{SQL_INSTRUCTIONS}

Schema:
{SCHEMA}

Question:
{question}

SQL:
""".strip()

def build_explanation_prompt(question, columns, rows) -> str:
    return f"""
You are an AI data analyst.

The user asked:
"{question}"

SQL query results:
Columns: {columns}
Rows: {rows}

Explain the result clearly in plain English.
If there are no rows, say that the data is not available.
Do not invent numbers or assumptions.
""".strip()

def is_safe_sql(sql: str) -> bool:
    sql_lower = sql.lower()
    if not sql_lower.startswith("select"):
        return False
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in sql_lower:
            return False
    return True

def execute_sql(sql: str):
    try:
        conn = sqlite3.connect(DB)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return {"columns": columns, "rows": rows}

    except Exception as e:
        return {"error": str(e)}

def main():
    # LLM1 for SQL generation
    sql_llm = Ollama(model="mistral", temperature=0)

    # LLM2 for explanation
    explain_llm = Ollama(model="mistral", temperature=0.3)

    print("Tesla SQL Assistant based on Ollama (Mistral).")
    print("Ask questions about Tesla vehicles.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input(">> ").strip()

        if question.lower() == "quit":
            print("Goodbye")
            break

        # SQL generation
        sql_prompt = build_sql_prompt(question)
        sql = sql_llm.invoke(sql_prompt).strip()

        if sql == "INVALID QUERY":
            print("I cannot answer this question using the available data.\n")
            continue

        if not is_safe_sql(sql):
            print("Unsafe query detected and so execution blocked.\n")
            continue

        # SQL execution
        result = execute_sql(sql)

        if "error" in result:
            print("Database error:", result["error"], "\n")
            continue

        columns = result["columns"]
        rows = result["rows"]

        # LLM explanation
        explain_prompt = build_explanation_prompt(question, columns, rows)
        answer = explain_llm.invoke(explain_prompt)

        print("\nAnswer:")
        print(answer.strip())
        print("-" * 100)

if __name__ == "__main__":
    main()
