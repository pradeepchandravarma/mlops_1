import os
import json
import re
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ---- Configuration (from env only) ----
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TABLE_NAME = os.getenv("STUDENT_TABLE_NAME", "student_performance")

# "feature flag" for demo: if no key => endpoint should be disabled
LLM_ENABLED = bool(OPENAI_API_KEY) and (OpenAI is not None)

# ---- Safety: allow only SELECT ----
_BANNED = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|call|execute|do|vacuum)\b",
    re.IGNORECASE,
)


def _is_safe_select(sql: str) -> bool:
    if not sql:
        return False

    s = sql.strip()
    s = s.strip(";").strip()

    # must start with SELECT
    if not s.lower().startswith("select"):
        return False

    # no multi-statement
    if ";" in s:
        return False

    # ban write operations
    if _BANNED.search(s):
        return False

    return True


def _ensure_limit(sql: str, limit: int = 50) -> str:
    """
    If the LLM didn't add LIMIT, add it to protect the DB.
    Simple heuristic: if 'limit' appears anywhere, don't modify.
    """
    s = sql.strip().rstrip(";").strip()
    if re.search(r"\blimit\b", s, flags=re.IGNORECASE):
        return s
    return f"{s} LIMIT {limit}"


def _get_db_conn():
    if not all([DB_HOST, DB_USER, DB_PASSWORD]):
        raise RuntimeError("DB env vars missing: set DB_HOST, DB_USER, DB_PASSWORD on the API service.")
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=5,
    )


def _fetch_table_schema_text() -> str:
    with _get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name=%s
                ORDER BY ordinal_position;
                """,
                (TABLE_NAME,),
            )
            rows = cur.fetchall()

    if not rows:
        raise RuntimeError(f"Table '{TABLE_NAME}' not found in DB.")
    cols = "\n".join([f"- {name} ({dtype})" for name, dtype in rows])
    return f"Table: {TABLE_NAME}\nColumns:\n{cols}"


def _openai_client():
    if not LLM_ENABLED:
        if OpenAI is None:
            raise RuntimeError("LLM disabled: OpenAI SDK not installed in API image.")
        raise RuntimeError("LLM disabled: OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=OPENAI_API_KEY)


def _extract_sql(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```sql\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    m = re.search(r"(select\s.+)$", t, flags=re.IGNORECASE | re.DOTALL)
    return (m.group(1).strip() if m else t).strip()


def generate_sql(question: str, schema_text: str) -> str:
    client = _openai_client()

    system = f"""
You are a senior data analyst. Convert the user question to a SINGLE PostgreSQL SQL SELECT query.

Rules (strict):
- Output ONLY the SQL query. No markdown. No explanations.
- Must be a single SELECT statement (no semicolons, no multiple statements).
- Do NOT use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/COPY.
- Use only the given table and columns.
- Prefer simple, readable SQL.
- LIMIT results to 50 unless the question asks otherwise.

Database schema:
{schema_text}
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )

    raw = resp.choices[0].message.content or ""
    sql = _extract_sql(raw)

    if not _is_safe_select(sql):
        # treat as bad request
        raise ValueError(f"Unsafe or invalid SQL generated: {sql}")

    sql = _ensure_limit(sql, 50)
    return sql


def run_sql(sql: str) -> List[Dict[str, Any]]:
    if not _is_safe_select(sql):
        raise ValueError("Rejected unsafe SQL (must be a single SELECT).")

    with _get_db_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def summarize_answer(question: str, sql: str, rows: List[Dict[str, Any]]) -> str:
    # If LLM isn't enabled, return simple summary
    if not LLM_ENABLED:
        return f"Returned {len(rows)} row(s)."

    client = _openai_client()
    preview = json.dumps(rows[:20], ensure_ascii=False)

    system = """
You are a helpful assistant. Answer the user question using ONLY the query results provided.
Be concise. If results are empty, say you couldn't find matching records.
""".strip()

    user = f"""
Question: {question}

SQL:
{sql}

Rows (JSON, up to 20):
{preview}
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def handle_llm_query(question: str) -> Dict[str, Any]:
    schema = _fetch_table_schema_text()
    sql = generate_sql(question, schema)
    rows = run_sql(sql)
    answer = summarize_answer(question, sql, rows)
    return {"sql": sql, "rows": rows, "answer": answer}
