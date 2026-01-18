import os
import json
from typing import Optional, Literal, Dict, Any

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


REQUIRED_COLS = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "Performance Index",
]

DATA_PATH = os.getenv("STUDENT_CSV_PATH", "data/Student_Performance.csv")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------- Data ----------
def validate_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    validate_df(df)
    return df


def schema_text(df: pd.DataFrame) -> str:
    # Provide the LLM with schema + categorical values to reduce dumb mistakes
    extras = sorted(df["Extracurricular Activities"].dropna().astype(str).unique().tolist())
    return (
        "CSV schema:\n"
        f"- Columns: {', '.join(df.columns)}\n"
        f"- 'Extracurricular Activities' values: {extras}\n"
        "Rules:\n"
        "- If user asks for statistics, compute them using pandas.\n"
        "- If user asks for a plot or it would materially help, set needs_plot=true and provide a plot spec.\n"
        "- Never invent numbers. If unsure, request the specific column(s).\n"
    )


# ---------- Plotting ----------
PlotKind = Literal["hist", "scatter", "bar_mean", "box"]


def apply_filter(df: pd.DataFrame, filter_expr: Optional[str]) -> pd.DataFrame:
    """
    Very simple, safe filtering: only allow patterns like:
    - "Hours Studied > 5"
    - "Extracurricular Activities == 'Yes'"
    - "Sleep Hours >= 7"
    """
    if not filter_expr:
        return df

    # Block dangerous stuff
    forbidden = ["__", "import", "os.", "sys.", "exec", "eval", "open(", "read(", "write("]
    if any(tok in filter_expr for tok in forbidden):
        raise ValueError("Unsafe filter expression.")

    # Use pandas query with python engine (still keep it simple)
    return df.query(filter_expr, engine="python")


def make_plot(df: pd.DataFrame, kind: PlotKind, x: str, y: Optional[str] = None,
              groupby: Optional[str] = None, agg: str = "mean"):

    fig, ax = plt.subplots()

    if kind == "hist":
        ax.hist(df[x].dropna(), bins=30)
        ax.set_xlabel(x)
        ax.set_ylabel("Count")

    elif kind == "scatter":
        if not y:
            raise ValueError("scatter requires y")
        ax.scatter(df[x], df[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    elif kind == "bar_mean":
        if not y:
            raise ValueError("bar_mean requires y")
        group_col = groupby or x
        if agg != "mean":
            raise ValueError("Only mean is supported for bar_mean in this template.")
        means = df.groupby(group_col)[y].mean()
        ax.bar(means.index.astype(str), means.values)
        ax.set_xlabel(group_col)
        ax.set_ylabel(f"Mean {y}")

    elif kind == "box":
        if not y:
            # single column box
            ax.boxplot(df[x].dropna())
            ax.set_ylabel(x)
        else:
            # grouped boxplot
            df.boxplot(column=y, by=x, ax=ax)
            plt.suptitle("")
            ax.set_title(f"{y} by {x}")

    else:
        raise ValueError(f"Unknown plot kind: {kind}")

    fig.tight_layout()
    return fig


# ---------- LLM output schema ----------
class PlotSpec(BaseModel):
    kind: PlotKind = Field(..., description="One of: hist, scatter, bar_mean, box")
    x: str = Field(..., description="Column for x or grouping")
    y: Optional[str] = Field(None, description="Column for y/value if required")
    filter: Optional[str] = Field(None, description="Optional pandas query filter expression")
    groupby: Optional[str] = Field(None, description="Optional grouping column (defaults to x)")
    agg: Optional[str] = Field("mean", description="Aggregation for bar_mean (mean only)")


class LLMResponse(BaseModel):
    answer: str = Field(..., description="Direct answer for the user")
    needs_plot: bool = Field(..., description="Whether a plot should be shown")
    plot: Optional[PlotSpec] = Field(None, description="Plot spec if needs_plot is true")


def ask_llm(user_query: str, df: pd.DataFrame, api_key: str) -> LLMResponse:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

    # Give LLM a tiny preview + schema so it can map columns correctly
    preview = df.head(8).to_dict(orient="records")

    system = (
        "You are a data analysis assistant for a CSV dataset.\n"
        "You MUST produce a JSON response that matches this schema:\n"
        "{answer: string, needs_plot: boolean, plot: {kind,x,y,filter,groupby,agg} | null}\n"
        "Rules:\n"
        "- Do not hallucinate numeric results. Your answer should instruct what to compute.\n"
        "- If a plot would help materially OR user asks for visualization, set needs_plot=true.\n"
        "- Only use columns that exist in schema.\n"
        "- Plot kinds allowed: hist, scatter, bar_mean, box.\n"
        "- filter should be a simple pandas query like: \"`Hours Studied` > 5\" or \"`Extracurricular Activities` == 'Yes'\".\n"
    )

    prompt = f"""
SCHEMA:
{schema_text(df)}

DATA PREVIEW (first rows):
{json.dumps(preview, indent=2)}

USER QUESTION:
{user_query}

Return ONLY valid JSON, no markdown.
"""

    raw = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": prompt}]).content

    # Hard parse JSON + validate against Pydantic schema
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # If the model returns non-JSON, force a minimal fallback
        return LLMResponse(answer="I couldn't parse the model output. Please rephrase your question.", needs_plot=False)

    try:
        return LLMResponse(**data)
    except Exception as e:
        return LLMResponse(answer=f"Invalid LLM response schema: {e}", needs_plot=False)


def compute_answer_locally(df: pd.DataFrame, user_query: str) -> str:
    """
    Minimal deterministic calculator for common statistical questions.
    This is OPTIONAL; the LLM can answer conceptually, but numbers should come from here.
    You can extend this progressively.
    """
    q = user_query.lower()

    # Example: avg performance when hours > 5
    if "average" in q and "performance" in q and ("more than 5" in q or "> 5" in q):
        over5 = df[df["Hours Studied"] > 5]["Performance Index"].mean()
        le5 = df[df["Hours Studied"] <= 5]["Performance Index"].mean()
        return f"Avg Performance Index when Hours Studied > 5: {over5:.2f}. When <= 5: {le5:.2f}. Difference: {(over5-le5):.2f}."

    # Example: extracurricular difference
    if "difference" in q and ("extracurricular" in q or "extra curricular" in q):
        yes = df[df["Extracurricular Activities"] == "Yes"]["Performance Index"].mean()
        no = df[df["Extracurricular Activities"] == "No"]["Performance Index"].mean()
        return f"Avg Performance Index (Yes): {yes:.2f}. (No): {no:.2f}. Difference: {(yes-no):.2f}."

    return ""


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Student Performance LLM Analyst", layout="wide")
st.title("Student Performance Analyst")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Set it as an environment variable.")
    st.stop()

try:
    df = load_csv(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset from {DATA_PATH}: {e}")
    st.stop()

with st.expander("Dataset preview", expanded=False):
    st.write(f"Using dataset path: `{DATA_PATH}`")
    st.dataframe(df.head(20), use_container_width=True)

st.subheader("Dataset preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Ask anything")
user_query = st.text_input("Your question")

if st.button("Run"):

    # Optional: deterministic numeric answer for known patterns
    deterministic = compute_answer_locally(df, user_query)

    llm_resp = ask_llm(user_query, df, OPENAI_API_KEY)

    # If we have deterministic numbers, prepend them and force correctness
    if deterministic:
        st.success(deterministic)

    st.markdown("### LLM answer")
    st.write(llm_resp.answer)

    if llm_resp.needs_plot:
        if not llm_resp.plot:
            st.warning("LLM requested a plot but did not provide plot spec.")
        else:
            try:
                # Validate columns exist
                for col in [llm_resp.plot.x, llm_resp.plot.y, llm_resp.plot.groupby]:
                    if col and col not in df.columns and col != "HoursBucket":
                        raise ValueError(f"Unknown column in plot spec: {col}")

                df_plot = df.copy()

                # Special convenience bucket if LLM asks for it
                if llm_resp.plot.x == "HoursBucket" or llm_resp.plot.groupby == "HoursBucket":
                    df_plot["HoursBucket"] = df_plot["Hours Studied"].apply(lambda v: ">5" if v > 5 else "<=5")

                # Apply filter if any
                df_plot = apply_filter(df_plot, llm_resp.plot.filter)

                fig = make_plot(
                    df_plot,
                    kind=llm_resp.plot.kind,
                    x=llm_resp.plot.x,
                    y=llm_resp.plot.y,
                    groupby=llm_resp.plot.groupby,
                    agg=llm_resp.plot.agg or "mean",
                )
                st.markdown("### Plot")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Plot generation failed: {e}")