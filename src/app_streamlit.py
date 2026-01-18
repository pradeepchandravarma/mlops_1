""""
import os
import streamlit as st
import requests

# Base API URL (no /predict hardcoded)
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Final predict endpoint
PREDICT_URL = f"{API_BASE_URL.rstrip('/')}/predict"

st.write("Calling API at:", PREDICT_URL)

st.title("Suganthy V1 - Student Performance Predictor (UI)")

hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=6.0)
prev = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=75.0)
extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
papers = st.number_input("Sample Question Papers Practiced", min_value=0.0, max_value=50.0, value=4.0)

if st.button("Predict"):
    payload = {
        "features": {
            "Hours Studied": hours,
            "Previous Scores": prev,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleep,
            "Sample Question Papers Practiced": papers,
        }
    }

    try:
        r = requests.post(PREDICT_URL, json=payload, timeout=10)
        r.raise_for_status()
        pred = r.json()["prediction"]
        st.success(f"Predicted Performance Index: {pred:.2f}")
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
"""
import os
import json
import re
from typing import Optional, Literal, List, Dict, Any

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# ============================================================
# CONFIG
# ============================================================
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_BASE_URL.rstrip('/')}/predict"

DATA_PATH = os.getenv("STUDENT_CSV_PATH", "data/Student_Performance.csv")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

REQUIRED_COLS = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "Performance Index",
]

PlotKind = Literal["hist", "scatter", "bar_mean", "box", "cdf", "bar_count"]


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=True)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def dataset_overview_text(df: pd.DataFrame) -> str:
    n = len(df)
    cols = list(df.columns)
    extras = sorted(df["Extracurricular Activities"].dropna().astype(str).unique().tolist())
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return (
        f"Dataset: Student Performance\n"
        f"- Rows: {n}\n"
        f"- Columns: {', '.join(cols)}\n"
        f"- Numeric columns: {', '.join(numeric_cols)}\n"
        f"- Extracurricular Activities values: {extras}\n"
        f"Target/outcome: Performance Index (higher is better).\n"
        f"Note: This dataset supports correlation/association analysis, not causal proof."
    )


# ============================================================
# SCOPE GUARD (dataset-only, NOT ChatGPT)
# ============================================================
ANALYTIC_KEYWORDS = {
    "average", "avg", "mean", "median", "percentile", "quantile",
    "distribution", "hist", "histogram", "chart", "plot", "graph",
    "compare", "difference", "correlation", "relationship", "count",
    "how many", "number of", "percentage", "percent", "proportion",
    "cdf", "describe", "overview", "columns", "dataset"
}

OVERVIEW_PHRASES = [
    "explain the dataset", "about the dataset", "dataset overview",
    "what is this dataset", "describe the dataset", "what are the columns",
    "what does this dataset contain", "data description", "summary of the dataset",
    "explain me about", "explain about the dataset", "tell me about the dataset",
    "describe columns"
]


def is_out_of_scope(user_query: str, df: pd.DataFrame) -> bool:
    q = user_query.lower().strip()

    # Allow dataset overview requests
    if any(p in q for p in OVERVIEW_PHRASES):
        return False

    # Allow queries mentioning columns
    if any(col.lower() in q for col in df.columns):
        return False

    # Allow analytic keywords (LLM can map to columns)
    if any(k in q for k in ANALYTIC_KEYWORDS):
        return False

    # Otherwise out-of-scope: don't answer general knowledge
    return True


# ============================================================
# SAFE FILTERING
# ============================================================
def apply_structured_filter(df: pd.DataFrame, f: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not f:
        return df

    col = f.get("col")
    op = f.get("op")
    value = f.get("value")

    if col not in df.columns:
        raise ValueError(f"Unknown filter column: {col}")

    if op == ">":
        return df[df[col] > value]
    if op == ">=":
        return df[df[col] >= value]
    if op == "<":
        return df[df[col] < value]
    if op == "<=":
        return df[df[col] <= value]
    if op == "==":
        return df[df[col] == value]
    if op == "!=":
        return df[df[col] != value]

    raise ValueError(f"Unsupported operator: {op}")


def _find_col_mention(q: str) -> Optional[str]:
    mapping = {
        "hours studied": "Hours Studied",
        "study hours": "Hours Studied",
        "previous scores": "Previous Scores",
        "sleep hours": "Sleep Hours",
        "sleep": "Sleep Hours",
        "papers": "Sample Question Papers Practiced",
        "practice papers": "Sample Question Papers Practiced",
        "sample question papers": "Sample Question Papers Practiced",
        "performance index": "Performance Index",
        "performance": "Performance Index",
        "extracurricular": "Extracurricular Activities",
        "extra curricular": "Extracurricular Activities",
    }
    for k, v in mapping.items():
        if k in q:
            return v
    return None


def user_asked_how_computed(user_query: str) -> bool:
    q = user_query.lower()
    return any(k in q for k in ["how was this computed", "how did you compute", "how did you calculate", "calculation", "show the steps"])


def user_asked_for_plot(user_query: str) -> bool:
    q = user_query.lower()
    return any(k in q for k in ["plot", "graph", "chart", "visualize", "visualise", "show me"])


# ============================================================
# TOOLS (deterministic truth)
# ============================================================
def tool_explain(tool: str, args: Dict[str, Any]) -> str:
    if tool == "count":
        return f"Counted rows after applying filter {args.get('filter')}."
    if tool == "count_pct":
        return "Computed count and percentage: count(filter)/total_rows*100."
    if tool == "group_mean":
        return "Computed group-wise mean: df.groupby(group_col)[value_col].mean()."
    if tool == "group_median":
        return "Computed group-wise median: df.groupby(group_col)[value_col].median()."
    if tool == "percentile":
        return "Computed percentile using numpy.percentile(values, p)."
    if tool == "summary_stats":
        return "Computed summary stats (mean/median/std/min/max/p10/p25/p75/p90) from filtered values."
    if tool == "mean_filter":
        return "Computed mean of value_col over rows matching filter: df_filtered[value_col].mean()."
    if tool == "correlation":
        return "Computed Pearson correlation: df[[x,y]].corr().iloc[0,1]."
    if tool == "distribution_bins":
        return "Computed histogram bins using numpy.histogram."
    if tool == "overview":
        return "Generated dataset overview from df shape, columns, dtypes, and basic unique values."
    return "Computed result using deterministic pandas/numpy operations."


def run_tool(df: pd.DataFrame, req: Dict[str, Any]) -> Dict[str, Any]:
    tool = req["tool"]
    args = req.get("args", {})
    explanation = tool_explain(tool, args)

    if tool == "overview":
        return {"tool": "overview", "args": args, "result": {"text": dataset_overview_text(df)}, "explanation": explanation}

    if tool == "count":
        f = args.get("filter")
        df2 = apply_structured_filter(df, f)
        return {"tool": "count", "args": args, "result": {"count": int(len(df2))}, "explanation": explanation}

    if tool == "count_pct":
        f = args.get("filter")
        df2 = apply_structured_filter(df, f)
        total = len(df)
        count = len(df2)
        pct = (count / total * 100.0) if total else 0.0
        return {
            "tool": "count_pct",
            "args": args,
            "result": {"count": int(count), "total": int(total), "pct": float(pct)},
            "explanation": explanation,
        }

    if tool == "group_mean":
        group_col = args["group_col"]
        value_col = args.get("value_col", "Performance Index")
        means = df.groupby(group_col)[value_col].mean().to_dict()
        counts = df.groupby(group_col)[value_col].count().to_dict()
        return {"tool": "group_mean", "args": args, "result": {"means": means, "counts": counts}, "explanation": explanation}

    if tool == "group_median":
        group_col = args["group_col"]
        value_col = args.get("value_col", "Performance Index")
        med = df.groupby(group_col)[value_col].median().to_dict()
        counts = df.groupby(group_col)[value_col].count().to_dict()
        return {"tool": "group_median", "args": args, "result": {"medians": med, "counts": counts}, "explanation": explanation}

    if tool == "percentile":
        col = args["col"]
        p = float(args["p"])
        f = args.get("filter")
        df2 = apply_structured_filter(df, f)
        vals = df2[col].dropna().astype(float).values
        if len(vals) == 0:
            return {"tool": "percentile", "args": args, "result": {"p": p, "value": None, "n": 0}, "explanation": explanation}
        value = float(np.percentile(vals, p))
        return {"tool": "percentile", "args": args, "result": {"p": p, "value": value, "n": int(len(vals))}, "explanation": explanation}

    if tool == "summary_stats":
        col = args["col"]
        f = args.get("filter")
        df2 = apply_structured_filter(df, f)
        vals = df2[col].dropna().astype(float).values
        if len(vals) == 0:
            return {"tool": "summary_stats", "args": args, "result": {"n": 0}, "explanation": explanation}
        result = {
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "p90": float(np.percentile(vals, 90)),
        }
        return {"tool": "summary_stats", "args": args, "result": result, "explanation": explanation}

    if tool == "mean_filter":
        value_col = args.get("value_col", "Performance Index")
        f = args.get("filter")
        df2 = apply_structured_filter(df, f)
        vals = df2[value_col].dropna().astype(float).values
        if len(vals) == 0:
            return {"tool": "mean_filter", "args": args, "result": {"n": 0, "mean": None}, "explanation": explanation}
        return {"tool": "mean_filter", "args": args, "result": {"n": int(len(vals)), "mean": float(np.mean(vals))}, "explanation": explanation}

    if tool == "correlation":
        x = args["x"]
        y = args["y"]
        corr = float(df[[x, y]].corr().iloc[0, 1])
        return {"tool": "correlation", "args": args, "result": {"corr": corr}, "explanation": explanation}

    if tool == "distribution_bins":
        col = args["col"]
        bins = int(args.get("bins", 20))
        f = args.get("filter")
        df2 = apply_structured_filter(df, f)
        vals = df2[col].dropna().astype(float).values
        if len(vals) == 0:
            return {"tool": "distribution_bins", "args": args, "result": {"bins": bins, "hist": [], "edges": [], "n": 0}, "explanation": explanation}
        hist, edges = np.histogram(vals, bins=bins)
        return {
            "tool": "distribution_bins",
            "args": args,
            "result": {"bins": bins, "hist": hist.tolist(), "edges": edges.tolist(), "n": int(len(vals))},
            "explanation": explanation,
        }

    raise ValueError(f"Unknown tool: {tool}")


# ============================================================
# PLOTTING (deterministic)
# ============================================================
def make_plot(
    df: pd.DataFrame,
    kind: PlotKind,
    x: str,
    y: Optional[str] = None,
    groupby: Optional[str] = None,
    agg: str = "mean",
    filter_struct: Optional[Dict[str, Any]] = None,
    bins: int = 20,
):
    dfp = apply_structured_filter(df, filter_struct)
    fig, ax = plt.subplots()

    if kind == "hist":
        ax.hist(dfp[x].dropna(), bins=bins)
        ax.set_xlabel(x)
        ax.set_ylabel("Count")

    elif kind == "cdf":
        vals = np.sort(dfp[x].dropna().astype(float).values)
        if len(vals) == 0:
            raise ValueError("No data to plot CDF.")
        yvals = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, yvals)
        ax.set_xlabel(x)
        ax.set_ylabel("CDF")

    elif kind == "scatter":
        if not y:
            raise ValueError("scatter requires y")
        ax.scatter(dfp[x], dfp[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    elif kind == "bar_mean":
        if not y:
            raise ValueError("bar_mean requires y")
        group_col = groupby or x
        if agg != "mean":
            raise ValueError("Only mean is supported for bar_mean.")
        means = dfp.groupby(group_col)[y].mean()
        ax.bar(means.index.astype(str), means.values)
        ax.set_xlabel(group_col)
        ax.set_ylabel(f"Mean {y}")

    elif kind == "bar_count":
        group_col = groupby or x
        counts = dfp.groupby(group_col).size()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel(group_col)
        ax.set_ylabel("Count")

    elif kind == "box":
        if not y:
            ax.boxplot(dfp[x].dropna())
            ax.set_ylabel(x)
        else:
            dfp.boxplot(column=y, by=x, ax=ax)
            plt.suptitle("")
            ax.set_title(f"{y} by {x}")

    else:
        raise ValueError(f"Unknown plot kind: {kind}")

    fig.tight_layout()
    return fig


# ============================================================
# PLOTTING SPEC (deterministic)
# ============================================================
def infer_plot_spec(user_query: str) -> Optional[Dict[str, Any]]:
    q = user_query.lower()
    if not user_asked_for_plot(user_query):
        return None

    if "extracurricular" in q or "extra curricular" in q:
        if "hours" in q or "studied" in q:
            return {"kind": "box", "x": "Extracurricular Activities", "y": "Hours Studied", "groupby": None, "filter_struct": None}
        return {"kind": "bar_count", "x": "Extracurricular Activities", "y": None, "groupby": "Extracurricular Activities", "filter_struct": None}

    if ("hours" in q or "studied" in q) and "performance" in q:
        return {"kind": "scatter", "x": "Hours Studied", "y": "Performance Index", "groupby": None, "filter_struct": None}

    if any(k in q for k in ["distribution", "hist", "histogram"]):
        col = _find_col_mention(q) or "Performance Index"
        return {"kind": "hist", "x": col, "y": None, "groupby": None, "filter_struct": None}

    if "cdf" in q:
        col = _find_col_mention(q) or "Performance Index"
        return {"kind": "cdf", "x": col, "y": None, "groupby": None, "filter_struct": None}

    col = _find_col_mention(q) or "Performance Index"
    if col == "Extracurricular Activities":
        return {"kind": "bar_count", "x": "Extracurricular Activities", "y": None, "groupby": "Extracurricular Activities", "filter_struct": None}
    return {"kind": "hist", "x": col, "y": None, "groupby": None, "filter_struct": None}


# ============================================================
# TOOL FORCING (PATCHED: EXACT MATCH + MEAN FILTER)
# ============================================================
def force_tools_if_needed(user_query: str) -> List[Dict[str, Any]]:
    q = user_query.lower()

    # Dataset overview
    if any(p in q for p in OVERVIEW_PHRASES) or ("dataset" in q and ("explain" in q or "describe" in q or "columns" in q)):
        return [{"tool": "overview", "args": {}}]

    # Count / percentage questions (PATCHED: handle ONLY/EXACTLY/==)
    if any(k in q for k in ["how many", "number of", "count", "percentage", "percent", "proportion"]):
        col = _find_col_mention(q) or "Sleep Hours"

        # exact match: "only 5", "exactly 5", "equal to 5", "= 5"
        exact = re.search(r"(only|exactly|equal to|equals|=)\s*(\d+(\.\d+)?)", q)
        if exact:
            val = float(exact.group(2))
            return [{"tool": "count_pct", "args": {"filter": {"col": col, "op": "==", "value": val}}}]

        # inequality: more than/over/>/>=/less than/under/</<=
        ineq = re.search(r"(more than|over|>|>=|less than|under|<|<=)\s*(\d+(\.\d+)?)", q)
        if ineq:
            op_raw = ineq.group(1)
            val = float(ineq.group(2))

            if op_raw in [">", "more than", "over"]:
                op = ">"
            elif op_raw == ">=":
                op = ">="
            elif op_raw in ["<", "less than", "under"]:
                op = "<"
            else:
                op = "<="

            return [{"tool": "count_pct", "args": {"filter": {"col": col, "op": op, "value": val}}}]

        return [{"tool": "count_pct", "args": {"filter": None}}]

    # Average/mean performance when Hours Studied condition (PATCHED)
    if ("average" in q or "avg" in q or "mean" in q) and "performance" in q and ("hour" in q or "hours studied" in q):
        # exact: "only/exactly/equal to/= 5"
        exact = re.search(r"(only|exactly|equal to|equals|=)\s*(\d+(\.\d+)?)", q)
        if exact:
            val = float(exact.group(2))
            return [{"tool": "mean_filter", "args": {"value_col": "Performance Index", "filter": {"col": "Hours Studied", "op": "==", "value": val}}}]

        # inequality: more than/over/>/>=/less than/under/</<=
        ineq = re.search(r"(more than|over|>|>=|less than|under|<|<=)\s*(\d+(\.\d+)?)", q)
        if ineq:
            op_raw = ineq.group(1)
            val = float(ineq.group(2))
            if op_raw in [">", "more than", "over"]:
                op = ">"
            elif op_raw == ">=":
                op = ">="
            elif op_raw in ["<", "less than", "under"]:
                op = "<"
            else:
                op = "<="
            return [{"tool": "mean_filter", "args": {"value_col": "Performance Index", "filter": {"col": "Hours Studied", "op": op, "value": val}}}]

        # No threshold provided: global mean
        return [{"tool": "mean_filter", "args": {"value_col": "Performance Index", "filter": None}}]

    # Median
    if "median" in q:
        col = _find_col_mention(q) or "Performance Index"
        return [{"tool": "summary_stats", "args": {"col": col, "filter": None}}]

    # Percentile
    if "percentile" in q or "quantile" in q:
        col = _find_col_mention(q) or "Performance Index"
        pm = re.search(r"(\d{1,2})(st|nd|rd|th)?\s*(percentile)", q)
        p = float(pm.group(1)) if pm else 90.0
        return [{"tool": "percentile", "args": {"col": col, "p": p, "filter": None}}]

    # Compare extracurricular
    if ("difference" in q or "compare" in q) and ("extracurricular" in q or "extra curricular" in q):
        return [
            {"tool": "group_mean", "args": {"group_col": "Extracurricular Activities", "value_col": "Performance Index"}},
            {"tool": "group_median", "args": {"group_col": "Extracurricular Activities", "value_col": "Performance Index"}},
        ]

    # Distribution
    if "distribution" in q or "histogram" in q:
        col = _find_col_mention(q) or "Performance Index"
        return [{"tool": "distribution_bins", "args": {"col": col, "bins": 20, "filter": None}}]

    # Correlation/relationship
    if "correlation" in q or "relationship" in q:
        candidates = ["Hours Studied", "Previous Scores", "Sleep Hours", "Sample Question Papers Practiced", "Performance Index"]
        mentioned = [c for c in candidates if c.lower() in q]
        if len(mentioned) >= 2:
            return [{"tool": "correlation", "args": {"x": mentioned[0], "y": mentioned[1]}}]

    # Plot request fallback -> compute stats for the most likely column
    if user_asked_for_plot(user_query):
        col = _find_col_mention(q) or "Performance Index"
        return [{"tool": "summary_stats", "args": {"col": col, "filter": None}}]

    # Default
    return [{"tool": "summary_stats", "args": {"col": "Performance Index", "filter": None}}]


# ============================================================
# LLM: response framing + grounded narrative
# ============================================================
class PlotSpecModel(BaseModel):
    kind: PlotKind = Field(...)
    x: str
    y: Optional[str] = None
    filter_struct: Optional[Dict[str, Any]] = None
    groupby: Optional[str] = None
    agg: Optional[str] = "mean"


class PlanModel(BaseModel):
    answer: str
    needs_plot: bool
    plot: Optional[PlotSpecModel] = None
    explain: bool = False


def ask_llm_for_answer(user_query: str, df: pd.DataFrame) -> PlanModel:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
    preview = df.head(6).to_dict(orient="records")
    schema = {
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "extras": sorted(df["Extracurricular Activities"].dropna().astype(str).unique().tolist()),
    }

    system = (
        "You are a dataset-bound analytical agent.\n"
        "Do NOT answer general knowledge questions.\n"
        "Do NOT invent numeric results.\n"
        "Return ONLY JSON:\n"
        "{answer: string, needs_plot: boolean, plot: {kind,x,y,filter_struct,groupby,agg} | null, explain: boolean}\n"
    )

    prompt = f"""
SCHEMA:
{json.dumps(schema, indent=2)}

DATA PREVIEW:
{json.dumps(preview, indent=2)}

USER QUESTION:
{user_query}

If the user asks for a chart/plot/graph, set needs_plot=true.
If user asks "how computed", set explain=true.
Return ONLY JSON.
"""
    raw = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": prompt}]).content
    try:
        data = json.loads(raw)
        return PlanModel(**data)
    except Exception:
        return PlanModel(
            answer="I can answer dataset questions (counts, means, medians, percentiles, distributions, comparisons, charts). Please rephrase using dataset columns.",
            needs_plot=user_asked_for_plot(user_query),
            plot=None,
            explain=user_asked_how_computed(user_query),
        )


def finalize_answer(user_query: str, plan: PlanModel, tool_results: List[Dict[str, Any]]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)

    system = (
        "You are a dataset analyst.\n"
        "Use TOOL_RESULTS as the ONLY source of numeric truth.\n"
        "If user asked for plot, describe what the plot shows.\n"
        "Mention correlation != causation where relevant.\n"
        "Be concise and factual.\n"
    )

    prompt = f"""
USER QUESTION:
{user_query}

PLAN (draft wording):
{plan.answer}

TOOL_RESULTS (trusted):
{json.dumps(tool_results, indent=2)}

Write the final answer. Do not invent numbers.
If explain=true, add a short 'How computed' section using tool explanations.
"""
    return llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": prompt}]).content


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Student Performance App (Predictor + Copilot)", layout="wide")
st.title("Student Performance App (Predictor + Dataset Copilot)")

tabs = st.tabs(["🎯 Predictor", "🤖 Copilot (Dataset-only)"])


# ---------- TAB 1: Predictor ----------
with tabs[0]:
    st.subheader("Student Performance Predictor (UI)")
    st.caption(f"Calling API at: {PREDICT_URL}")

    hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=6.0)
    prev = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=75.0)
    extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    papers = st.number_input("Sample Question Papers Practiced", min_value=0.0, max_value=50.0, value=4.0)

    if st.button("Predict"):
        payload = {
            "features": {
                "Hours Studied": hours,
                "Previous Scores": prev,
                "Extracurricular Activities": extra,
                "Sleep Hours": sleep,
                "Sample Question Papers Practiced": papers,
            }
        }

        try:
            r = requests.post(PREDICT_URL, json=payload, timeout=10)
            r.raise_for_status()
            pred = r.json()["prediction"]
            st.success(f"Predicted Performance Index: {pred:.2f}")
        except requests.exceptions.RequestException as e:
            st.error(f"API call failed: {e}")


# ---------- TAB 2: Copilot ----------
with tabs[1]:
    st.subheader("Student Performance Copilot (Dataset-bound Analytical Agent)")

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set. Set it as an environment variable in ECS task definition and redeploy.")
        st.stop()

    try:
        df = load_df(DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load dataset from {DATA_PATH}: {e}")
        st.stop()

    with st.expander("Dataset preview", expanded=False):
        st.write(f"Using dataset path: `{DATA_PATH}`")
        st.dataframe(df.head(20), use_container_width=True)

    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = []

    for msg in st.session_state["copilot_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask dataset questions (counts, means, medians, percentiles, distributions, comparisons, charts...)")

    if user_query:
        st.session_state["copilot_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if is_out_of_scope(user_query, df):
                final = (
                    "❌ Out of scope.\n\n"
                    "This copilot only answers questions about the Student Performance dataset.\n"
                    "Ask about: Hours Studied, Sleep Hours, Previous Scores, Sample Question Papers Practiced, "
                    "Extracurricular Activities, Performance Index."
                )
                st.markdown(final)
                st.session_state["copilot_messages"].append({"role": "assistant", "content": final})
            else:
                with st.spinner("Analyzing..."):
                    plan = ask_llm_for_answer(user_query, df)
                    plan.explain = plan.explain or user_asked_how_computed(user_query)

                    # Always run deterministic tools
                    forced_tools = force_tools_if_needed(user_query)
                    tool_results: List[Dict[str, Any]] = []
                    for req in forced_tools:
                        try:
                            tool_results.append(run_tool(df, req))
                        except Exception as e:
                            tool_results.append({"tool": req.get("tool"), "args": req.get("args"), "error": str(e), "explanation": "Tool execution failed."})

                    # Plot spec
                    plot_spec = infer_plot_spec(user_query)
                    if plot_spec:
                        plan.needs_plot = True
                        plan.plot = PlotSpecModel(**plot_spec)

                    # Grounded final answer
                    final = finalize_answer(user_query, plan, tool_results)
                    st.markdown(final)

                    if plan.explain and tool_results:
                        with st.expander("How this was computed (deterministic)", expanded=False):
                            for tr in tool_results:
                                st.write(f"- **{tr.get('tool')}**: {tr.get('explanation')}")

                    if plan.needs_plot and plan.plot:
                        try:
                            if plan.plot.x not in df.columns:
                                raise ValueError(f"Unknown x column: {plan.plot.x}")
                            if plan.plot.y and plan.plot.y not in df.columns:
                                raise ValueError(f"Unknown y column: {plan.plot.y}")
                            if plan.plot.groupby and plan.plot.groupby not in df.columns:
                                raise ValueError(f"Unknown groupby column: {plan.plot.groupby}")

                            fig = make_plot(
                                df,
                                kind=plan.plot.kind,
                                x=plan.plot.x,
                                y=plan.plot.y,
                                groupby=plan.plot.groupby,
                                agg=plan.plot.agg or "mean",
                                filter_struct=plan.plot.filter_struct,
                                bins=20,
                            )
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not render plot: {e}")

                st.session_state["copilot_messages"].append({"role": "assistant", "content": final})