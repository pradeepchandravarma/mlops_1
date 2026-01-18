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
from typing import Optional, Literal, List, Dict, Any, Tuple

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# =========================
# Config
# =========================
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

# Plot kinds supported by our deterministic renderer
PlotKind = Literal["hist", "scatter", "bar_mean", "box", "cdf", "bar_count"]


# =========================
# Data Loading
# =========================
@st.cache_data(show_spinner=True)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def schema_text(df: pd.DataFrame) -> str:
    extras = sorted(df["Extracurricular Activities"].dropna().astype(str).unique().tolist())
    return (
        "Dataset is a Student Performance CSV.\n"
        f"Columns: {', '.join(df.columns)}.\n"
        f"Extracurricular Activities values: {extras}.\n"
        "Rules:\n"
        "- You are a dataset-bound analytical agent.\n"
        "- NEVER answer general knowledge questions.\n"
        "- NEVER invent numeric results.\n"
        "- Request tools to compute numbers.\n"
        "- If a plot is helpful or user asks for it, return a plot spec.\n"
    )


# =========================
# Scope guard (NOT ChatGPT)
# =========================
ANALYTIC_KEYWORDS = {
    "average", "avg", "mean", "median", "percentile", "quantile",
    "distribution", "hist", "histogram", "chart", "plot", "graph",
    "compare", "difference", "correlation", "relationship", "count",
    "how many", "number of", "percentage", "percent", "proportion",
    "cdf"
}

def is_out_of_scope(user_query: str, df: pd.DataFrame) -> bool:
    q = user_query.lower()

    # If query mentions any dataset column names (rough check), allow
    if any(col.lower() in q for col in df.columns):
        return False

    # If query contains analytic keywords but no column, still allow (LLM may map)
    if any(k in q for k in ANALYTIC_KEYWORDS):
        return False

    # Otherwise, out of scope
    return True


# =========================
# Filters (safe)
# =========================
def apply_structured_filter(df: pd.DataFrame, f: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    Structured filter format:
      {"col":"Sleep Hours","op":">","value":5}
      {"col":"Extracurricular Activities","op":"==","value":"Yes"}
    """
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


# =========================
# Tools (deterministic truth)
# =========================
# Each tool returns:
# - result payload
# - explanation: how it was computed (for query explanation)
def tool_explain(tool: str, args: Dict[str, Any]) -> str:
    if tool == "count":
        return f"Computed COUNT of rows after applying filter {args.get('filter')}."
    if tool == "count_pct":
        return "Computed COUNT and PERCENTAGE: count(filter)/total_rows*100."
    if tool == "group_mean":
        return "Computed group-wise MEAN using df.groupby(group_col)[value_col].mean()."
    if tool == "group_median":
        return "Computed group-wise MEDIAN using df.groupby(group_col)[value_col].median()."
    if tool == "percentile":
        return "Computed percentile using numpy.percentile(column_values, p)."
    if tool == "summary_stats":
        return "Computed summary stats (mean, median, std, min, max, percentiles) from the filtered dataset."
    if tool == "correlation":
        return "Computed Pearson correlation using df[[x,y]].corr().iloc[0,1]."
    if tool == "distribution_bins":
        return "Computed histogram bin counts using numpy.histogram."
    return "Computed result using deterministic pandas/numpy operations."

def run_tool(df: pd.DataFrame, req: Dict[str, Any]) -> Dict[str, Any]:
    tool = req["tool"]
    args = req.get("args", {})
    explanation = tool_explain(tool, args)

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
        return {
            "tool": "group_mean",
            "args": args,
            "result": {"means": means, "counts": counts},
            "explanation": explanation,
        }

    if tool == "group_median":
        group_col = args["group_col"]
        value_col = args.get("value_col", "Performance Index")
        med = df.groupby(group_col)[value_col].median().to_dict()
        counts = df.groupby(group_col)[value_col].count().to_dict()
        return {
            "tool": "group_median",
            "args": args,
            "result": {"medians": med, "counts": counts},
            "explanation": explanation,
        }

    if tool == "percentile":
        col = args["col"]
        p = float(args["p"])
        f = args.get("filter")
        df2 = apply_structured_filter(df, f)
        vals = df2[col].dropna().astype(float).values
        if len(vals) == 0:
            return {"tool": "percentile", "args": args, "result": {"p": p, "value": None}, "explanation": explanation}
        value = float(np.percentile(vals, p))
        return {"tool": "percentile", "args": args, "result": {"p": p, "value": value}, "explanation": explanation}

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
            return {"tool": "distribution_bins", "args": args, "result": {"bins": bins, "hist": [], "edges": []}, "explanation": explanation}
        hist, edges = np.histogram(vals, bins=bins)
        return {
            "tool": "distribution_bins",
            "args": args,
            "result": {"bins": bins, "hist": hist.tolist(), "edges": edges.tolist(), "n": int(len(vals))},
            "explanation": explanation,
        }

    raise ValueError(f"Unknown tool: {tool}")


# =========================
# Plot rendering (deterministic)
# =========================
def make_plot(df: pd.DataFrame, kind: PlotKind, x: str, y: Optional[str] = None,
              groupby: Optional[str] = None, agg: str = "mean",
              filter_struct: Optional[Dict[str, Any]] = None,
              bins: int = 20):
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


# =========================
# Intent forcing: ALWAYS compute for analytics
# =========================
def _find_col_mention(q: str) -> Optional[str]:
    # Map common phrases to columns
    mapping = {
        "hours studied": "Hours Studied",
        "study hours": "Hours Studied",
        "previous scores": "Previous Scores",
        "sleep hours": "Sleep Hours",
        "sleep": "Sleep Hours",
        "papers": "Sample Question Papers Practiced",
        "practice papers": "Sample Question Papers Practiced",
        "performance": "Performance Index",
        "performance index": "Performance Index",
        "extracurricular": "Extracurricular Activities",
        "extra curricular": "Extracurricular Activities",
    }
    for k, v in mapping.items():
        if k in q:
            return v
    return None


def force_tools_if_needed(user_query: str) -> List[Dict[str, Any]]:
    """
    Core reliability layer:
    - count/how many -> count_pct or count
    - median/percentile -> summary_stats / percentile
    - difference/compare -> group_mean / group_median
    - distribution -> distribution_bins
    - correlation -> correlation
    """
    q = user_query.lower()

    # Count questions (example: "How many students sleep more than 5 hours?")
    if any(k in q for k in ["how many", "number of", "count"]):
        # try to infer a numeric condition like "> 5"
        m = re.search(r"(sleep|hours studied|previous scores|performance index|performance|papers|practice)\s*(more than|over|>|>=|less than|under|<|<=)\s*(\d+(\.\d+)?)", q)
        col = _find_col_mention(q)
        if m and col:
            op_raw = m.group(2)
            val = float(m.group(3))
            op = ">"
            if op_raw in [">", "more than", "over"]:
                op = ">"
            elif op_raw in [">="]:
                op = ">="
            elif op_raw in ["<", "less than", "under"]:
                op = "<"
            elif op_raw in ["<="]:
                op = "<="

            return [{"tool": "count_pct", "args": {"filter": {"col": col, "op": op, "value": val}}}]

        # If no condition parsed, still allow count of all rows (rare)
        return [{"tool": "count", "args": {"filter": None}}]

    # Median questions
    if "median" in q:
        col = _find_col_mention(q) or "Performance Index"
        return [{"tool": "summary_stats", "args": {"col": col, "filter": None}}]

    # Percentile / quantile
    if "percentile" in q or "quantile" in q:
        col = _find_col_mention(q) or "Performance Index"
        # detect percentile value like "90th" or "p90" or "90 percentile"
        pm = re.search(r"(\d{1,2})(st|nd|rd|th)?\s*(percentile)", q)
        p = float(pm.group(1)) if pm else 90.0
        return [{"tool": "percentile", "args": {"col": col, "p": p, "filter": None}}]

    # Difference/compare extracurricular (use both mean + median for stronger insight)
    if ("difference" in q or "compare" in q) and ("extracurricular" in q or "extra curricular" in q):
        return [
            {"tool": "group_mean", "args": {"group_col": "Extracurricular Activities", "value_col": "Performance Index"}},
            {"tool": "group_median", "args": {"group_col": "Extracurricular Activities", "value_col": "Performance Index"}},
        ]

    # Hours > 5 vs <= 5 (mean + count% both sides)
    if ("difference" in q or "compare" in q or "average" in q or "mean" in q) and "hour" in q and "performance" in q:
        if "> 5" in q or "more than 5" in q or "over 5" in q:
            return [
                {"tool": "filter_summary", "args": {"filter": {"col": "Hours Studied", "op": ">", "value": 5}}},
                {"tool": "filter_summary", "args": {"filter": {"col": "Hours Studied", "op": "<=", "value": 5}}},
                {"tool": "count_pct", "args": {"filter": {"col": "Hours Studied", "op": ">", "value": 5}}},
            ]

    # Correlation / relationship
    if "correlation" in q or "relationship" in q:
        # try to identify two columns
        candidates = [
            "Hours Studied", "Previous Scores", "Sleep Hours",
            "Sample Question Papers Practiced", "Performance Index"
        ]
        mentioned = [c for c in candidates if c.lower() in q]
        # also map common phrases
        if not mentioned:
            for phrase, col in {
                "hours studied": "Hours Studied",
                "previous scores": "Previous Scores",
                "sleep hours": "Sleep Hours",
                "sample question papers": "Sample Question Papers Practiced",
                "performance index": "Performance Index",
                "performance": "Performance Index",
            }.items():
                if phrase in q and col not in mentioned:
                    mentioned.append(col)

        if len(mentioned) >= 2:
            return [{"tool": "correlation", "args": {"x": mentioned[0], "y": mentioned[1]}}]

    # Distribution questions
    if "distribution" in q or "histogram" in q:
        col = _find_col_mention(q) or "Performance Index"
        return [{"tool": "distribution_bins", "args": {"col": col, "bins": 20, "filter": None}}]

    # Default: no forced tools
    return []


# =========================
# Implicit plotting rules (no user must say "plot")
# =========================
def infer_plot_if_helpful(user_query: str, df: pd.DataFrame, tool_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    If user asks 'show me a chart/plot/graph' -> always plot something appropriate.
    If user asks distribution -> histogram or CDF.
    If user asks compare groups -> bar_mean or box.
    If user asks counts by group -> bar_count.
    Returns plot spec dict (structured filter supported).
    """
    q = user_query.lower()

    wants_plot = any(k in q for k in ["show me a chart", "show me a plot", "plot", "graph", "visualize", "visualise", "chart"])
    is_distribution = any(k in q for k in ["distribution", "hist", "histogram", "cdf"])
    is_compare_extra = ("extracurricular" in q or "extra curricular" in q) and ("compare" in q or "difference" in q)
    is_hours_threshold = ("hours" in q and ("more than 5" in q or "> 5" in q)) and "performance" in q

    # explicit request
    if wants_plot and is_compare_extra:
        return {"kind": "bar_mean", "x": "Extracurricular Activities", "y": "Performance Index", "groupby": "Extracurricular Activities", "filter_struct": None}

    if wants_plot and is_hours_threshold:
        # create bucket using filter_struct isn't enough; we just show histogram or CDF of performance
        return {"kind": "box", "x": "Hours Studied", "y": "Performance Index", "groupby": None, "filter_struct": None}

    if wants_plot and is_distribution:
        col = _find_col_mention(q) or "Performance Index"
        if "cdf" in q:
            return {"kind": "cdf", "x": col, "y": None, "groupby": None, "filter_struct": None}
        return {"kind": "hist", "x": col, "y": None, "groupby": None, "filter_struct": None}

    # implicit plot even if user didn't ask: distribution questions
    if is_distribution:
        col = _find_col_mention(q) or "Performance Index"
        return {"kind": "hist", "x": col, "y": None, "groupby": None, "filter_struct": None}

    # implicit plot for compare extracurricular
    if is_compare_extra:
        return {"kind": "bar_mean", "x": "Extracurricular Activities", "y": "Performance Index", "groupby": "Extracurricular Activities", "filter_struct": None}

    return None


# =========================
# LLM plan + final answer
# =========================
class PlotSpecModel(BaseModel):
    kind: Literal["hist", "scatter", "bar_mean", "box", "cdf", "bar_count"] = Field(...)
    x: str
    y: Optional[str] = None
    filter_struct: Optional[Dict[str, Any]] = None
    groupby: Optional[str] = None
    agg: Optional[str] = "mean"


class PlanModel(BaseModel):
    answer: str
    needs_plot: bool
    plot: Optional[PlotSpecModel] = None
    tool_requests: Optional[List[Dict[str, Any]]] = None
    explain: bool = False  # if user asked "how was this computed?"


def ask_llm_for_plan(user_query: str, df: pd.DataFrame) -> PlanModel:
    """
    LLM decides wording + optional plot + optional extra tools.
    Reliability is enforced by forced tools + deterministic computations.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
    preview = df.head(8).to_dict(orient="records")

    system = (
        "You are a dataset-bound analytical agent for a Student Performance CSV.\n"
        "You MUST return ONLY valid JSON.\n"
        "Schema:\n"
        "{answer: string, needs_plot: boolean, plot: {kind,x,y,filter_struct,groupby,agg} | null, tool_requests: [{tool,args}] | null, explain: boolean}\n"
        "Rules:\n"
        "- Never answer general knowledge questions.\n"
        "- Never invent numeric results.\n"
        "- If you need numbers, request tool_requests.\n"
        "- If a plot helps or user asks, set needs_plot=true and provide plot spec.\n"
        "- filter_struct must be structured: {col, op, value}.\n"
        f"{schema_text(df)}"
    )

    prompt = f"""
DATA PREVIEW:
{json.dumps(preview, indent=2)}

USER QUESTION:
{user_query}

Return ONLY JSON, no markdown.
"""

    raw = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": prompt}]).content
    try:
        data = json.loads(raw)
        return PlanModel(**data)
    except Exception:
        return PlanModel(
            answer="I couldn't parse a valid plan. Please ask a dataset-related question using the dataset columns.",
            needs_plot=False,
            plot=None,
            tool_requests=None,
            explain=False,
        )


def finalize_answer(user_query: str, plan: PlanModel, tool_results: List[Dict[str, Any]]) -> str:
    """
    Second LLM pass grounded on tool results.
    Adds "how computed" explanation if requested.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
    system = (
        "You are a dataset analyst. Use TOOL RESULTS as the ONLY source of numeric truth. "
        "Explain clearly, mention limitations (correlation != causation), and give practical, dataset-based insights. "
        "If the user asked how it was computed, include a short explanation using the provided per-tool explanations."
    )

    prompt = f"""
USER QUESTION:
{user_query}

PLAN DRAFT (wording):
{plan.answer}

TOOL RESULTS (trusted):
{json.dumps(tool_results, indent=2)}

If explain=true, include a brief 'How this was computed' section based on tool explanations.
Write the final answer now.
"""
    return llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": prompt}]).content


def user_asked_how_computed(user_query: str) -> bool:
    q = user_query.lower()
    return any(k in q for k in ["how was this computed", "how did you compute", "how did you calculate", "calculation", "show the steps"])


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Student App (Predictor + Copilot)", layout="wide")
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
        st.error("OPENAI_API_KEY is not set. Set it as an environment variable and restart Streamlit.")
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

    user_query = st.chat_input("Ask dataset questions (counts, medians, percentiles, distributions, comparisons, charts...)")

    if user_query:
        st.session_state["copilot_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            # Scope guard
            if is_out_of_scope(user_query, df):
                final = (
                    "❌ Out of scope.\n\n"
                    "This copilot only answers questions about the Student Performance dataset.\n"
                    "Ask about: Hours Studied, Sleep Hours, Previous Scores, Practice Papers, Extracurricular Activities, Performance Index."
                )
                st.markdown(final)
                st.session_state["copilot_messages"].append({"role": "assistant", "content": final})
            else:
                with st.spinner("Analyzing..."):
                    # 1) LLM plan (wording + possible plot spec)
                    plan = ask_llm_for_plan(user_query, df)
                    plan.explain = plan.explain or user_asked_how_computed(user_query)

                    # 2) Forced tools for reliability
                    forced = force_tools_if_needed(user_query)

                    # 3) Run tools (forced first, else LLM requested)
                    tool_results: List[Dict[str, Any]] = []
                    requests_to_run = forced if forced else (plan.tool_requests or [])

                    # Fallback: if user asks analytic keyword but no tools picked, compute summary stats on Performance Index
                    if not requests_to_run and any(k in user_query.lower() for k in ["average", "mean", "median", "percentile", "distribution", "compare", "difference", "correlation", "count", "how many", "percentage"]):
                        requests_to_run = [{"tool": "summary_stats", "args": {"col": "Performance Index", "filter": None}}]

                    for req in requests_to_run:
                        try:
                            tool_results.append(run_tool(df, req))
                        except Exception as e:
                            tool_results.append({"tool": req.get("tool"), "args": req.get("args"), "error": str(e), "explanation": "Tool execution failed."})

                    # 4) Implicit plot if helpful or user asked
                    implicit_plot = infer_plot_if_helpful(user_query, df, tool_results)

                    # Merge implicit plot into plan if plan didn't request a plot
                    if implicit_plot and not plan.needs_plot:
                        plan.needs_plot = True
                        plan.plot = PlotSpecModel(
                            kind=implicit_plot["kind"],
                            x=implicit_plot["x"],
                            y=implicit_plot.get("y"),
                            filter_struct=implicit_plot.get("filter_struct"),
                            groupby=implicit_plot.get("groupby"),
                            agg="mean",
                        )

                    # 5) Final answer grounded on tool outputs
                    final = finalize_answer(user_query, plan, tool_results)
                    st.markdown(final)

                    # 6) Show "How computed" explicitly if requested (in addition to LLM section)
                    if plan.explain:
                        with st.expander("How this was computed (deterministic)", expanded=False):
                            for tr in tool_results:
                                st.write(f"- **{tr.get('tool')}**: {tr.get('explanation')}")

                    # 7) Render plot if requested
                    if plan.needs_plot and plan.plot:
                        try:
                            # Validate columns
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