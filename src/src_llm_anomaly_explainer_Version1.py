"""
LLM-Powered Anomaly Explanation Agent

Demonstrates how to combine a trained anomaly detection pipeline with an LLM
(via OpenAI-compatible API) to automatically generate human-readable explanations
of detected anomalies.

This bridges the gap between data pipelines and AI agent development.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python src/llm_anomaly_explainer.py
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# 1. Reuse existing preprocessing pipeline
# -------------------------------------------------------------------
from preprocessing import load_data, normalize_data, segment_windows


def compute_reconstruction_errors(windows: np.ndarray) -> np.ndarray:
    """
    Compute per-window reconstruction error.
    In production this loads the trained LSTM Autoencoder.
    For demonstration, we simulate errors with statistical deviation.
    """
    # Simulate reconstruction errors (replace with real model inference)
    mean_per_window = np.mean(windows, axis=(1, 2))
    errors = np.abs(mean_per_window - np.median(mean_per_window))
    return errors


def detect_anomalies(errors: np.ndarray, percentile: float = 95) -> list:
    """Flag windows with error above the threshold."""
    threshold = np.percentile(errors, percentile)
    anomaly_indices = np.where(errors > threshold)[0].tolist()
    return anomaly_indices, threshold


# -------------------------------------------------------------------
# 2. Build structured context for the LLM
# -------------------------------------------------------------------
def build_anomaly_report(df: pd.DataFrame, anomaly_indices: list,
                         window_size: int = 50, step_size: int = 25) -> list:
    """
    For each anomalous window, extract summary statistics
    that an LLM can use to generate explanations.
    """
    reports = []
    columns = list(df.columns)
    for idx in anomaly_indices[:5]:  # Limit to top 5 for API cost control
        start = idx * step_size
        end = start + window_size
        window_df = df.iloc[start:end]
        report = {
            "window_index": int(idx),
            "time_range": f"rows {start}-{end}",
            "features": columns,
            "mean_values": window_df.mean().round(4).to_dict(),
            "std_values": window_df.std().round(4).to_dict(),
            "min_values": window_df.min().round(4).to_dict(),
            "max_values": window_df.max().round(4).to_dict(),
        }
        reports.append(report)
    return reports


# -------------------------------------------------------------------
# 3. LLM Agent — Generate natural-language explanations
# -------------------------------------------------------------------
def explain_anomalies_with_llm(reports: list) -> str:
    """
    Send anomaly context to an OpenAI-compatible LLM and receive
    a structured, human-readable explanation.

    This pattern is directly applicable to:
    - Summarizing flagged legal documents
    - Explaining data quality issues in LLM training corpora
    - Generating automated audit reports
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[INFO] openai package not installed. Returning template response.")
        return _fallback_explanation(reports)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[INFO] OPENAI_API_KEY not set. Returning template response.")
        return _fallback_explanation(reports)

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a data quality analyst. Given anomaly reports from a sensor "
        "data pipeline, provide a concise explanation of what might have caused "
        "each anomaly. Focus on which features deviated and possible root causes. "
        "Format your response as a numbered list."
    )

    user_prompt = (
        "The following windows were flagged as anomalous by our LSTM Autoencoder "
        "pipeline. Please analyze and explain each:\n\n"
        + json.dumps(reports, indent=2)
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    return response.choices[0].message.content


def _fallback_explanation(reports: list) -> str:
    """Template-based fallback when no API key is available."""
    lines = ["=== Anomaly Explanation Report (Template Mode) ===\n"]
    for r in reports:
        lines.append(f"Window {r['window_index']} ({r['time_range']}):")
        lines.append(f"  Features analyzed: {', '.join(r['features'][:5])}...")
        lines.append(f"  Potential cause: Statistical deviation detected in sensor readings.\n")
    return "\n".join(lines)


# -------------------------------------------------------------------
# 4. Main pipeline: Preprocess → Detect → Explain
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("LLM-Powered Anomaly Explanation Agent")
    print("=" * 60)

    # Step 1: Load and preprocess
    data_path = Path(__file__).resolve().parent.parent / "data" / "filtered_sensor_data.csv"
    print(f"\n[1/4] Loading data...")
    try:
        df = load_data(str(data_path))
    except FileNotFoundError:
        print(f"  Data file not found. Creating synthetic demo data...")
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(500, 6),
            columns=[f"sensor_{i}" for i in range(6)]
        )
        # Inject synthetic anomaly
        df.iloc[400:420] *= 5

    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} features")

    # Step 2: Normalize & window
    print("[2/4] Preprocessing (normalize + sliding window)...")
    data_norm = normalize_data(df)
    windows = segment_windows(data_norm, window_size=50, step_size=25)
    print(f"  Created {windows.shape[0]} windows of shape {windows.shape[1:]}")

    # Step 3: Detect anomalies
    print("[3/4] Detecting anomalies...")
    errors = compute_reconstruction_errors(windows)
    anomaly_indices, threshold = detect_anomalies(errors)
    print(f"  Threshold: {threshold:.5f}")
    print(f"  Anomalous windows: {anomaly_indices}")

    # Step 4: Generate LLM explanation
    print("[4/4] Generating LLM-powered explanation...\n")
    reports = build_anomaly_report(df, anomaly_indices)
    explanation = explain_anomalies_with_llm(reports)
    print(explanation)
