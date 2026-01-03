import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Divine Speech Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern look
st.markdown(
    """
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41424b;
    }
    .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stExpander"] {
        background-color: #262730;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #41424b;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Constants
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "llm_history.jsonl"

st.title("🧠 Divine Speech Live Monitor")
st.markdown("Watching for LLM interactions in real-time...")


# Helpers
def load_logs():
    data = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                # Read all lines
                lines = f.readlines()
                # Parse last 50 lines to keep it fast
                for line in lines[-50:]:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            st.error(f"Error reading logs: {e}")
    return data


# Main Loop placeholder
main_placeholder = st.empty()

while True:
    with main_placeholder.container():
        logs = load_logs()

        if not logs:
            st.info(
                "No logs found yet. Run the optimization/training script to see activity."
            )
            st.code("Waiting for logs at: " + str(LOG_FILE.absolute()))
        else:
            # Stats Area
            df = pd.DataFrame(logs)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Interactions", len(df))
            with col2:
                avg_time = df["duration"].mean() if "duration" in df.columns else 0
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
            with col3:
                last_time = df.iloc[-1]["duration"] if "duration" in df.columns else 0
                st.metric("Last Latency", f"{last_time:.2f}s")
            with col4:
                success_rate = (
                    (df["status"] == "success").mean() * 100
                    if "status" in df.columns
                    else 0
                )
                st.metric("Success Rate", f"{success_rate:.1f}%")

            st.markdown("---")
            st.subheader("Recent Activity Stream")

            # Activity Feed (Reverse chronological)
            for i, log in enumerate(reversed(logs)):
                timestamp = log.get("timestamp", "Unknown Time")
                duration = log.get("duration", 0)
                status = log.get("status", "unknown")
                status_icon = "✅" if status == "success" else "❌"

                # Determine title based on context if available
                prompt_preview = log.get("prompt", "")[:50] + "..."

                expander_title = f"{status_icon} [{timestamp}] Interaction #{len(logs)-i} ({duration:.2f}s)"

                # Colors based on status
                border_color = "green" if status == "success" else "red"

                with st.expander(expander_title, expanded=(i == 0)):
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown("### 📤 Prompt")
                        st.code(log.get("prompt", ""), language="text")
                    with c2:
                        st.markdown("### 📥 Response")
                        response_content = log.get("response", "")
                        # Try to detect code in response
                        if "```" in response_content:
                            st.markdown(response_content)
                        else:
                            st.code(response_content, language="json")

                    if "error" in log:
                        st.error(f"Error Details: {log['error']}")

    time.sleep(1)  # Refresh every 1 second
