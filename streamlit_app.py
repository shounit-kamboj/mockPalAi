import asyncio
import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="MockPal AI", page_icon="🎤", layout="centered")


# =====================================================================================
# KEEP: helpers from the old app (unchanged)
# =====================================================================================
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    # app_id only labels who SENT the event; the backend's app_id is "rag_mock_interviewer".
    # Matching them is tidier but not required: events are routed by event NAME.
    return inngest.Inngest(app_id="rag_mock_interviewer", is_production=False)


def get_event_loop() -> asyncio.AbstractEventLoop:
    # Streamlit reruns this whole script on every click, so keep one event loop in session_state
    if "event_loop" not in st.session_state:
        st.session_state.event_loop = asyncio.new_event_loop()
    return st.session_state.event_loop


def run_async(coro):
    loop = get_event_loop()
    return loop.run_until_complete(coro)


def save_uploaded_pdf(file) -> Path:
    # Same as before: the backend reads the resume from disk, so we pass a PATH in the event,
    # never the file contents (Inngest events have a size limit).
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    resp = requests.get(f"{_inngest_api_base()}/events/{event_id}/runs")
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    # Asks the Inngest dev server "is the run for this event finished?" every 0.5s.
    # Not needed until step 5 (there's no backend function for the new event yet).
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


# =====================================================================================
# NEW: send the interview event (modelled on the old send_rag_query_event)
# =====================================================================================
async def send_interview_start_event(resume_path: Path, jd_text: str) -> str:
    client = get_inngest_client()
    result = await client.send(
              inngest.Event(
                  name="rag/interview_start",     # the backend function in step 4 listens for this name
                  data={
                      "resume_path": str(resume_path.resolve()),
                      "jd_text": jd_text,
                  },
              )
          )
    return result[0]


# =====================================================================================
# NEW: the page
# =====================================================================================
st.title("MockPal AI")
st.caption("Upload your resume and paste a job description to get a tailored mock interview.")

# A form groups the inputs so the script only acts when the button is pressed,
# instead of on every keystroke/upload.
with st.form("interview_form"):
    resume = st.file_uploader("Upload your resume", accept_multiple_files=False, type="pdf")
    jd_text = st.text_area("Paste your job description", height=200)
    submitted = st.form_submit_button("Start interview")

if submitted:
    if not resume or not jd_text.strip():
        st.error("Please upload a resume and paste a job description.")
        st.stop()
    path = save_uploaded_pdf(resume)
    event_id = run_async(send_interview_start_event(path, jd_text.strip()))
    with st.spinner("Writing your interview..."):
        output = wait_for_run_output(event_id)

    questions = output.get("questions", [])
    if not questions:
        st.warning("No questions came back. Check the run at localhost:8288.")
    else:
        st.subheader("Your interview")
        for i, q in enumerate(questions, start=1):
            st.markdown(f"**{i}.** {q['question']}")