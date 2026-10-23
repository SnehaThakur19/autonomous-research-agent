"""Streamlit Community Cloud entrypoint.

The cloud runs: streamlit run streamlit_app.py
which executes the multi-step wizard in app/webapp.py.
"""

import runpy

runpy.run_path("app/webapp.py", run_name="__main__")
