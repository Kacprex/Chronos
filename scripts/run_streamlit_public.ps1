# Read-only public dashboard (binds to all interfaces)
.\.venv\Scripts\activate
streamlit run python\streamlit\public_dashboard\app.py --server.port 8601 --server.address 0.0.0.0
