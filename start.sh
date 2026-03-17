#!/bin/bash

# Start the FastAPI backend on port 8000 in the background
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend on port 8501 in the foreground 
# (Streamlit will keep the container running)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
