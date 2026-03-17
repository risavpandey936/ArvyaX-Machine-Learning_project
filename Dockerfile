# Use a lightweight python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list and install (caching layers)
COPY requirements.txt .

# Optimize install size by not caching pip downloads
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make sure our start script is executable
RUN chmod +x start.sh

# Expose ports for both APIs
# Streamlit will map to 8501 inside the container
EXPOSE 8501
# FastAPI will map to 8000 inside the container
EXPOSE 8000

# When the container boots, run both applications
CMD ["./start.sh"]
