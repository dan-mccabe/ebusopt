# For local testing on Apple Silicon, change amd64 to arm64
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
# Bring in all files needed for the app, but no more than what's necessary
COPY beb_chargers/gtfs_beb ./beb_chargers/gtfs_beb
COPY beb_chargers/zebra ./beb_chargers/zebra
COPY beb_chargers/data/gtfs/metro_may24 ./beb_chargers/data/gtfs/metro_may24
COPY beb_chargers/data/gtfs/trimet_may24 ./beb_chargers/data/gtfs/trimet_may24
COPY requirements.txt .
COPY setup.py .
COPY add_ga_tag.py .
COPY summarize_gtfs.py .
COPY __init__.py .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install -e .
RUN python3 add_ga_tag.py
RUN python3 summarize_gtfs.py

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "beb_chargers/zebra/🏠_Home.py", "--server.port=8080", "--server.address=0.0.0.0"]
