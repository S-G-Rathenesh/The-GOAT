# THE GOAT

A mobile-friendly AI application for goat weight estimation and temporary weight anomaly detection using smartphone images.

## Features
- Image-based goat health classification using the provided dataset
- Feature extraction for body length, body height, chest girth, and body fullness
- Breed-aware expected weight estimation using breed, age, and sex adjustments
- Temporary Weight Anomaly Score based on weight deviation and body fullness
- Fair-trade decision support for goat transactions
- Streamlit web app interface for quick testing and demonstration

## Dataset
The project uses the provided Goat_Dataset folder:
- healthy_goat/
- unhealthy_goat/

## Project structure
- app.py - Streamlit user interface
- src/goat_analysis.py - image processing and ML pipeline
- requirements.txt - Python dependencies

## Setup
1. Open a terminal in the project folder.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the application:
   streamlit run app.py

## Usage
- Upload a goat image or use a sample from the dataset
- Select breed, age, sex, and observed weight
- Click Analyze goat
- Review the estimated weight, expected weight range, anomaly score, and fair-trade indication

## Deploy on Streamlit Community Cloud

This project is a Streamlit application, so it should be deployed with a Streamlit
runtime rather than Vercel's Python serverless runtime. Vercel expects `app.py` to
export a WSGI/ASGI variable named `app`, `application`, or `handler`; this
application is a Streamlit script and intentionally does not export one.

1. Push the project to GitHub with `app.py`, `src/`, `models/`, `Goat_Dataset/`,
   and `requirements.txt` in the same project directory.
2. Open [share.streamlit.io](https://share.streamlit.io/) and choose **Deploy an
   app**.
3. Select the repository and branch.
4. Set **Main file path** to `app.py`. If the repository contains a parent
   folder, select the folder that contains `app.py` as the app root.
5. Deploy. Streamlit Community Cloud installs the packages from
   `requirements.txt` automatically.

Do not add a Vercel Python function wrapper around `app.py`; it would make the
build pass but would not provide a working Streamlit session.

## Deploy on Render

For a Render **Web Service**, use `MAD_Projects` as the root directory and use:

```text
Build command: pip install -r requirements.txt
Start command: streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

The `runtime.txt` file pins Render to Python 3.12 because the pinned NumPy and
pandas versions require a Python version with compatible prebuilt wheels.

## Example output
The app provides:
- Health classification: Healthy Goat / Unhealthy Goat
- Estimated weight in kg
- Expected weight range
- Deviation percentage
- Temporary Weight Anomaly Score
- Fair-trade decision status
