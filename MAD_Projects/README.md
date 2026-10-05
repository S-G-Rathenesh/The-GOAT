# Goat Fair-Weight Analyzer

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

## Example output
The app provides:
- Health classification: Healthy Goat / Unhealthy Goat
- Estimated weight in kg
- Expected weight range
- Deviation percentage
- Temporary Weight Anomaly Score
- Fair-trade decision status
