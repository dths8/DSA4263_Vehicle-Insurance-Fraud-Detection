<p align="center">
  <strong style="font-size:64px;">DSA4263 Automobiles Insurance Fraud Detection</strong> <br>
  <em>Machine Learning for Detecting Fraudulent Insurance Claim Cases</em> <br>
</p>

# Project Overview
Insurance fraud remains a persistent challenge that costs the industry billions of dollars each year and undermines the integrity of legitimate claims. Fraudulent activities, such as false claims, exaggerated losses, or coordinated fraud rings, not only lead to financial losses but also increase premiums for honest policyholders and erode trust in insurers.

Traditional fraud detection in insurance often relies on manual reviews or rule-based systems. While these methods can identify obvious cases, they struggle with new and complex fraud patterns that evolve over time. Static rules can result in missed fraud cases or excessive false positives, both of which reduce operational efficiency and customer satisfaction.

This project explores the application of machine learning methods to transform fraud detection in the insurance sector by leveraging data-driven insights. Our goal is to build a more intelligent fraud detection system that enhances accuracy, reduces manual workload, and safeguards both insurers and customers.

# Project Structure
Below is the project structure:
```
DSA4263_Vehicle-Insurance-Fraud-Detection/
├── data/                                     # Data files
├── model_training/                           # Model training files
│   ├── unsupervised_learning_models.ipynb    # Notebook for unsupervised learning models
│   ├── network_models.ipynb                  # Notebook for network models
│   └── supervised_learning_models.ipynb      # Notebook for supervised learning models
├── models/                                   # Saved models
├── streamlit/                                # Streamlit files
├── .gitignore                                # Git ignore file
├── Dockerfile                                # Dockerfile
├── README.md                                 # Project documentation
├── app.py                                    # Dashboard app
├── data_dictionary.txt                       # Data dictionary file
└── requirements.txt                          # Requirements file
```


# Cloning repository

You can clone the repository here:
```
git clone https://github.com/dths8/DSA4263_Vehicle-Insurance-Fraud-Detection.git
```

# Running the Dashboard with Docker

## Prerequisites

- Please make sure that you have Docker installed on your local machine. You can download and install Docker from https://www.docker.com/products/docker-desktop/.
- The dashboard uses Python 3.10 as specified in the Dockerfile.

## Build and Run Instructions
1. Navigate to the dashboard directory
   ```bash
   cd DSA4263_Vehicle-Insurance-Fraud-Dectection
   ```
2. Build the Docker image :

   ```bash
   docker build -t my-streamlit-app .
   ```
3. Run the Docker image :

   ```bash
   docker run -p 8501:8501 my-streamlit-app
   ```
4. The application will be accessible at `http://localhost:8501`.

## Configuration

- The application exposes port `8501` as defined in the Docker Compose file.
- No additional environment variables are required for this setup.

For further details, refer to the project documentation or contact the development team.

# Dashboard Guide
## Project Description and Workflow
After running the Docker image and accessing the application, you should see a brief description and workflow of our project:
<img width="677" height="596" alt="Screenshot 2025-11-19 at 1 04 49 AM" src="https://github.com/user-attachments/assets/69e98cab-e8a4-477e-8bd1-67068ec9d08a" />

## Model Simulation
Following this introduction, you will find the model simulation section. This interactive section allows you to input features that precisely match the data found in the original dataset. Insurance claimants have to fill these up before submitting their case for claim processing. Once all claim details are entered, selecting the 'Submit Claim for Fraud Prediction' button triggers the model to instantly assess the claim and determine its probability of being fraudulent.

<img width="677" height="589" alt="Screenshot 2025-11-19 at 1 04 58 AM" src="https://github.com/user-attachments/assets/abdd43e5-838a-4d49-8a16-97ea4f852c63" />
<img width="677" height="886" alt="Screenshot 2025-11-19 at 1 05 07 AM" src="https://github.com/user-attachments/assets/d9d69351-f2cd-4d2e-9faf-8f35915ca556" />

## Prediction Result and Explanability
Lastly, after clicking the button, the model's prediction result will be generated to show the probability of the claim being fraudulent and also a recommendation for the action to undertake based on the prediction result. Below that, there will also be an explanation section that details the top features driving the model's prediction, showing which factors most strongly contributed to the claim being flagged as fraudulent or deemed legitimate.

<img width="677" height="864" alt="Screenshot 2025-11-19 at 1 05 24 AM" src="https://github.com/user-attachments/assets/85254ae6-0281-4353-949a-7fa7680ee413" />

<h2>Contributors</h2>
This project is developed by the following developers:<br>

| Name            | Github                                      |
|-----------------|---------------------------------------------|
| Derek Tan       | [dths8](https://github.com/dths8)           |
| Jun An Koh      | [Junan86](https://github.com/Junan86)       |
| Marcus Yong     | [honeylew01](https://github.com/honeylew01) |
