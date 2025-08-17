# DemandFlow

AI-Powered Demand Forecasting Using Time-LLM

---

## Live Application

**Try it live:**  
[http://34.203.230.22:8501/](http://34.203.230.22:8501/)

---

**Model Applicability Notice:**
This model was specifically trained on the provided synthetic retail dataset. For accurate forecasting in other domains or for different stores, it is strongly recommended to retrain or fine-tune the model on relevant historical data from the target environment. Using out-of-domain data without retraining may lead to suboptimal or less meaningful predictions.

---

## Features and Workflow

DemandFlow is an intelligent demand forecasting platform that combines time series learning and Large Language Models. It allows businesses to estimate future inventory needs based on multiple factors, including weather, competitor pricing, promotions, and seasonality.

**Workflow:**

1. **Product and Region Selection**
   - Users choose the product and forecast period.
   - Region input determines the weather, holidays, and seasonality context.

2. **Data Enrichment**
   - The backend agent:
     - Fetches weather forecasts.
     - Identifies upcoming holidays or promotions.
     - Scrapes competitor pricing for the selected product in the chosen region.
   - The user is prompted to enter their planned price and discount.

3. **Forecasting**
   - All collected and user-provided inputs are processed by the TimeLLM model.
   - The model predicts the number of units likely to be sold.

4. **Explanation**
   - The prediction is explained using an agent hosted on AWS SageMaker.
   - Users see why the forecasted number was produced, referencing the input conditions.

---

## Data Sources

- The forecasting model was trained on an **artificial retail sales dataset** available here:  
  [https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset)

**Important Note:**  
This dataset is synthetic. While predictions can be realistic, they may not always be fully accurate in production scenarios.

---

## Testing Instructions

To evaluate the platform:

1. Visit [http://34.203.230.22:8501/](http://34.203.230.22:8501/).
2. Select a **region**, **product**, and **forecast period**.
3. Enter a **price and discount**.
4. Submit the form to get a forecast.

**Tips for Testing:**
- For more plausible predictions, use prices in a similar range to the dataset.
- Because this is a demonstration built on synthetic data, extremely high or low prices may lead to less accurate outputs.

---

## Project Structure

- `frontend/`
  - Streamlit app (user interface).
- `backend_server.py`
  - FastAPI backend serving prediction APIs.
- `models/`
  - TimeLLM model code and pre-trained weights.
- `testing.py`
  - Script to test predictions directly without the web interface.
- `main.py`
  - Example usage of SageMaker-hosted LLM for generating explanations.

---

## Future Scope

- Integrate real-time weather APIs.
- Connect to live e-commerce pricing feeds for competitor analysis.
- Support multiple languages.
- Implement user authentication and save historical forecasts.

---

## Deployment Details

- **Platform:** AWS EC2
- **Frontend:** Streamlit served via public IP
- **Backend:** FastAPI for model predictions
- **LLM Explanations:** AWS SageMaker endpoint

---

## Contact

For any questions or suggestions, please open an issue on GitHub.

---

