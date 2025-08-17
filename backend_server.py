from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import json
import datetime
try:
    from serpapi import GoogleSearch
except ImportError:
    print("Warning: serpapi not installed. Install with: pip install google-search-results")
    GoogleSearch = None

try:
    from testing import predict_units_sold
except ImportError:
    print("Warning: testing module not found. Using fallback prediction.")
    def predict_units_sold(**kwargs):
        return 55.0

app = FastAPI(title="DemandFlow API")

# Add CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your existing configuration
SERPAPI_KEY = "0af24643f792c32a0e3623bd42cc5e685e6ec72bfb45b497a37a678331b27076"
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
endpoint_name = "jumpstart-dft-meta-textgeneration-l-20250702-160644"

class PredictionRequest(BaseModel):
    city: str
    product: str
    days_ahead: int
    price: float
    discount: float

# Copy your exact functions from main.py
def serp_search(query, location, num_results=5):
    print(f"\n[🔍 SerpAPI QUERY]: '{query}' @ '{location}'")
    if GoogleSearch is None:
        return ["SerpAPI not available - using mock data"]
    
    try:
        params = {
            "engine": "google",
            "q": query,
            "location": location,
            "hl": "en",
            "gl": "in",
            "google_domain": "google.co.in",
            "num": num_results,
            "api_key": SERPAPI_KEY
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        if "organic_results" not in results or not results["organic_results"]:
            return ["No results available."]
        return [r.get("title", "") + " " + r.get("snippet", "") for r in results["organic_results"]]
    except Exception as e:
        print(f"SerpAPI error: {e}")
        return ["Search temporarily unavailable"]

def call_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.3,
            "top_p": 0.9
        }
    }
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    if isinstance(result, list):
        return result[0]["generated_text"]
    elif isinstance(result, dict):
        return result.get("generated_text", "No text generated.")
    else:
        return "Unexpected response format from LLM."

def summarize_category(label, texts):
    import re
    joined_text = "\n".join(texts)
    if label.lower() == "weather":
        keywords = ["cloud", "rain", "storm", "sun", "clear", "humid", "breezy", "showers", "fog", "wind"]
        cleaned = []
        for line in joined_text.splitlines():
            cleaned_line = re.sub(r'[^a-zA-Z\\s]', '', line.lower())
            if any(word in cleaned_line for word in keywords):
                cleaned.append(line.strip())
        if cleaned:
            joined_text = "\n".join(cleaned)
    
    trend_hint = ""
    if label.lower() == "weather":
        trend_hint = "\n\nWrite a short 1-sentence summary describing sky conditions, rain chance, and temperature."

    prompt = f"""You are a weather summarizer.

Below is a weather forecast for a city over the next few days. Summarize it in exactly one helpful sentence.{trend_hint}

Forecast:
{joined_text}"""
    return call_llm(prompt).strip()

@app.post("/predict")
async def predict_demand(request: PredictionRequest):
    try:
        print(f"\n=== Processing prediction for {request.product} in {request.city} ===")
        
        location = f"{request.city}, India"
        start_date = datetime.date.today()
        end_date = start_date + datetime.timedelta(days=request.days_ahead)
        date_range_str = f"from {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"

        weather_texts = serp_search(f"weather forecast in {request.city} for next {request.days_ahead} days", location)[:3]
        seasonality_texts = serp_search(f"sales demand trends for {request.product} in {request.city}", location)[:3]
        competitor_texts = serp_search(f"{request.product} price in {request.city}", location)[:3]
        holidays_texts = serp_search(f"public holidays in {request.city} {date_range_str}", location, 10)[:3]

        weather_summary = summarize_category("weather", weather_texts)
        seasonality_summary = summarize_category("seasonality", seasonality_texts)
        competitor_summary = summarize_category("competitor pricing", competitor_texts)
        holiday_summary = summarize_category("holidays", holidays_texts)

        holiday_flag = "holiday" in holiday_summary.lower()
        
        prediction = predict_units_sold(
            days_ahead=request.days_ahead,
            discount=request.discount,
            holiday=holiday_flag,
            comp_price=0,
            weather=weather_summary,
            seasonality=seasonality_summary,
            desired_price=request.price
        )

        explanation_prompt = f"""Explain why the predicted units sold is {round(prediction, 2)}.

City: {request.city}
Product: {request.product}
Forecast horizon: {request.days_ahead} days
Desired Price: {request.price}
Discount: {request.discount}%
Holiday/Promotion: {'Yes' if holiday_flag else 'No'}

Context:
1. Weather: {weather_summary}
2. Seasonality: {seasonality_summary}
3. Competitor Pricing: {competitor_summary}
4. Holidays: {holiday_summary}

Give a short reason for this prediction in 1–2 lines."""

        explanation = call_llm(explanation_prompt)

        return {
            "predicted_units": float(round(prediction, 1)),
            "explanation": explanation.strip()
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)