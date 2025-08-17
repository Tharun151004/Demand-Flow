import boto3
import json
import datetime
from serpapi import GoogleSearch
from testing import predict_units_sold

SERPAPI_KEY = "SERP_API"
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
endpoint_name = "EndPoint"

def serp_search(query, location, num_results=5):
    print(f"\n[SerpAPI QUERY]: '{query}' @ '{location}'")
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
        print("⚠  No results found for this query.")
        return ["No results available."]
    return [r.get("title", "") + " " + r.get("snippet", "") for r in results["organic_results"]]

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
    print(" Raw LLM Response:", result)
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

    prompt = f"""
You are a weather summarizer.

Below is a weather forecast for a city over the next few days. Summarize it in exactly one helpful sentence.{trend_hint}

Forecast:
{joined_text}
"""
    print(f"\n Prompt for {label}:\n{prompt}\n")
    return call_llm(prompt).strip()


def main():
    print("\n=== Demand Forecasting Agent ===\n")
    city = input("Enter your CITY: ").strip()
    product = input("Enter PRODUCT to forecast: ").strip()

    timeframes = {
        "1": ("1 Day", 1),
        "2": ("1 Week", 7),
        "3": ("2 Weeks", 14),
        "4": ("3 Weeks", 21)
    }

    print("\nSelect TIME FRAME:")
    for k, v in timeframes.items():
        print(f"{k}. {v[0]}")
    while True:
        tf_choice = input("Enter choice [1-4]: ").strip()
        if tf_choice in timeframes:
            timeframe_label, days_ahead = timeframes[tf_choice]
            break

    print("\nSearching Google via SerpAPI...")
    location = f"{city}, India"
    start_date = datetime.date.today()
    end_date = start_date + datetime.timedelta(days=days_ahead)
    date_range_str = f"from {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"

    weather_texts = serp_search(f"weather forecast in {city} for next {days_ahead} days", location)[:3]
    seasonality_texts = serp_search(f"sales demand trends for {product} in {city}", location)[:3]
    competitor_texts = serp_search(f"{product} price in {city}", location)[:3]
    holidays_texts = serp_search(f"public holidays in {city} {date_range_str}", location, 10)[:3]

    #print("\nRAW SEARCH RESULTS")
    #print("\nWeather")
    for t in weather_texts: print(t)
    #print("\nSeasonality")
    for t in seasonality_texts: print(t)
    #print("\n Competitor Pricing ")
    for t in competitor_texts: print(t)
    #print("\nHolidays")
    for t in holidays_texts: print(t)

    print("\nCalling LLM to summarize each category...\n")
    weather_summary = summarize_category("weather", weather_texts)
    seasonality_summary = summarize_category("seasonality", seasonality_texts)
    competitor_summary = summarize_category("competitor pricing", competitor_texts)
    holiday_summary = summarize_category("holidays", holidays_texts)

    print("\n[SUMMARY FROM LLM]")
    print(f"1. Weather: {weather_summary}")
    print(f"2. Seasonality: {seasonality_summary}")
    print(f"3. Competitor Pricing: {competitor_summary}")
    print(f"4. Holidays: {holiday_summary}")

    desired_price = input("\nEnter DESIRED PRICE (any text or number): ").strip()
    discount = input("Enter DISCOUNT (%): ").strip()
    holiday_flag = "Yes" if "holiday" in holiday_summary.lower() else "No"

    prediction = predict_units_sold(
        days_ahead=days_ahead,
        discount=float(discount) if discount else 0,
        holiday=(holiday_flag == "Yes"),
        comp_price=0,
        weather=weather_summary,
        seasonality=seasonality_summary,
        desired_price=float(desired_price) if desired_price.replace(".", "").isdigit() else 0
    )

    print(f"\n[PREDICTION]\nEstimated Units Sold: {round(prediction, 2)}")

    explanation_prompt = (
        f"Explain why the predicted units sold is {round(prediction, 2)}.\n\n"
        f"City: {city}\n"
        f"Product: {product}\n"
        f"Forecast horizon: {timeframe_label}\n"
        f"Desired Price: {desired_price}\n"
        f"Discount: {discount}%\n"
        f"Holiday/Promotion: {holiday_flag}\n\n"
        f"Context:\n"
        f"1. Weather: {weather_summary}\n"
        f"2. Seasonality: {seasonality_summary}\n"
        f"3. Competitor Pricing: {competitor_summary}\n"
        f"4. Holidays: {holiday_summary}\n\n"
        f"Give a short reason for this prediction in 1–2 lines."
    )

    print(f"\n Explanation Prompt Sent to LLM:\n{explanation_prompt}\n")
    explanation = call_llm(explanation_prompt)

    print("\n[EXPLANATION]")
    print(explanation)

if __name__ == "__main__":
    main()
