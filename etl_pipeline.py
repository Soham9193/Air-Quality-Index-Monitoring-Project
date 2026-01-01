import requests
import pandas as pd
import datetime
import time
import os
import random
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.linear_model import LinearRegression
import numpy as np

# --- CONFIGURATION ---
API_KEY = "4588f68c2f9bb2f4fa7f7e0ef316a211"  # <--- PASTE KEY HERE
CITIES = ["Mumbai", "Thane", "Navi Mumbai"]
GOOGLE_SHEET_NAME = "Weather_SDP_Data"
JSON_KEYFILE = "credentials.json"
OUTPUT_FILE = "aqi_live_data.csv"

# Define Headers Globally to ensure consistency
HEADERS = [
    "City", "Latitude", "Longitude", "Timestamp", 
    "AQI_Level", "PM2_5", "PM10", "CO", "NO2", 
    "Temperature", "Humidity", "Wind_Speed", 
    "Health_Advice", "Predicted_PM2_5"
]

# --- GOOGLE SHEETS AUTH ---
def connect_to_sheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(JSON_KEYFILE, scope)
        client = gspread.authorize(creds)
        sheet = client.open(GOOGLE_SHEET_NAME).sheet1
        return sheet
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to Google Sheets ({e})")
        return None

def check_and_add_sheet_headers(sheet):
    """
    Checks if the Google Sheet is empty. If so, adds the headers.
    """
    if not sheet: return
    try:
        # Check if first row is empty
        if not sheet.row_values(1):
            print(" -> Google Sheet is empty. Adding Headers...")
            sheet.append_row(HEADERS)
    except Exception as e:
        print(f" -> Error checking sheet headers: {e}")

def update_google_sheet(sheet, data_dict):
    if not sheet: return

    # Create row list in exact order of HEADERS
    row = [data_dict.get(h, "") for h in HEADERS]
    
    try:
        sheet.append_row(row)
        print(f" -> Uploaded {data_dict['City']} to Google Sheets.")
    except Exception as e:
        print(f" -> Failed to upload to Sheets: {e}")

# --- HELPER FUNCTIONS ---

def get_health_advice(aqi_score):
    if aqi_score <= 2: return "Safe. Enjoy outdoor activities."
    elif aqi_score == 3: return "Moderate. Sensitive groups reduce exertion."
    elif aqi_score >= 4: return "⚠️ High Pollution! Wear a mask."
    return "Unknown"

def get_lat_lon(city_name):
    url = f"https://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
    try:
        response = requests.get(url).json()
        if isinstance(response, list) and len(response) > 0:
            return response[0]['lat'], response[0]['lon']
    except:
        pass
    return None, None

def fetch_combined_data(city):
    lat, lon = get_lat_lon(city)
    if not lat: return None
    
    # 1. Fetch Air Pollution
    poll_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    poll_data = requests.get(poll_url).json()
    
    # 2. Fetch Weather
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    weather_data = requests.get(weather_url).json()

    if 'list' in poll_data and 'main' in weather_data:
        components = poll_data['list'][0]['components']
        aqi = poll_data['list'][0]['main']['aqi']
        
        return {
            "City": city,
            "Latitude": lat,
            "Longitude": lon,
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "AQI_Level": aqi,
            "PM2_5": components['pm2_5'],
            "PM10": components['pm10'],
            "CO": components['co'],
            "NO2": components['no2'],
            "Temperature": weather_data['main']['temp'],
            "Humidity": weather_data['main']['humidity'],
            "Wind_Speed": weather_data['wind']['speed'],
            "Health_Advice": get_health_advice(aqi)
        }
    return None

def predict_future_aqi(df, city):
    city_data = df[df['City'] == city].tail(20)
    
    if len(city_data) < 5:
        return city_data.iloc[-1]['PM2_5'] * random.uniform(0.9, 1.1)
    
    X = np.array(range(len(city_data))).reshape(-1, 1)
    y = city_data['PM2_5'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    prediction = model.predict([[len(city_data) + 1]])
    return max(0, prediction[0])

# --- MAIN JOB ---
def run_job():
    print(f"\n--- Fetching Data: {datetime.datetime.now()} ---")
    sheet = connect_to_sheets()
    
    # Auto-add headers to Google Sheet if needed
    check_and_add_sheet_headers(sheet)
    
    # Load Local CSV
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        # Ensure all columns exist
        for col in HEADERS:
            if col not in df.columns: df[col] = 0
    else:
        df = pd.DataFrame(columns=HEADERS)

    new_rows = []
    
    for city in CITIES:
        data = fetch_combined_data(city)
        if data:
            # Fix for FutureWarning: concat logic
            current_row_df = pd.DataFrame([data])
            
            # Temporary DF for prediction logic
            if not df.empty:
                temp_df = pd.concat([df, current_row_df], ignore_index=True)
            else:
                temp_df = current_row_df
            
            predicted_val = predict_future_aqi(temp_df, city)
            data["Predicted_PM2_5"] = round(predicted_val, 2)
            
            new_rows.append(data)
            print(f"Fetched {city}: AQI={data['AQI_Level']} | Temp={data['Temperature']}°C | Wind={data['Wind_Speed']}m/s")
            
            # Upload to Google Sheet
            update_google_sheet(sheet, data)

    # Save to Local CSV (with retry logic)
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        save_success = False
        attempts = 0
        while not save_success and attempts < 3:
            try:
                # header=True only if file doesn't exist
                write_header = not os.path.exists(OUTPUT_FILE)
                new_df.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False)
                save_success = True
                print(" -> Data saved to local CSV.")
            except PermissionError:
                attempts += 1
                print(" -> File locked by PowerBI. Retrying...")
                time.sleep(2)

# --- EXECUTION ---
# Run once immediately to verify
run_job()

print("\n✅ Scheduler Started. Running every 15 minutes...")
print("Press Ctrl+C to stop the script.")

while True:
    time.sleep(900) # 900 seconds = 15 minutes
    run_job()