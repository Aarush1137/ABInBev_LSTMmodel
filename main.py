from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.utils
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load pre-trained LSTM model
model = load_model(r'E:\aninbev\temp\lstm_hoe.keras')

# Indian cities coordinates
CITY_COORDINATES = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Delhi": {"lat": 28.7041, "lon": 77.1025},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Surat": {"lat": 21.1702, "lon": 72.8311},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Kanpur": {"lat": 26.4499, "lon": 80.3319},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Indore": {"lat": 22.7196, "lon": 75.8577},
    "Thane": {"lat": 19.2183, "lon": 72.9781},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185},
    "Patna": {"lat": 25.5941, "lon": 85.1376},
    "Vadodara": {"lat": 22.3072, "lon": 73.1812},
    "Ghaziabad": {"lat": 28.6692, "lon": 77.4538},
    "Ludhiana": {"lat": 30.9009, "lon": 75.8573},
    "Agra": {"lat": 27.1767, "lon": 78.0081},
    "Nashik": {"lat": 19.9975, "lon": 73.7898},
    "Faridabad": {"lat": 28.4089, "lon": 77.3178},
    "Meerut": {"lat": 28.9845, "lon": 77.7064},
    "Rajkot": {"lat": 22.3039, "lon": 70.8022},
    "Kalyan-Dombivli": {"lat": 19.2403, "lon": 73.1305},
    "Vasai-Virar": {"lat": 19.3919, "lon": 72.8397},
    "Varanasi": {"lat": 25.3176, "lon": 82.9739},
    "Srinagar": {"lat": 34.0837, "lon": 74.7973},
    "Aurangabad": {"lat": 19.8762, "lon": 75.3433},
    "Dhanbad": {"lat": 23.7957, "lon": 86.4304},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723},
    "Navi Mumbai": {"lat": 19.0330, "lon": 73.0297},
    "Allahabad": {"lat": 25.4358, "lon": 81.8463},
    "Ranchi": {"lat": 23.3441, "lon": 85.3096},
    "Howrah": {"lat": 22.5958, "lon": 88.2636},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558},
    "Jabalpur": {"lat": 23.1815, "lon": 79.9864},
    "Gwalior": {"lat": 26.2183, "lon": 78.1828},
    "Vijayawada": {"lat": 16.5062, "lon": 80.6480},
    "Jodhpur": {"lat": 26.2389, "lon": 73.0243},
    "Madurai": {"lat": 9.9252, "lon": 78.1198},
    "Raipur": {"lat": 21.2514, "lon": 81.6296},
    "Kota": {"lat": 25.2138, "lon": 75.8648},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794},
    "Solapur": {"lat": 17.6599, "lon": 75.9064},
    "Hubballi-Dharwad": {"lat": 15.3647, "lon": 75.1239},
    "Mysore": {"lat": 12.2958, "lon": 76.6394},
    "Tiruchirappalli": {"lat": 10.7905, "lon": 78.7047},
    "Bareilly": {"lat": 28.3670, "lon": 79.4304},
    "Aligarh": {"lat": 27.8974, "lon": 78.0880},
    "Moradabad": {"lat": 28.8386, "lon": 78.7733},
    "Jalandhar": {"lat": 31.3260, "lon": 75.5762},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245},
    "Salem": {"lat": 11.6643, "lon": 78.1460},
    "Warangal": {"lat": 17.9784, "lon": 79.5941},
    "Guntur": {"lat": 16.3067, "lon": 80.4365},
    "Bhiwandi": {"lat": 19.2813, "lon": 73.0483},
    "Saharanpur": {"lat": 29.9679, "lon": 77.5510},
    "Bathinda": {"lat": 26.7606, "lon": 83.3732},
    "Bikaner": {"lat": 28.0229, "lon": 73.3119},
    "Amravati": {"lat": 20.9320, "lon": 77.7523},
    "Noida": {"lat": 28.5355, "lon": 77.3910},
    "Jamshedpur": {"lat": 22.8046, "lon": 86.2029},
    "Bhilai": {"lat": 21.1938, "lon": 81.3509},
    "Cuttack": {"lat": 20.4625, "lon": 85.8828},
    "Firozabad": {"lat": 27.1591, "lon": 78.3958},
    "Kochi": {"lat": 9.9312, "lon": 76.2673},
    "Dehradun": {"lat": 30.3165, "lon": 78.0322},
    "Durgapur": {"lat": 23.5330, "lon": 87.3215},
    "Ajmer": {"lat": 26.4499, "lon": 74.6399}
}

def preprocess_data(file):
    """Preprocess uploaded CSV data."""
    sales_data = pd.read_csv(file)
    numerical_cols = [col for col in sales_data.columns if col.startswith('hoe_')]
    sales = sales_data[numerical_cols]
    
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)
    return sales_data, sales_scaled, scaler, numerical_cols

def create_sequences(data, time_steps=4):
    """Create sequences for LSTM model."""
    sequences = []
    for i in range(len(data) - time_steps):
        seq = data[i:(i + time_steps)]
        sequences.append(seq)
    return np.array(sequences)

def generate_forecast(sales_scaled, scaler, n_features):
    """Generate forecasts using the LSTM model."""
    time_steps = 4
    sequences = create_sequences(sales_scaled, time_steps)
    
    if len(sequences) == 0:
        raise ValueError("Not enough data points to create sequences")
    
    predictions = model.predict(sequences[-1].reshape(1, time_steps, n_features))
    predictions_reshaped = predictions.reshape(1, n_features)
    forecast = scaler.inverse_transform(predictions_reshaped).flatten()
    return forecast

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        sales_data, sales_scaled, scaler, numerical_cols = preprocess_data(file)
        n_features = len(numerical_cols)
        
        try:
            forecast = generate_forecast(sales_scaled, scaler, n_features)
            cities = sales_data['City'].tolist()

            # Prepare map data with coordinates
            map_data = []
            forecast_data = []
            
            for city, forecast_value in zip(cities, forecast):
                # Add to forecast data list
                forecast_data.append({
                    'city': city,
                    'forecast': round(float(forecast_value), 2)
                })
                
                # Add to map data if coordinates exist
                if city in CITY_COORDINATES:
                    map_data.append({
                        'City': city,
                        'Latitude': CITY_COORDINATES[city]['lat'],
                        'Longitude': CITY_COORDINATES[city]['lon'],
                        'Forecast': round(float(forecast_value), 2)
                    })
            
            map_df = pd.DataFrame(map_data)

            # Create Mapbox Plot
            fig = px.scatter_mapbox(
                map_df,
                lat="Latitude",
                lon="Longitude",
                size="Forecast",
                color="Forecast",
                hover_name="City",
                mapbox_style="carto-positron",
                title="Sales Forecast for Indian Cities",
                zoom=4
            )
            
            # Prepare data for template
            template_data = {
                'map_json': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'forecast_data': forecast_data,
                'total_cities': len(forecast_data),
                'average_forecast': round(float(np.mean(forecast)), 2),
                'max_forecast': round(float(np.max(forecast)), 2),
                'min_forecast': round(float(np.min(forecast)), 2)
            }
            
            return render_template("index.html", **template_data)
            
        except Exception as e:
            return f"Error generating forecast: {str(e)}", 500
            
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)