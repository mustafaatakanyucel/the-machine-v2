import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import holidays
import time

class DataPipeline:
    """
    An enterprise-level data pipeline for The Machine v2.0.
    This class handles fetching, merging, and engineering features from crime, weather, and holiday data.
    """
    def __init__(self, crime_limit=100000, n_clusters=50):
        """
        Initializes the DataPipeline.
        Args:
            crime_limit (int): The number of crime records to fetch from the API.
            n_clusters (int): The number of spatial clusters (Risk Zones) to create.
        """
        self.crime_limit = crime_limit
        self.n_clusters = n_clusters
        self.us_holidays = holidays.US()
        print("DataPipeline for The Machine v2.0 initialized.")

    def _fetch_crime_data(self):
        """Fetches crime data from the City of Chicago API."""
        print(f"Fetching {self.crime_limit} crime records...")
        url = f"https://data.cityofchicago.org/resource/ijzp-q8t2.json?$limit={self.crime_limit}"
        df = pd.read_json(url)
        
        # Basic cleaning
        df['Date'] = pd.to_datetime(df['date']).dt.date
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        df.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude'}, inplace=True)
        print("Crime data fetched and cleaned.")
        return df

    def _fetch_weather_data(self, start_date, end_date):
        """Fetches historical weather data from the Open-Meteo API."""
        print(f"Fetching weather data from {start_date} to {end_date}...")
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            "latitude=41.8781&longitude=-87.6298&"
            f"start_date={start_date}&end_date={end_date}&"
            "daily=temperature_2m_mean,precipitation_sum&timezone=America/Chicago"
        )
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("Failed to fetch weather data. Please check the API.")
            
        weather_data = response.json()['daily']
        df_weather = pd.DataFrame(weather_data)
        df_weather['Date'] = pd.to_datetime(df_weather['time']).dt.date
        df_weather.rename(columns={
            'temperature_2m_mean': 'Temperature',
            'precipitation_sum': 'Precipitation'
        }, inplace=True)
        print("Weather data fetched.")
        return df_weather[['Date', 'Temperature', 'Precipitation']]

    def _add_holiday_feature(self, df):
        """Adds a boolean feature indicating if the date is a US holiday."""
        print("Adding holiday feature...")
        df['Is_Holiday'] = df['Date'].apply(lambda x: x in self.us_holidays)
        return df

    def _create_spatial_features(self, df):
        """Creates 'Risk_Zone' clusters and calculates 'Spatial_Lag'."""
        print("Engineering spatial features: Risk Zones and Spatial Lag...")
        coords = df[['Latitude', 'Longitude']]
        
        # 1. KMeans Clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        df['Risk_Zone'] = kmeans.fit_predict(coords)
        
        # Save centroids for the app
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Latitude', 'Longitude'])
        centroids['Risk_Zone'] = centroids.index
        

        # 2. Spatial Lag Calculation
        # Calculate crime density per zone
        crime_density = df.groupby('Risk_Zone').size().reset_index(name='crime_count')
        
        # Find nearest neighbors for each cluster centroid
        nn = NearestNeighbors(n_neighbors=5, metric='haversine') # Using 5 nearest neighbors
        nn.fit(np.radians(centroids[['Latitude', 'Longitude']]))
        distances, indices = nn.kneighbors(np.radians(centroids[['Latitude', 'Longitude']]))
        
        # Calculate spatial lag (average crime in neighboring zones)
        spatial_lag_map = {}
        for i in range(self.n_clusters):
            neighbor_zones = indices[i][1:] # Exclude the zone itself
            neighbor_crime_counts = crime_density[crime_density['Risk_Zone'].isin(neighbor_zones)]['crime_count']
            spatial_lag_map[i] = neighbor_crime_counts.mean()
            
        df['Spatial_Lag'] = df['Risk_Zone'].map(spatial_lag_map).fillna(0)

        # Add the mean spatial lag to the centroids file for the app to use
        centroids['Spatial_Lag_Mean'] = centroids['Risk_Zone'].map(spatial_lag_map).fillna(0)
        centroids.to_csv('centroids.csv', index=False)
        print("KMeans centroids saved to 'centroids.csv'.")
        print("Spatial features created.")
        return df

    def run(self):
        """
        Executes the full data pipeline.
        """
        start_time = time.time()
        
        # Step 1: Fetch and merge primary data sources
        df_crime = self._fetch_crime_data()
        min_date, max_date = df_crime['Date'].min(), df_crime['Date'].max()
        
        df_weather = self._fetch_weather_data(min_date, max_date)
        
        df = pd.merge(df_crime, df_weather, on='Date', how='left')
        
        # Step 2: Add temporal features
        df = self._add_holiday_feature(df)
        
        # Step 3: Add spatial features
        df = self._create_spatial_features(df)
        
        # Final processing
        # For simplicity, we'll drop rows where weather data might be missing
        df.dropna(inplace=True)
        
        # Save the processed data
        df.to_csv('processed_crime_data.csv', index=False)
        
        end_time = time.time()
        print("\n--- Data Pipeline Complete ---")
        print(f"Final dataset shape: {df.shape}")
        print(f"Processed data saved to 'processed_crime_data.csv'.")
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")
        
        return df

if __name__ == '__main__':
    pipeline = DataPipeline(crime_limit=100000, n_clusters=50)
    processed_df = pipeline.run()
    print("\n--- Pipeline Output Sample ---")
    print(processed_df.head())
