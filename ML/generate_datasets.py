import pandas as pd
import numpy as np
import random
from faker import Faker
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def generate_russian_license_plate():
    letters = "ABEKMHOPCTYX"
    region = random.randint(1, 199)
    plate = f"{random.choice(letters)}{random.randint(100, 999)}{random.choice(letters)}{random.choice(letters)}{region:02d}"
    return plate

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df['Last_Service_Date'] = pd.to_datetime(df['Last_Service_Date'], errors='coerce')
    df['Warranty_Expiry_Date'] = pd.to_datetime(df['Warranty_Expiry_Date'], errors='coerce')

    df['Tire_Condition'] = df['Tire_Condition'].map({"New": 2, "Good": 1, "Worn Out": 0})
    df['Brake_Condition'] = df['Brake_Condition'].map({"New": 2, "Good": 1, "Worn Out": 0})
    df['Battery_Status'] = df['Battery_Status'].map({"New": 2, "Good": 1, "Weak": 0})

    df['Service_History'] = df['Service_History'].fillna(0)
    df['Accident_History'] = df['Accident_History'].fillna(0)

    df['Days_Since_Last_Service'] = (pd.Timestamp.now() - df['Last_Service_Date']).dt.days
    df['Days_To_Warranty_Expiry'] = (df['Warranty_Expiry_Date'] - pd.Timestamp.now()).dt.days

    features = [
        "Car_ID", "License_Plate", "Mileage", "Vehicle_Age", "Fuel_Efficiency", "Service_History", "Accident_History",
        "Reported_Issues", "Tire_Condition", "Brake_Condition", "Battery_Status", "Days_Since_Last_Service"
    ]

    df.insert(0, 'Car_ID', range(1, len(df) + 1))
    df['License_Plate'] = [generate_russian_license_plate() for _ in range(len(df))]
    df = df[features]

    return df

def generate_trips_dataset(cars_df):
    faker = Faker('en_US')

    drivers = {i: f"{faker.first_name()} {faker.last_name()}" for i in range(1, 11)}

    trips = []
    for driver_id, driver_name in drivers.items():
        car_ids = np.random.choice(range(1, 101), size=100, replace=True)  # ID машин от 1 до 100
        for car_id in car_ids:
            trip_length = round(random.uniform(1.0, 30.0), 1)
            rating = random.choices([1, 2, 3, 4, 5], weights=[5, 5, 10, 40, 40])[0]
            trip_date = faker.date_between(start_date=pd.Timestamp("2024-01-01"), end_date=pd.Timestamp("2024-12-31"))
            license_plate = cars_df.loc[cars_df['Car_ID'] == car_id, 'License_Plate'].values[0]
            trips.append([driver_id, driver_name, car_id, license_plate, trip_length, trip_date, rating])

    trips_df = pd.DataFrame(trips, columns=[
        'Driver_ID', 'Driver_Name', 'Car_ID', 'License_Plate', 'Trip_Length_km', 'Trip_Date', 'Trip_Rating'
    ])

    return trips_df

def generate_rating_dataset(trips_df):
    rating_data = []
    driver_ids = trips_df['Driver_ID'].unique()

    for driver_id in driver_ids:
        driver_trips = trips_df[trips_df['Driver_ID'] == driver_id]
        avg_trip_length = round(driver_trips['Trip_Length_km'].mean(), 2)
        experience_years = random.randint(1, 40)
        fines = random.choices([0, random.randint(1, 50)], weights=[6, 4])[0]
        driver_name = driver_trips.iloc[0]['Driver_Name']
        rating_data.append([driver_id, driver_name, avg_trip_length, experience_years, fines])

    rating_df = pd.DataFrame(rating_data, columns=[
        'Driver_ID', 'Driver_Name', 'Avg_Trip_Length_km', 'Experience_Years', 'Total_Fines'
    ])

    return rating_df

def train_model(cars_df):
    features = [
        "Mileage", "Vehicle_Age", "Fuel_Efficiency", "Service_History", "Accident_History",
        "Reported_Issues", "Tire_Condition", "Brake_Condition", "Battery_Status", "Days_Since_Last_Service"
    ]

    X = cars_df[features]
    y = cars_df['Days_Since_Last_Service']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, 'service_date_model.joblib')

    return model

if __name__ == "__main__":
    os.makedirs('datasets', exist_ok=True)

    cars_filepath = 'vehicle_maintenance_data.csv'
    full_cars_df = load_dataset(cars_filepath)

    train_model(full_cars_df)

    sample_cars_df = full_cars_df.sample(n=100, random_state=42).reset_index(drop=True)
    sample_cars_df['Car_ID'] = range(1, 101)

    sample_cars_df.to_csv('datasets/cars_dataset.csv', index=False, encoding='utf-8')

    trips_df = generate_trips_dataset(sample_cars_df)
    trips_df.to_csv('datasets/trips_dataset.csv', index=False, encoding='utf-8')

    rating_df = generate_rating_dataset(trips_df)
    rating_df.to_csv('datasets/rating_dataset.csv', index=False, encoding='utf-8')

    print("Датасеты успешно созданы и сохранены в папке datasets.")
