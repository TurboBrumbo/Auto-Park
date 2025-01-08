import pandas as pd
import numpy as np
import random
from faker import Faker
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def generate_russian_license_plate():
    letters = "ABEKMHOPCTYX"
    plate = f"{random.choice(letters)}{random.randint(100, 999)}{random.choice(letters)}{random.choice(letters)}"
    return plate

def load_full_dataset(filepath):
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

    df.insert(0, 'id', [generate_russian_license_plate() for _ in range(len(df))])

    features = [
        "id", "Mileage", "Vehicle_Age", "Reported_Issues", "Fuel_Efficiency", "Service_History",
        "Accident_History", "Tire_Condition", "Brake_Condition", "Battery_Status",
        "Days_Since_Last_Service", "Days_To_Warranty_Expiry"
    ]

    df = df[features]

    df.columns = [
        "id", "mileage", "vehicle_age", "reported_issues", "fuel_efficiency", "service_history",
        "accident_history", "tire_condition", "brake_condition", "battery_status",
        "days_since_last_service", "days_to_warranty_expire"
    ]
    df.set_index('id', inplace=True)

    return df

def sample_cars_dataset(full_df, sample_size=100):
    return full_df.sample(n=sample_size, random_state=42)

def generate_trips_dataset(cars_df):
    faker = Faker('en_US')

    drivers = {i: f"{faker.first_name()} {faker.last_name()}" for i in range(1, 11)}

    trips = []
    for driver_id in drivers.keys():
        ids = np.random.choice(cars_df.index, size=100, replace=True)
        for id in ids:
            trip_length = round(random.uniform(1.0, 30.0), 1)
            rating = random.choices([1, 2, 3, 4, 5], weights=[5, 5, 10, 40, 40])[0]
            trip_date = faker.date_between(start_date=pd.Timestamp("2024-01-01"), end_date=pd.Timestamp("2024-12-31"))
            trips.append([id, driver_id, trip_length, trip_date, rating])

    trips_df = pd.DataFrame(trips, columns=[
        'car_id', 'driver_id', 'length', 'date', 'rating'
    ])

    return trips_df

def generate_rating_dataset(trips_df):
    faker = Faker('en_US')

    rating_data = []
    driver_ids = trips_df['driver_id'].unique()

    driver_names = {
        driver_id: f"{faker.first_name()} {faker.last_name()[0]}."
        for driver_id in driver_ids
    }

    for driver_id in driver_ids:
        experience_years = random.randint(1, 40)
        driver_age = random.randint(20, 55)
        initials = driver_names[driver_id]
        rating_data.append([initials, driver_age, experience_years])

    rating_df = pd.DataFrame(rating_data, columns=[
        'initials', 'age', 'experience'
    ])

    return rating_df

def train_service_model(cars_df):
    features = [
        "mileage", "vehicle_age", "reported_issues", "fuel_efficiency", "service_history",
        "accident_history", "tire_condition", "brake_condition", "battery_status",
        "days_since_last_service", "days_to_warranty_expire"
    ]
    target = "days_since_last_service"

    X = cars_df[features]
    y = cars_df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, 'service_date.joblib')
    print("Модель успешно обучена и сохранена.")

    return model

if __name__ == "__main__":
    os.makedirs('datasets', exist_ok=True)

    cars_filepath = 'vehicle_maintenance_data.csv'

    full_cars_df = load_full_dataset(cars_filepath)

    train_service_model(full_cars_df)

    sampled_cars_df = sample_cars_dataset(full_cars_df)

    sampled_cars_df.to_csv('datasets/cars.csv', encoding='utf-8')

    trips_df = generate_trips_dataset(sampled_cars_df)
    trips_df.to_csv('datasets/trips.csv', index=False, encoding='utf-8')

    rating_df = generate_rating_dataset(trips_df)
    rating_df.to_csv('datasets/drivers.csv', index=False, encoding='utf-8')

    print("Датасеты успешно созданы и сохранены в папке datasets.")



