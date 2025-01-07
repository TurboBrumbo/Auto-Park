import pandas as pd
import numpy as np
import random
from faker import Faker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
    df['Is_Warranty_Expired'] = (df['Days_To_Warranty_Expiry'] < 0).astype(int)

    df = df[df['Vehicle_Model'] == 'Car']

    df = df.drop(columns=['Vehicle_Model', 'Engine_Size', 'Odometer_Reading', 'Owner_Type', 'Last_Service_Date', 'Warranty_Expiry_Date', 'Insurance_Premium'])

    df = df.sample(n=100, random_state=42).reset_index(drop=True)
    df.insert(0, 'Car_ID', range(1, 101))

    return df

def generate_trips_dataset(cars_df):
    faker = Faker('en_US')

    drivers = {i: f"{faker.first_name()} {faker.first_name()[0]}.{faker.first_name()[0]}." for i in range(1, 11)}

    trips = []
    for driver_id, driver_name in drivers.items():
        car_ids = np.random.choice(cars_df['Car_ID'], size=100, replace=True)
        for car_id in car_ids:
            trip_length = round(random.uniform(1.0, 30.0), 1)
            rating = random.choices([1, 2, 3, 4, 5], weights=[5, 5, 10, 40, 40])[0]
            trip_date = faker.date_between(start_date=pd.Timestamp("2024-01-01"), end_date=pd.Timestamp("2024-12-31"))
            trips.append([driver_id, driver_name, car_id, trip_length, trip_date, rating])

    trips_df = pd.DataFrame(trips, columns=[
        'Driver_ID', 'Driver_Name', 'Car_ID', 'Trip_Length_km', 'Trip_Date', 'Trip_Rating'
    ])

    return trips_df

def predict_next_service_date(cars_df):
    cars_df['Next_Service_Days'] = np.where(cars_df['Need_Maintenance'] == 0, 0, np.nan)

    train_data = cars_df[cars_df['Need_Maintenance'] == 1]

    features = [
        "Mileage", "Vehicle_Age", "Fuel_Efficiency", "Service_History", "Accident_History",
        "Tire_Condition", "Brake_Condition", "Battery_Status", "Days_Since_Last_Service"
    ]
    X = train_data[features]
    y = train_data['Days_Since_Last_Service']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predicted_days = model.predict(X).astype(int)
    cars_df.loc[cars_df['Need_Maintenance'] == 1, 'Next_Service_Days'] = predicted_days

    return cars_df

def generate_rating_dataset(trips_df):
    rating_data = []
    driver_ids = trips_df['Driver_ID'].unique()

    for driver_id in driver_ids:
        driver_trips = trips_df[trips_df['Driver_ID'] == driver_id]
        avg_trip_length = round(driver_trips['Trip_Length_km'].mean(), 2)
        experience_years = random.randint(1, 40)
        fines = random.choices([0, random.randint(1, 50)], weights=[6, 4])[0]
        rating_data.append([driver_id, avg_trip_length, experience_years, fines])

    rating_df = pd.DataFrame(rating_data, columns=[
        'Driver_ID', 'Avg_Trip_Length_km', 'Experience_Years', 'Total_Fines'
    ])

    return rating_df

if __name__ == "__main__":
    cars_filepath = 'vehicle_maintenance_data.csv'  # Обновите путь к файлу, если требуется
    cars_df = load_dataset(cars_filepath)

    cars_df = predict_next_service_date(cars_df)

    columns = [col for col in cars_df.columns if col not in ['Need_Maintenance', 'Next_Service_Days']]
    columns += ['Need_Maintenance', 'Next_Service_Days']
    cars_df = cars_df[columns]

    cars_df.to_csv('cars_dataset.csv', index=False, encoding='utf-8')

    trips_df = generate_trips_dataset(cars_df)

    trips_df.to_csv('trips_dataset.csv', index=False, encoding='utf-8')

    rating_df = generate_rating_dataset(trips_df)

    rating_df.to_csv('rating_dataset.csv', index=False, encoding='utf-8')

    print("Датасеты успешно созданы и сохранены: cars_dataset.csv, trips_dataset.csv и rating_dataset.csv.")

