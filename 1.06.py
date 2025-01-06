import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash.dash_table
import plotly.express as px

# 1. Загрузка данных
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

    # Фильтрация только автомобилей
    df = df[df['Vehicle_Model'] == 'Car']

    # Выбор 100 случайных записей и добавление столбца ID
    df = df.sample(n=100, random_state=42).reset_index(drop=True)
    df.insert(0, 'Car_ID', range(1, 101))

    return df

# Изначальная загрузка данных
data = load_dataset('C:\\Users\\turbo\\PycharmProjects\\AutoPark\\vehicle_maintenance_data.csv')

# 2. Подготовка признаков и целевой переменной
potential_features = [
    "Mileage",
    "Vehicle_Age",
    "Fuel_Efficiency",
    "Service_History",
    "Accident_History",
    "Tire_Condition",
    "Brake_Condition",
    "Battery_Status",
    "Days_Since_Last_Service",
    "Days_To_Warranty_Expiry",
    "Is_Warranty_Expired",
]
features = [feature for feature in potential_features if feature in data.columns]
X = data[features]
y = data["Need_Maintenance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy:.2f}")
print(classification_report(y_test, y_pred, zero_division=0))

# 5. Создание дашборда
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Дашборд технического обслуживания автопарка"),
    dcc.Graph(id="maintenance-graph"),
    html.Div([
        html.H3("Статистика автомобилей"),
        dash.dash_table.DataTable(
            id="data-table",
            columns=[
                {"name": "ID автомобиля", "id": "Car_ID"},
                {"name": "Пробег", "id": "Mileage"},
                {"name": "Возраст автомобиля", "id": "Vehicle_Age"},
                {"name": "Экономия топлива", "id": "Fuel_Efficiency"},
                {"name": "История обслуживания", "id": "Service_History"},
                {"name": "История аварий", "id": "Accident_History"},
                {"name": "Состояние шин", "id": "Tire_Condition"},
                {"name": "Состояние тормозов", "id": "Brake_Condition"},
                {"name": "Состояние батареи", "id": "Battery_Status"},
                {"name": "Дни с последнего обслуживания", "id": "Days_Since_Last_Service"},
                {"name": "Дни до окончания гарантии", "id": "Days_To_Warranty_Expiry"},
                {"name": "Гарантия истекла", "id": "Is_Warranty_Expired"},
                {"name": "Необходимость обслуживания", "id": "Need_Maintenance"},
                {"name": "Предсказанное обслуживание", "id": "Predicted_Maintenance"},
            ],
            page_size=10,
            style_table={"overflowX": "auto"},
        ),
    ]),
    dcc.Interval(
        id="interval-component",
        interval=60*1000,  # 1 минута в миллисекундах
        n_intervals=0,
    ),
])

@app.callback(
    [Output("maintenance-graph", "figure"),
     Output("data-table", "data")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard(n_intervals):
    data = load_dataset('C:\\Users\\turbo\\PycharmProjects\\AutoPark\\vehicle_maintenance_data.csv')
    data["Predicted_Maintenance"] = model.predict(data[features])

    # Добавляем колонку для ошибок
    data["Error"] = data["Need_Maintenance"] != data["Predicted_Maintenance"]

    # Обновляем график для отображения ошибок
    fig = px.scatter(
        data,
        x="Mileage",
        y="Car_ID",
        color=data["Error"].map({True: "Ошибочно", False: "Верно"}),
        symbol="Need_Maintenance",
        title="Потребность в обслуживании в зависимости от пробега",
        labels={
            "color": "Верность предсказания",
            "Need_Maintenance": "Необходимость обслуживания (0 - не нужно, 1 - нужно)",
            "Predicted_Maintenance": "Предсказанное обслуживание",
            "Mileage": "Пробег",
            "Car_ID": "ID автомобиля"
        }
    )
    return fig, data.to_dict("records")

if __name__ == "__main__":
    app.run_server(debug=True)