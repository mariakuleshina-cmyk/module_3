import sqlite3
import random

DB_NAME = "data.db"

def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Создание таблицы traffic
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature INTEGER,
            cars INTEGER,
            light INTEGER,
            traffic_level INTEGER
        )
    """)

    # Заполнение 1000 записей случайными данными
    for i in range(1000):
        temperature = random.randint(-30, 40)
        cars = random.randint(0, 80)
        light = random.randint(0, 1)

        # Коэффициент погоды
        weather_factor = 1.5 if (temperature < -10 or temperature > 35) else 1.0
        # Коэффициент освещённости7
        light_factor = 1.2 if light == 1 else 0.7

        # Эффективное количество машин с учётом коэффициентов
        effective_cars = cars * weather_factor * light_factor

        # Определение уровня трафика
        if effective_cars < 20:
            traffic_level = 0
        elif effective_cars < 50:
            traffic_level = 1
        else:
            traffic_level = 2

        cursor.execute(
            "INSERT INTO traffic (temperature, cars, light, traffic_level) VALUES (?, ?, ?, ?)",
            (temperature, cars, light, traffic_level)
        )

    conn.commit()
    conn.close()
    print("База данных создана. Добавлено 1000 записей.")


if __name__ == "__main__":
    create_database()