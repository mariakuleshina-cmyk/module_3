import sqlite3
from datetime import datetime

import numpy as np
from PyQt5 import QtWidgets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

DB = "data.db"
MODEL = "traffic_model.keras"
LOG = "training_log.db"


# ---------- подготовка БД ----------
def prepare_data_db():
    """Таблица traffic + столбец traffic_level, заполненный формулой."""
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS traffic (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        temperature INTEGER, cars INTEGER, light INTEGER, traffic_level INTEGER)""")

    cols = [r[1] for r in cur.execute("PRAGMA table_info(traffic)")]
    if "traffic_level" not in cols:
        cur.execute("ALTER TABLE traffic ADD COLUMN traffic_level INTEGER")

    # заполняем traffic_level, где пусто (формула из ЛР6)
    for row_id, cars, light in cur.execute("SELECT id, cars, light FROM traffic").fetchall():
        if cars is None:
            continue
        score = cars + (10 if not light else 0)
        level = 0 if score < 25 else (1 if score < 55 else 2)
        cur.execute("UPDATE traffic SET traffic_level=? WHERE id=?", (level, row_id))

    conn.commit()
    conn.close()


def prepare_log_db():
    """Таблица epochs. Если структура старая — пересоздаём."""
    conn = sqlite3.connect(LOG)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(epochs)")
    cols = [r[1] for r in cur.fetchall()]
    if cols and "saved" not in cols:
        cur.execute("DROP TABLE epochs")

    cur.execute("""CREATE TABLE IF NOT EXISTS epochs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT, epoch INTEGER, loss REAL, accuracy REAL, saved TEXT)""")

    conn.commit()
    conn.close()


# ---------- данные / модель ----------
def get_data():
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT temperature, cars, light, traffic_level FROM traffic").fetchall()
    conn.close()
    X = np.array([[r[0], r[1], r[2]] for r in rows], dtype=float)
    Y = np.array([r[3] for r in rows], dtype=int)
    X[:, 0] = (X[:, 0] + 30) / 70
    X[:, 1] = X[:, 1] / 80
    return X, to_categorical(Y, num_classes=3)


def make_model():
    m = Sequential([
        Dense(16, activation='relu', input_shape=(3,)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')])
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


# ---------- окно ----------
class TrainingWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обучение")
        self.resize(800, 500)

        self.model = None
        self.epoch = 0
        self.X = self.Y = None

        # СНАЧАЛА готовим БД, ПОТОМ строим UI и читаем историю
        prepare_data_db()
        prepare_log_db()

        # кнопки
        self.b_train = QtWidgets.QPushButton("Обучение (1 эпоха)")
        self.b_loss = QtWidgets.QPushButton("Потери")
        self.b_log = QtWidgets.QPushButton("Логи")

        top = QtWidgets.QHBoxLayout()
        for b in (self.b_train, self.b_loss, self.b_log):
            b.setMinimumHeight(36)
            top.addWidget(b)

        # таблицы
        self.t_loss = QtWidgets.QTableWidget(0, 3)
        self.t_loss.setHorizontalHeaderLabels(["Эпоха", "Loss", "Accuracy"])
        self.t_loss.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        self.t_log = QtWidgets.QTableWidget(0, 4)
        self.t_log.setHorizontalHeaderLabels(["Время", "Эпоха", "Сохранение", "Loss / Acc"])
        self.t_log.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.t_loss)
        self.stack.addWidget(self.t_log)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.stack)

        self.b_train.clicked.connect(self.train)
        self.b_loss.clicked.connect(lambda: self.stack.setCurrentWidget(self.t_loss))
        self.b_log.clicked.connect(lambda: self.stack.setCurrentWidget(self.t_log))

        self.load_history()

    def train(self):
        try:
            if self.X is None:
                self.X, self.Y = get_data()
                if len(self.X) == 0:
                    raise RuntimeError("В таблице traffic нет данных")

            if self.model is None:
                self.model = make_model()

            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h = self.model.fit(self.X, self.Y, epochs=1, verbose=0, shuffle=True)
            loss, acc = float(h.history['loss'][0]), float(h.history['accuracy'][0])

            self.epoch += 1
            try:
                self.model.save(MODEL)
                saved = "да"
            except Exception as e:
                saved = f"ошибка: {e}"

            # в UI — потери
            r = self.t_loss.rowCount()
            self.t_loss.insertRow(r)
            for i, v in enumerate([self.epoch, f"{loss:.4f}", f"{acc:.4f}"]):
                self.t_loss.setItem(r, i, QtWidgets.QTableWidgetItem(str(v)))

            # в UI — лог
            r = self.t_log.rowCount()
            self.t_log.insertRow(r)
            for i, v in enumerate([t, self.epoch, saved, f"loss={loss:.4f}, acc={acc:.4f}"]):
                self.t_log.setItem(r, i, QtWidgets.QTableWidgetItem(str(v)))

            # в БД логов
            conn = sqlite3.connect(LOG)
            conn.execute(
                "INSERT INTO epochs (start_time, epoch, loss, accuracy, saved) VALUES (?,?,?,?,?)",
                (t, self.epoch, loss, acc, saved))
            conn.commit()
            conn.close()

            self.stack.setCurrentWidget(self.t_loss)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))

    def load_history(self):
        conn = sqlite3.connect(LOG)
        try:
            rows = conn.execute(
                "SELECT start_time, epoch, loss, accuracy, saved FROM epochs ORDER BY id"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        conn.close()

        for t, ep, loss, acc, saved in rows:
            r = self.t_loss.rowCount()
            self.t_loss.insertRow(r)
            for i, v in enumerate([ep, f"{loss:.4f}", f"{acc:.4f}"]):
                self.t_loss.setItem(r, i, QtWidgets.QTableWidgetItem(str(v)))

            r = self.t_log.rowCount()
            self.t_log.insertRow(r)
            for i, v in enumerate([t, ep, saved, f"loss={loss:.4f}, acc={acc:.4f}"]):
                self.t_log.setItem(r, i, QtWidgets.QTableWidgetItem(str(v)))

            self.epoch = ep