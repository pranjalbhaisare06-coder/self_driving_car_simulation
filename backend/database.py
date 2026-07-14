import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "simulation.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS simulation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode INTEGER,
            reward REAL,
            speed REAL,
            collision INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_log(episode, reward, speed, collision):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO simulation_logs
        (episode, reward, speed, collision)
        VALUES (?, ?, ?, ?)
    """, (episode, reward, speed, collision))

    conn.commit()
    conn.close()


def get_logs():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM simulation_logs
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]