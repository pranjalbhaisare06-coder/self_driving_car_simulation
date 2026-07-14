import sqlite3

DATABASE = "database/simulation.db"

def create_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        episode INTEGER,

        reward REAL,

        speed REAL,

        collision INTEGER,

        lane TEXT
    )
    """)

    conn.commit()

    conn.close()

if __name__ == "__main__":
    create_database()
    print("Database Created Successfully")