from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from pathlib import Path

# Get the correct path relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up to healthcare_kiosk/
DATABASE_DIR = PROJECT_ROOT / "backend" / "database"
DATABASE_PATH = DATABASE_DIR / "healthcare.db"

# Ensure directory exists
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database file path
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

print(f"[INFO] Database URL: {DATABASE_URL}")
print(f"[INFO] Database Path: {DATABASE_PATH}")

# SQLAlchemy engine and session
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base model (shared across models.py)
Base = declarative_base()

# Create all tables (called from main.py during startup)
def init_db():
    from . import models
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")
