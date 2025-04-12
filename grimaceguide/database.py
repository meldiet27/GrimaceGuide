#Database manager for App

import sqlite3
import datetime
import io
from PIL import Image as PILImage
import os
from pathlib import Path

from .config import DATABASE_PATH

#Initializes the DatabaseManager class
class DatabaseManager:
    def __init__(self, db_path=DATABASE_PATH):
        #Stores the database path from config or use provided path
        self.db_path = db_path
        self.conn = None
        #Initializes database tables on creation
        self.init_db()
    
    def init_db(self):
        """Initialize the database and create tables if they don't exist"""
        try:
            #Ensures the directory exists - create parent directories if needed
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            #Connects to SQLite database (creates file if it doesn't exist)
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            #Creates images table for file information of both original and processed images
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                original_path TEXT NOT NULL,
                processed_path TEXT,
                thumbnail_data BLOB
            )
            ''')
            
            #Creates scores table - stores both individual element and total FGS scores for each processed image
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                ears_score INTEGER,
                eyes_score INTEGER,
                muzzle_score INTEGER,
                whiskers_score INTEGER,
                head_score INTEGER,
                total_score INTEGER,
                processing_method TEXT,
                processing_date TEXT,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
            ''')
            
            #Creates landmarks table - stores facial landmarks detected by AI
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS landmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                animal_id TEXT,
                landmark_type TEXT,
                x_coord REAL,
                y_coord REAL,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
            ''')
            
            self.conn.commit()
            print("Database initialized successfully")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    #Function for storing file information into database
    def store_image(self, filename, original_path, processed_path=None):
        try:
            cursor = self.conn.cursor()
            #Tracks date and time when image was uploaded for history
            upload_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            #Generates thumbnail for history view, BLOB for faster retrieval of info
            thumbnail_data = None
            try:
                with open(original_path, "rb") as img_file:
                    img = PILImage.open(img_file)
                    img.thumbnail((100, 100))  #Resizes the image to 100x100 thumbnail
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    thumbnail_data = buffer.getvalue()
            except Exception as e:
                print(f"Error creating thumbnail: {e}")
            
            #Inserts image data into database and returns the generated ID
            cursor.execute(
                "INSERT INTO images (filename, upload_date, original_path, processed_path, thumbnail_data) VALUES (?, ?, ?, ?, ?)",
                (filename, upload_date, original_path, processed_path, thumbnail_data)
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error storing image: {e}")
            return None

    #Function that updates the path of the processed image
    def update_processed_image(self, image_id, processed_path):
        try:
            cursor = self.conn.cursor()
            #Updates path with processed image after processing with API
            cursor.execute(
                "UPDATE images SET processed_path = ? WHERE id = ?",
                (processed_path, image_id)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating processed image: {e}")
            return False

    #Stores FGS scores
    def store_scores(self, image_id, scores, processing_method="API"):
        try:
            cursor = self.conn.cursor()
            #Tracks date and time from  when scores were calculated for analysis
            processing_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            #Stores individual category scores and the total score
            #Tracks whether scores came from API or local model
            cursor.execute(
                """INSERT INTO scores 
                   (image_id, ears_score, eyes_score, muzzle_score, whiskers_score, head_score, 
                    total_score, processing_method, processing_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (image_id, scores['ears'], scores['eyes'], scores['muzzle'], 
                 scores['whiskers'], scores['head'], scores['total_score'],
                 processing_method, processing_date)
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error storing scores: {e}")
            return None

    #Stores landmark data from API processing of image
    def store_landmarks(self, image_id, landmarks_result):
        try:
            cursor = self.conn.cursor()
            
            #Parses the complex landmarks structure from API into (x,y) coordinates
            for i, animal_data in enumerate(landmarks_result):
                for animal, details in animal_data.items():
                    #Creates a unique ID for each animal in case of multiple detections
                    animal_id = f"{animal}_{i}"
                    landmarks = details.get('landmarks', [])
                    
                    for j, landmark in enumerate(landmarks):
                        x = landmark.get('x', 0)
                        y = landmark.get('y', 0)
                        landmark_type = landmark.get('type', f"point_{j}")
                        
                        cursor.execute(
                            "INSERT INTO landmarks (image_id, animal_id, landmark_type, x_coord, y_coord) VALUES (?, ?, ?, ?, ?)",
                            (image_id, animal_id, landmark_type, x, y)
                        )
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error storing landmarks: {e}")
            return False

    #Fetches history of most recent images in database
    def get_image_history(self, limit=10):
        try:
            cursor = self.conn.cursor()
            #Joins images with their scores for use in history view of previous analyses
            cursor.execute(
                """SELECT i.id, i.filename, i.upload_date, i.original_path, i.processed_path, 
                   s.ears_score, s.eyes_score, s.muzzle_score, s.whiskers_score, s.head_score, s.total_score
                   FROM images i
                   LEFT JOIN scores s ON i.id = s.image_id
                   ORDER BY i.upload_date DESC
                   LIMIT ?""", (limit,)
            )
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting image history: {e}")
            return []

    #Closes the app connection to the database
    def close(self):
        if self.conn:
            self.conn.close()
