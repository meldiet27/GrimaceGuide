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
        """Initialize the database and create tables if they don't exist. Also handles schema updates."""
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
            
            #Creates landmarks table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS landmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                animal_id TEXT,
                landmark_label TEXT, -- Use landmark_label
                x_coord REAL,
                y_coord REAL,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
            ''')

            # --- Schema Update Logic ---
            # Check if landmark_label column exists
            cursor.execute("PRAGMA table_info(landmarks)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'landmark_label' not in columns:
                print("Updating landmarks table schema: Adding landmark_label column...")
                try:
                    cursor.execute("ALTER TABLE landmarks ADD COLUMN landmark_label TEXT")
                    print("Column 'landmark_label' added.")
                except sqlite3.OperationalError as e:
                     # Handle case where column might exist despite initial check (rare)
                     if "duplicate column name" in str(e):
                         print("Column 'landmark_label' already exists.")
                     else:
                         raise e # Re-raise other operational errors

            # Optional: Remove old landmark_type column if it exists and is no longer needed
            if 'landmark_type' in columns:
                 print("Updating landmarks table schema: Removing old landmark_type column...")
                 # Note: SQLite has limited ALTER TABLE support. Dropping columns requires recreating the table.
                 # For simplicity here, we'll leave the old column, but ideally, a migration script would handle this.
                 # cursor.execute("ALTER TABLE landmarks DROP COLUMN landmark_type") # This won't work in SQLite directly
                 print("Note: Old 'landmark_type' column still exists. Manual cleanup or table recreation needed for full removal.")

            # --- End Schema Update Logic ---

            self.conn.commit()
            print("Database initialized/updated successfully")
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

    # Stores landmark data from API processing of image
    def store_landmarks(self, image_id, landmarks_result):
        """Store labeled landmarks in the database."""
        try:
            cursor = self.conn.cursor()
            
            # Define a hypothetical mapping from landmark index/type to label
            # This needs to be adjusted based on the actual API output structure
            # Example: Assuming the API returns landmarks in a specific order
            landmark_labels = [
                'left_eye_outer_corner', 'left_eye_inner_corner', 'right_eye_inner_corner', 'right_eye_outer_corner',
                'nose_tip', 'left_mouth_corner', 'right_mouth_corner',
                'left_ear_base', 'left_ear_tip', 'right_ear_base', 'right_ear_tip',
                'head_top', 'chin_bottom',
                # Add placeholders for extra landmarks based on logs (up to index 47)
                'point_13', 'point_14', 'point_15', 'point_16', 'point_17', 'point_18', 'point_19',
                'point_20', 'point_21', 'point_22', 'point_23', 'point_24', 'point_25', 'point_26',
                'point_27', 'point_28', 'point_29', 'point_30', 'point_31', 'point_32', 'point_33',
                'point_34', 'point_35', 'point_36', 'point_37', 'point_38', 'point_39', 'point_40',
                'point_41', 'point_42', 'point_43', 'point_44', 'point_45', 'point_46', 'point_47'
                # Add more labels if the API returns even more landmarks
            ]

            for i, animal_data in enumerate(landmarks_result):
                for animal, details in animal_data.items():
                    animal_id = f"{animal}_{i}"
                    landmarks = details.get('landmarks', [])
                    
                    for j, landmark in enumerate(landmarks):
                        x = landmark.get('x', 0)
                        y = landmark.get('y', 0)
                        # Assign label based on index, fallback to generic name if index out of bounds
                        label = landmark_labels[j] if j < len(landmark_labels) else f"point_{j}" 
                        
                        cursor.execute(
                            "INSERT INTO landmarks (image_id, animal_id, landmark_label, x_coord, y_coord) VALUES (?, ?, ?, ?, ?)",
                            (image_id, animal_id, label, x, y)
                        )
            
            self.conn.commit()
            print(f"Stored {len(landmarks)} landmarks for image_id {image_id}")
            return True
        except sqlite3.Error as e:
            print(f"Error storing landmarks: {e}")
            return False
        except IndexError:
             print(f"Error: More landmarks received than defined labels. Check API output and landmark_labels list.")
             # Store remaining landmarks with generic labels
             try:
                 cursor = self.conn.cursor() # Ensure cursor is available
                 for k in range(len(landmark_labels), len(landmarks)):
                     landmark = landmarks[k]
                     x = landmark.get('x', 0)
                     y = landmark.get('y', 0)
                     label = f"point_{k}"
                     cursor.execute(
                         "INSERT INTO landmarks (image_id, animal_id, landmark_label, x_coord, y_coord) VALUES (?, ?, ?, ?, ?)",
                         (image_id, animal_id, label, x, y)
                     )
                 self.conn.commit()
                 print(f"Stored remaining {len(landmarks) - len(landmark_labels)} landmarks with generic labels.")
             except sqlite3.Error as e_inner:
                 print(f"Error storing remaining landmarks: {e_inner}")
             return False # Indicate partial success or error

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

# Define the expected landmark labels - MUST MATCH api.py's get_labeled_landmarks
LANDMARK_LABELS = (
    # Left Ear (5 points)
    *[f'left_ear_{i+1}' for i in range(5)],
    # Right Ear (5 points)
    *[f'right_ear_{i+1}' for i in range(5)],
    # Right Eye (4 points)
    *[f'right_eye_{i+1}' for i in range(4)],
    # Right Eye Pupil (4 points)
    *[f'right_pupil_{i+1}' for i in range(4)],
    # Left Eye (4 points)
    *[f'left_eye_{i+1}' for i in range(4)],
    # Left Eye Pupil (4 points)
    *[f'left_pupil_{i+1}' for i in range(4)],
    # Nose (5 points)
    *[f'nose_{i+1}' for i in range(5)],
    # Mouth (6 points)
    *[f'mouth_{i+1}' for i in range(6)],
    # Left Whiskers (5 points)
    *[f'left_whisker_{i+1}' for i in range(5)],
    # Right Whiskers (5 points)
    *[f'right_whisker_{i+1}' for i in range(5)],
    # Chin (1 point)
    'chin_point'
)

def update_schema(conn):
    # Add columns for each landmark label if they don't exist
    cursor.execute("PRAGMA table_info(landmarks)")
    existing_columns = [column[1] for column in cursor.fetchall()]
    
    for label in LANDMARK_LABELS:
        # Sanitize label for column name (replace spaces, special chars)
        # Using simple replacement for this case
        column_name = label.replace(" ", "_").replace("-", "_").lower() + "_x"
        if column_name not in existing_columns:
            print(f"Adding column {column_name} to landmarks table.")
            cursor.execute(f"ALTER TABLE landmarks ADD COLUMN {column_name} REAL")
        column_name = label.replace(" ", "_").replace("-", "_").lower() + "_y"
        if column_name not in existing_columns:
            print(f"Adding column {column_name} to landmarks table.")
            cursor.execute(f"ALTER TABLE landmarks ADD COLUMN {column_name} REAL")

    # Add landmark_label column if it doesn't exist (for storing the original label)
    if 'landmark_label' not in existing_columns:
         print("Adding column landmark_label to landmarks table.")
         cursor.execute("ALTER TABLE landmarks ADD COLUMN landmark_label TEXT")


