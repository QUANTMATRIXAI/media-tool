"""
Database module for storing user files and progress
Supports EDA, Modeling, and Optimizer modules
"""

import sqlite3
import json
import pickle
import base64
from pathlib import Path
from datetime import datetime
import pandas as pd


class UserDatabase:
    """Manages user data, uploaded files, and progress across all modules"""
    
    def __init__(self, db_path="user_data.db"):
        self.db_path = db_path
        print(f"Initializing database at: {self.db_path}")
        self.init_database()
        print(f"Database initialized successfully")
    
    def _migrate_database(self, cursor):
        """Migrate existing database to new schema with project support"""
        try:
            # Check if user_files table exists and has old schema
            cursor.execute("PRAGMA table_info(user_files)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if columns and 'project_name' not in columns:
                print("Migrating database to project-based schema...")
                
                # Rename old table
                cursor.execute("ALTER TABLE user_files RENAME TO user_files_old")
                
                # Create new table with project_name
                cursor.execute('''
                    CREATE TABLE user_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        module TEXT NOT NULL,
                        project_name TEXT NOT NULL,
                        file_key TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        file_data BLOB NOT NULL,
                        file_type TEXT,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(username, module, project_name, file_key)
                    )
                ''')
                
                # Migrate data with default project name
                cursor.execute('''
                    INSERT INTO user_files (username, module, project_name, file_key, file_name, file_data, file_type, uploaded_at)
                    SELECT username, module, 'Default Project', file_key, file_name, file_data, file_type, uploaded_at
                    FROM user_files_old
                ''')
                
                # Drop old table
                cursor.execute("DROP TABLE user_files_old")
                
                print("Migration completed for user_files")
            
            # Check and migrate user_progress table
            cursor.execute("PRAGMA table_info(user_progress)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if columns and 'project_name' not in columns:
                print("Migrating user_progress table...")
                
                cursor.execute("ALTER TABLE user_progress RENAME TO user_progress_old")
                
                cursor.execute('''
                    CREATE TABLE user_progress (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        module TEXT NOT NULL,
                        project_name TEXT NOT NULL,
                        progress_data TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(username, module, project_name)
                    )
                ''')
                
                cursor.execute('''
                    INSERT INTO user_progress (username, module, project_name, progress_data, last_updated)
                    SELECT username, module, 'Default Project', progress_data, last_updated
                    FROM user_progress_old
                ''')
                
                cursor.execute("DROP TABLE user_progress_old")
                
                print("Migration completed for user_progress")
                
        except Exception as e:
            print(f"Migration error (might be normal if tables don't exist yet): {e}")
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if we need to migrate existing tables
        self._migrate_database(cursor)
        
        # Users table (already exists from login system)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                avatar TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Projects table - stores user projects
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                module TEXT NOT NULL,
                project_name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(username, module, project_name)
            )
        ''')
        
        # User files table - stores uploaded files per user per module per project
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                module TEXT NOT NULL,
                project_name TEXT NOT NULL,
                file_key TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_data BLOB NOT NULL,
                file_type TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(username, module, project_name, file_key)
            )
        ''')
        
        # User progress table - stores settings and state per module per project
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                module TEXT NOT NULL,
                project_name TEXT NOT NULL,
                progress_data TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(username, module, project_name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_project(self, username, module, project_name, description=""):
        """Create a new project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO projects (username, module, project_name, description, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, module, project_name, description, datetime.now(), datetime.now()))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False  # Project already exists
        except Exception as e:
            print(f"Error creating project: {e}")
            return False
    
    def get_projects(self, username, module):
        """Get all projects for a user in a module"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT project_name, description, created_at, last_accessed
                FROM projects
                WHERE username = ? AND module = ?
                ORDER BY last_accessed DESC
            ''', (username, module))
            
            results = cursor.fetchall()
            conn.close()
            
            projects = []
            for row in results:
                projects.append({
                    'name': row[0],
                    'description': row[1],
                    'created_at': row[2],
                    'last_accessed': row[3]
                })
            return projects
        except Exception as e:
            print(f"Error getting projects: {e}")
            return []
    
    def update_project_access(self, username, module, project_name):
        """Update last accessed time for a project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE projects
                SET last_accessed = ?
                WHERE username = ? AND module = ? AND project_name = ?
            ''', (datetime.now(), username, module, project_name))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating project access: {e}")
            return False
    
    def delete_project(self, username, module, project_name):
        """Delete a project and all its files"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete project files
            cursor.execute('''
                DELETE FROM user_files
                WHERE username = ? AND module = ? AND project_name = ?
            ''', (username, module, project_name))
            
            # Delete project progress
            cursor.execute('''
                DELETE FROM user_progress
                WHERE username = ? AND module = ? AND project_name = ?
            ''', (username, module, project_name))
            
            # Delete project
            cursor.execute('''
                DELETE FROM projects
                WHERE username = ? AND module = ? AND project_name = ?
            ''', (username, module, project_name))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deleting project: {e}")
            return False
    
    def save_file(self, username, module, project_name, file_key, uploaded_file):
        """Save an uploaded file for a user in a specific module and project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Reset file pointer to beginning and read file data
            uploaded_file.seek(0)
            file_data = uploaded_file.read()
            file_name = uploaded_file.name
            file_type = getattr(uploaded_file, 'type', 'application/octet-stream')
            
            # Debug info
            print(f"Saving file: {file_name}, size: {len(file_data)} bytes")
            print(f"Username: {username}, Module: {module}, Project: {project_name}, Key: {file_key}")
            
            # Reset file pointer again for potential reuse
            uploaded_file.seek(0)
            
            # Insert or replace file
            cursor.execute('''
                INSERT OR REPLACE INTO user_files 
                (username, module, project_name, file_key, file_name, file_data, file_type, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (username, module, project_name, file_key, file_name, file_data, file_type, datetime.now()))
            
            conn.commit()
            conn.close()
            print(f"File saved successfully!")
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            import traceback
            traceback.print_exc()
            import streamlit as st
            st.error(f"Database error: {str(e)}")
            return False
    
    def get_file(self, username, module, project_name, file_key):
        """Retrieve a saved file for a user in a specific module and project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_name, file_data, file_type, uploaded_at
                FROM user_files
                WHERE username = ? AND module = ? AND project_name = ? AND file_key = ?
            ''', (username, module, project_name, file_key))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'file_name': result[0],
                    'file_data': result[1],
                    'file_type': result[2],
                    'uploaded_at': result[3]
                }
            return None
        except Exception as e:
            print(f"Error retrieving file: {e}")
            return None
    
    def get_all_files(self, username, module, project_name):
        """Get all files for a user in a specific module and project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_key, file_name, file_type, uploaded_at
                FROM user_files
                WHERE username = ? AND module = ? AND project_name = ?
                ORDER BY uploaded_at DESC
            ''', (username, module, project_name))
            
            results = cursor.fetchall()
            conn.close()
            
            files = {}
            for row in results:
                files[row[0]] = {
                    'file_name': row[1],
                    'file_type': row[2],
                    'uploaded_at': row[3]
                }
            return files
        except Exception as e:
            print(f"Error retrieving files: {e}")
            return {}
    
    def delete_file(self, username, module, project_name, file_key):
        """Delete a saved file"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM user_files
                WHERE username = ? AND module = ? AND project_name = ? AND file_key = ?
            ''', (username, module, project_name, file_key))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    def save_progress(self, username, module, project_name, progress_data):
        """Save user progress/settings for a module and project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert progress_data dict to JSON
            progress_json = json.dumps(progress_data)
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_progress
                (username, module, project_name, progress_data, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, module, project_name, progress_json, datetime.now()))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving progress: {e}")
            return False
    
    def get_progress(self, username, module, project_name):
        """Retrieve user progress/settings for a module and project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT progress_data, last_updated
                FROM user_progress
                WHERE username = ? AND module = ? AND project_name = ?
            ''', (username, module, project_name))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'data': json.loads(result[0]),
                    'last_updated': result[1]
                }
            return None
        except Exception as e:
            print(f"Error retrieving progress: {e}")
            return None
    
    def clear_user_data(self, username, module=None, project_name=None):
        """Clear all data for a user (optionally for a specific module/project)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if project_name and module:
                cursor.execute('DELETE FROM user_files WHERE username = ? AND module = ? AND project_name = ?', (username, module, project_name))
                cursor.execute('DELETE FROM user_progress WHERE username = ? AND module = ? AND project_name = ?', (username, module, project_name))
                cursor.execute('DELETE FROM projects WHERE username = ? AND module = ? AND project_name = ?', (username, module, project_name))
            elif module:
                cursor.execute('DELETE FROM user_files WHERE username = ? AND module = ?', (username, module))
                cursor.execute('DELETE FROM user_progress WHERE username = ? AND module = ?', (username, module))
                cursor.execute('DELETE FROM projects WHERE username = ? AND module = ?', (username, module))
            else:
                cursor.execute('DELETE FROM user_files WHERE username = ?', (username,))
                cursor.execute('DELETE FROM user_progress WHERE username = ?', (username,))
                cursor.execute('DELETE FROM projects WHERE username = ?', (username,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error clearing user data: {e}")
            return False


# Global database instance
db = UserDatabase()
