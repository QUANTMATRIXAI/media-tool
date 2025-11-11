"""
Comment Manager Module

Handles all comment-related operations including saving, loading, updating,
deleting, and exporting comments with associated comparison data.
"""

import pandas as pd
import json
import uuid
from datetime import datetime, date
from pathlib import Path
import os


COMMENTS_FILE = "comments.csv"


def save_comment(
    comment_text: str,
    period1_start: date,
    period1_end: date,
    period2_start: date,
    period2_end: date,
    comparison_data: pd.DataFrame,
    agg_method: str,
    selected_metrics: list
) -> tuple[bool, str]:
    """
    Save a new comment with associated comparison data.
    
    Args:
        comment_text: The user's comment text
        period1_start: Start date of first period
        period1_end: End date of first period
        period2_start: Start date of second period
        period2_end: End date of second period
        comparison_data: DataFrame containing the comparison table
        agg_method: Aggregation method used (e.g., "Mean", "Sum")
        selected_metrics: List of selected metric names
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Validate comment text
        if not comment_text or not comment_text.strip():
            return False, "Comment text cannot be empty"
        
        # Validate comparison data
        if comparison_data is None or comparison_data.empty:
            return False, "Comparison data is missing or empty"
        
        # Validate dates
        if not all([period1_start, period1_end, period2_start, period2_end]):
            return False, "All period dates must be provided"
        
        # Validate metrics list
        if not selected_metrics or len(selected_metrics) == 0:
            return False, "At least one metric must be selected"
        
        # Generate unique ID and timestamp
        comment_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Serialize comparison DataFrame to JSON string
        try:
            comparison_json = comparison_data.to_json(orient='split', date_format='iso')
        except Exception as e:
            return False, f"Failed to serialize comparison data: {str(e)}"
        
        # Serialize metrics list to JSON string
        try:
            metrics_json = json.dumps(selected_metrics)
        except Exception as e:
            return False, f"Failed to serialize metrics list: {str(e)}"
        
        # Create new comment row
        new_comment = {
            'id': comment_id,
            'timestamp': timestamp,
            'period1_start': period1_start.strftime('%Y-%m-%d'),
            'period1_end': period1_end.strftime('%Y-%m-%d'),
            'period2_start': period2_start.strftime('%Y-%m-%d'),
            'period2_end': period2_end.strftime('%Y-%m-%d'),
            'agg_method': agg_method,
            'metrics': metrics_json,
            'comparison_data': comparison_json,
            'comment_text': comment_text.strip()
        }
        
        # Check if file exists
        file_exists = Path(COMMENTS_FILE).exists()
        
        # Create DataFrame with new comment
        new_df = pd.DataFrame([new_comment])
        
        try:
            if file_exists:
                # Append to existing file
                new_df.to_csv(COMMENTS_FILE, mode='a', header=False, index=False)
            else:
                # Create new file with headers
                new_df.to_csv(COMMENTS_FILE, mode='w', header=True, index=False)
        except PermissionError:
            return False, f"Permission denied: Cannot write to {COMMENTS_FILE}. Please check file permissions."
        except IOError as e:
            return False, f"File I/O error: {str(e)}"
        
        return True, "Comment saved successfully"
        
    except Exception as e:
        return False, f"Unexpected error while saving comment: {str(e)}"


def load_comments() -> tuple[pd.DataFrame, str]:
    """
    Load all saved comments from CSV.
    
    Returns:
        Tuple of (DataFrame, error_message: str)
        DataFrame with columns: id, timestamp, period1_start, period1_end,
        period2_start, period2_end, agg_method, metrics, comparison_data, comment_text
        Returns empty DataFrame if file doesn't exist or is corrupted
        error_message is empty string if successful, otherwise contains error details
    """
    required_columns = [
        'id', 'timestamp', 'period1_start', 'period1_end',
        'period2_start', 'period2_end', 'agg_method', 'metrics',
        'comparison_data', 'comment_text'
    ]
    
    empty_df = pd.DataFrame(columns=required_columns)
    
    try:
        # Check if file exists
        if not Path(COMMENTS_FILE).exists():
            # Return empty DataFrame with expected columns (not an error)
            return empty_df, ""
        
        # Check file permissions
        if not os.access(COMMENTS_FILE, os.R_OK):
            return empty_df, f"Permission denied: Cannot read {COMMENTS_FILE}"
        
        # Read CSV file
        try:
            df = pd.read_csv(COMMENTS_FILE)
        except pd.errors.EmptyDataError:
            return empty_df, "CSV file is empty"
        except pd.errors.ParserError as e:
            # Create backup of corrupted file
            backup_file = f"{COMMENTS_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                Path(COMMENTS_FILE).rename(backup_file)
                return empty_df, f"CSV file is corrupted. Backup created at {backup_file}"
            except Exception:
                return empty_df, "CSV file is corrupted and could not be backed up"
        
        # Validate that required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            # Create backup of invalid file
            backup_file = f"{COMMENTS_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                Path(COMMENTS_FILE).rename(backup_file)
                return empty_df, f"CSV file is missing columns: {', '.join(missing_columns)}. Backup created at {backup_file}"
            except Exception:
                return empty_df, f"CSV file is missing required columns: {', '.join(missing_columns)}"
        
        # Validate data is not empty
        if df.empty:
            return empty_df, ""
        
        # Parse date strings to date objects with error handling
        date_columns = ['period1_start', 'period1_end', 'period2_start', 'period2_end']
        try:
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
            
            # Check if any dates failed to parse
            if df[date_columns].isnull().any().any():
                return empty_df, "Some date values in the CSV file are invalid"
        except Exception as e:
            return empty_df, f"Failed to parse dates: {str(e)}"
        
        # Parse timestamp strings to datetime objects with error handling
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isnull().any():
                return empty_df, "Some timestamp values in the CSV file are invalid"
        except Exception as e:
            return empty_df, f"Failed to parse timestamps: {str(e)}"
        
        # Validate JSON fields
        for idx, row in df.iterrows():
            try:
                json.loads(row['metrics'])
            except Exception:
                return empty_df, f"Invalid JSON in metrics field for comment ID: {row.get('id', 'unknown')}"
            
            try:
                json.loads(row['comparison_data'])
            except Exception:
                return empty_df, f"Invalid JSON in comparison_data field for comment ID: {row.get('id', 'unknown')}"
        
        return df, ""
        
    except PermissionError:
        return empty_df, f"Permission denied: Cannot access {COMMENTS_FILE}"
    except IOError as e:
        return empty_df, f"File I/O error: {str(e)}"
    except Exception as e:
        return empty_df, f"Unexpected error loading comments: {str(e)}"


def update_comment(comment_id: str, new_text: str) -> tuple[bool, str]:
    """
    Update the text of an existing comment.
    
    Args:
        comment_id: Unique identifier of the comment to update
        new_text: New comment text
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Validate inputs
        if not comment_id or not comment_id.strip():
            return False, "Comment ID cannot be empty"
        
        if not new_text or not new_text.strip():
            return False, "Comment text cannot be empty"
        
        # Check if file exists
        if not Path(COMMENTS_FILE).exists():
            return False, "Comments file not found"
        
        # Check file permissions
        if not os.access(COMMENTS_FILE, os.R_OK | os.W_OK):
            return False, f"Permission denied: Cannot read or write to {COMMENTS_FILE}"
        
        # Load existing comments
        try:
            df = pd.read_csv(COMMENTS_FILE)
        except pd.errors.EmptyDataError:
            return False, "Comments file is empty"
        except pd.errors.ParserError:
            return False, "Comments file is corrupted"
        except Exception as e:
            return False, f"Failed to read comments file: {str(e)}"
        
        # Find the comment by ID
        if comment_id not in df['id'].values:
            return False, f"Comment with ID {comment_id} not found"
        
        # Update the comment text
        df.loc[df['id'] == comment_id, 'comment_text'] = new_text.strip()
        
        # Save back to CSV
        try:
            df.to_csv(COMMENTS_FILE, index=False)
        except PermissionError:
            return False, f"Permission denied: Cannot write to {COMMENTS_FILE}"
        except IOError as e:
            return False, f"File I/O error: {str(e)}"
        
        return True, "Comment updated successfully"
        
    except Exception as e:
        return False, f"Unexpected error updating comment: {str(e)}"


def delete_comment(comment_id: str) -> tuple[bool, str]:
    """
    Delete a comment by ID.
    
    Args:
        comment_id: Unique identifier of the comment to delete
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Validate input
        if not comment_id or not comment_id.strip():
            return False, "Comment ID cannot be empty"
        
        # Check if file exists
        if not Path(COMMENTS_FILE).exists():
            return False, "Comments file not found"
        
        # Check file permissions
        if not os.access(COMMENTS_FILE, os.R_OK | os.W_OK):
            return False, f"Permission denied: Cannot read or write to {COMMENTS_FILE}"
        
        # Load existing comments
        try:
            df = pd.read_csv(COMMENTS_FILE)
        except pd.errors.EmptyDataError:
            return False, "Comments file is empty"
        except pd.errors.ParserError:
            return False, "Comments file is corrupted"
        except Exception as e:
            return False, f"Failed to read comments file: {str(e)}"
        
        # Check if comment exists
        if comment_id not in df['id'].values:
            return False, f"Comment with ID {comment_id} not found"
        
        # Filter out the target comment
        df = df[df['id'] != comment_id]
        
        # Save back to CSV
        try:
            df.to_csv(COMMENTS_FILE, index=False)
        except PermissionError:
            return False, f"Permission denied: Cannot write to {COMMENTS_FILE}"
        except IOError as e:
            return False, f"File I/O error: {str(e)}"
        
        return True, "Comment deleted successfully"
        
    except Exception as e:
        return False, f"Unexpected error deleting comment: {str(e)}"


def export_comments() -> tuple[str, str]:
    """
    Export all comments to CSV format for download.
    
    Returns:
        Tuple of (csv_string: str, error_message: str)
        CSV string ready for download with human-readable format
        error_message is empty string if successful, otherwise contains error details
    """
    try:
        # Load all comments
        df, error_msg = load_comments()
        
        if error_msg:
            return "", f"Failed to load comments: {error_msg}"
        
        if df.empty:
            return "", "No comments to export"
        
        # Create a copy for export with formatted columns
        export_df = df.copy()
        
        # Format timestamp for readability
        try:
            export_df['timestamp'] = pd.to_datetime(export_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            return "", f"Failed to format timestamps: {str(e)}"
        
        # Deserialize metrics for readability
        try:
            export_df['metrics'] = export_df['metrics'].apply(
                lambda x: ', '.join(json.loads(x)) if pd.notna(x) and x else ''
            )
        except Exception as e:
            return "", f"Failed to deserialize metrics: {str(e)}"
        
        # Keep comparison_data as JSON (it's complex data)
        # Or optionally remove it for simpler export
        
        # Convert to CSV string
        try:
            csv_string = export_df.to_csv(index=False)
        except Exception as e:
            return "", f"Failed to generate CSV: {str(e)}"
        
        return csv_string, ""
        
    except Exception as e:
        return "", f"Unexpected error exporting comments: {str(e)}"
