import json
import os
from datetime import datetime
from pathlib import Path


class DiagnosisExporter:
    """
    Utility class to export diagnosis results to different formats.
    Currently supports JSON export.
    """
    def __init__(self):
        pass

    @staticmethod
    def export_to_json(diagnosis_data, output_dir=None):
        """
        Export diagnosis data to a JSON file.
        
        Args:
            diagnosis_data (dict): Dictionary containing diagnosis information
            output_dir (str): Directory to save the JSON file
            
        Returns:
            str: Path to the saved JSON file
        """
        # Set default output directory relative to project root
        output_path = os.path.join(output_dir, "data", "diagnosis_results")
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Create filename with timestamp
        filename = f"diagnosis_{timestamp}.json"
        file_path = os.path.join(output_path, filename)
        
        # Add metadata to diagnosis data
        diagnosis_data["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Write to JSON file
        with open(file_path, "w") as f:
            json.dump(diagnosis_data, f, indent=4)
            
        print(f"Diagnosis exported to: {file_path}")
        
        return str(file_path)