#!/usr/bin/env python
"""
Clear the document manifest to force re-processing of all documents.
Useful after fixing ingestion issues.
"""

import os
import json

def clear_manifest():
    """Clear the document processing manifest."""
    manifest_path = 'config/document_manifest.json'
    
    if os.path.exists(manifest_path):
        print(f"Found manifest at: {manifest_path}")
        
        # Backup the old manifest
        backup_path = manifest_path + '.backup'
        with open(manifest_path, 'r') as f:
            data = f.read()
        with open(backup_path, 'w') as f:
            f.write(data)
        print(f"Created backup at: {backup_path}")
        
        # Remove the manifest
        os.remove(manifest_path)
        print("✓ Manifest cleared successfully")
        print("All documents will be re-processed on next ingestion run")
    else:
        print("No manifest found. Documents will be processed fresh.")

if __name__ == "__main__":
    clear_manifest()
