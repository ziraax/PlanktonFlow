"""
Data Input Handlers for Different Dataset Formats
Supports hierarchical, CSV mapping, and EcoTaxa formats
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod


class DataInputHandler(ABC):
    """Abstract base class for data input handlers"""
    
    @abstractmethod
    def get_image_label_pairs(self) -> List[Tuple[str, str]]:
        """
        Returns list of (image_path, label) tuples
        """
        pass
    
    @abstractmethod
    def validate_input(self) -> bool:
        """
        Validates that the input source is valid and accessible
        """
        pass


class HierarchicalDataHandler(DataInputHandler):
    """Handles hierarchical/stratified datasets organized by class folders"""
    
    def __init__(self, config: Dict[str, Any]):
        self.data_path = Path(config['data_path'])
        self.subdirs = config.get('subdirs', [])
        
    def validate_input(self) -> bool:
        """Check if data path exists and has class folders"""
        if not self.data_path.exists():
            print(f"[ERROR] Data path does not exist: {self.data_path}")
            return False
            
        # If subdirs specified, check each one
        if self.subdirs:
            for subdir in self.subdirs:
                subdir_path = self.data_path / subdir
                if not subdir_path.exists():
                    print(f"[ERROR] Subdirectory does not exist: {subdir_path}")
                    return False
        else:
            # Check if there are class folders directly under data_path
            class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
            if not class_dirs:
                print(f"[ERROR] No class directories found in: {self.data_path}")
                return False
                
        return True
    
    def get_image_label_pairs(self) -> List[Tuple[str, str]]:
        """Extract image-label pairs from hierarchical structure"""
        image_label_pairs = []
        
        if self.subdirs:
            # Process each subdirectory
            for subdir in self.subdirs:
                subdir_path = self.data_path / subdir
                pairs = self._extract_from_directory(subdir_path)
                image_label_pairs.extend(pairs)
        else:
            # Process direct class folders
            pairs = self._extract_from_directory(self.data_path)
            image_label_pairs.extend(pairs)
            
        print(f"[INFO] Found {len(image_label_pairs)} images in hierarchical dataset")
        return image_label_pairs
    
    def _extract_from_directory(self, base_path: Path) -> List[Tuple[str, str]]:
        """Extract image-label pairs from a directory containing class folders"""
        pairs = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for class_dir in base_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    pairs.append((str(img_file), class_name))
                    
        return pairs


class CSVMappingDataHandler(DataInputHandler):
    """Handles flat datasets with CSV/TSV metadata mapping"""
    
    def __init__(self, config: Dict[str, Any]):
        self.images_path = Path(config['images_path'])
        self.metadata_file = Path(config['metadata_file'])
        self.image_column = config['image_column']
        self.label_column = config['label_column']
        self.separator = config['separator']
        self.image_path_prefix = config.get('image_path_prefix', '')
        
    def validate_input(self) -> bool:
        """Check if images directory and metadata file exist"""
        if not self.images_path.exists():
            print(f"[ERROR] Images directory does not exist: {self.images_path}")
            return False
            
        if not self.metadata_file.exists():
            print(f"[ERROR] Metadata file does not exist: {self.metadata_file}")
            return False
            
        # Quick check of CSV structure
        try:
            df = pd.read_csv(self.metadata_file, sep=self.separator, nrows=1)
            if self.image_column not in df.columns:
                print(f"[ERROR] Image column '{self.image_column}' not found in metadata file")
                return False
            if self.label_column not in df.columns:
                print(f"[ERROR] Label column '{self.label_column}' not found in metadata file")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to read metadata file: {e}")
            return False
            
        return True
    
    def get_image_label_pairs(self) -> List[Tuple[str, str]]:
        """Extract image-label pairs from CSV mapping"""
        try:
            # Read metadata
            df = pd.read_csv(self.metadata_file, sep=self.separator)
            
            # Remove rows with missing values
            df = df.dropna(subset=[self.image_column, self.label_column])
            
            image_label_pairs = []
            
            for _, row in df.iterrows():
                image_filename = row[self.image_column]
                label = str(row[self.label_column]).strip()
                
                # Construct full image path
                if self.image_path_prefix:
                    image_path = self.images_path / self.image_path_prefix / image_filename
                else:
                    image_path = self.images_path / image_filename
                
                # Check if image exists
                if image_path.exists():
                    image_label_pairs.append((str(image_path), label))
                else:
                    print(f"[WARNING] Image not found: {image_path}")
            
            print(f"[INFO] Found {len(image_label_pairs)} valid images from CSV mapping")
            return image_label_pairs
            
        except Exception as e:
            print(f"[ERROR] Failed to process CSV mapping: {e}")
            return []


class EcoTaxaDataHandler(DataInputHandler):
    """Handles EcoTaxa TSV export formats"""
    
    def __init__(self, config: Dict[str, Any]):
        self.data_path = Path(config['data_path'])
        self.metadata_file = config['metadata_file']  # relative to data_path
        self.separator = config.get('separator', '\t')  # TSV default
        
    def validate_input(self) -> bool:
        """Validate EcoTaxa input"""
        if not self.data_path.exists():
            print(f"[ERROR] Data path does not exist: {self.data_path}")
            return False
            
        metadata_path = self.data_path / self.metadata_file
        if not metadata_path.exists():
            print(f"[ERROR] EcoTaxa metadata file does not exist: {metadata_path}")
            return False
            
        # Quick check of EcoTaxa TSV structure (skip type line)
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                header_line = f.readline().strip().split(self.separator)
                type_line = f.readline().strip().split(self.separator)  # Skip this line
            
            # Remove quotes from column names for comparison
            clean_headers = [col.strip('"') for col in header_line]
            
            if 'object_id' not in clean_headers:
                print(f"[ERROR] Column 'object_id' not found in EcoTaxa TSV")
                print(f"[DEBUG] Available columns: {clean_headers[:10]}...")
                return False
            if 'object_annotation_category' not in clean_headers:
                print(f"[ERROR] Column 'object_annotation_category' not found in EcoTaxa TSV")
                print(f"[DEBUG] Available columns: {clean_headers[:10]}...")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to read EcoTaxa TSV: {e}")
            return False
            
        return True
    
    def get_image_label_pairs(self) -> List[Tuple[str, str]]:
        """Extract image-label pairs from EcoTaxa TSV export"""
        try:
            metadata_path = self.data_path / self.metadata_file
            
            # Read EcoTaxa TSV with special handling (skip type line)
            df = pd.read_csv(metadata_path, sep=self.separator, skiprows=[1])
            
            # Remove rows with missing values
            df = df.dropna(subset=['object_id', 'object_annotation_category'])
            
            # Clean up category names for folder compatibility
            df['object_annotation_category'] = df['object_annotation_category'].str.replace('<', '_', regex=False)
            df['object_annotation_category'] = df['object_annotation_category'].str.replace('>', '_', regex=False)
            
            image_label_pairs = []
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            for _, row in df.iterrows():
                object_id = str(row['object_id']).strip()
                label = str(row['object_annotation_category']).strip()
                
                # Try to find the image with different extensions
                image_path = None
                for ext in image_extensions:
                    candidate_path = self.data_path / f"{object_id}{ext}"
                    if candidate_path.exists():
                        image_path = candidate_path
                        break
                
                # If not found directly, try looking in subdirectories
                if image_path is None:
                    for subdir in self.data_path.iterdir():
                        if subdir.is_dir():
                            for ext in image_extensions:
                                candidate_path = subdir / f"{object_id}{ext}"
                                if candidate_path.exists():
                                    image_path = candidate_path
                                    break
                            if image_path:
                                break
                
                if image_path:
                    image_label_pairs.append((str(image_path), label))
                else:
                    print(f"[WARNING] EcoTaxa image not found: {object_id}")
            
            print(f"[INFO] Found {len(image_label_pairs)} valid images from EcoTaxa TSV")
            return image_label_pairs
            
        except Exception as e:
            print(f"[ERROR] Failed to process EcoTaxa TSV: {e}")
            return []


def create_data_handler(config: Dict[str, Any]) -> DataInputHandler:
    """Factory function to create appropriate data handler"""
    input_type = config['type']
    
    if input_type == 'hierarchical':
        return HierarchicalDataHandler(config)
    elif input_type == 'csv_mapping':
        return CSVMappingDataHandler(config)
    elif input_type == 'ecotaxa':
        return EcoTaxaDataHandler(config)
    else:
        raise ValueError(f"Unknown input source type: {input_type}")
