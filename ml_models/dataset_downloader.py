#!/usr/bin/env python3
"""
Dataset Downloader for Network Traffic Classification
Downloads and prepares multiple real-world datasets
"""

import os
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

class DatasetDownloader:
    """Download and prepare network traffic datasets"""
    
    def __init__(self, data_dir='../traffic_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and information
        self.datasets = {
            'cicids2017': {
                'name': 'CIC-IDS2017',
                'description': 'Canadian Institute for Cybersecurity Intrusion Detection Dataset',
                'url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
                'manual': True,
                'file_pattern': '*.csv',
                'features': ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                           'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
                           'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd Packet Length Mean',
                           'Bwd Packet Length Mean', 'Packet Length Mean', 'Packet Length Std',
                           'Protocol', 'Source Port', 'Destination Port']
            },
            'unsw_nb15': {
                'name': 'UNSW-NB15',
                'description': 'UNSW Network Behaviour 15 Dataset',
                'url': 'https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys?path=%2FUNSW-NB15%20-%20CSV%20Files',
                'manual': True,
                'file_pattern': 'UNSW*.csv',
                'features': ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
                           'sttl', 'dttl', 'sload', 'dload', 'sinpkt', 'dinpkt',
                           'sjit', 'djit', 'sport', 'dport', 'proto']
            },
            'iscx_vpn': {
                'name': 'ISCX-VPN-NonVPN',
                'description': 'VPN and Non-VPN Network Traffic',
                'url': 'https://www.unb.ca/cic/datasets/vpn.html',
                'manual': True,
                'file_pattern': '*.pcap',
                'features': ['Duration', 'Packets', 'Bytes', 'Protocol']
            },
            'moore': {
                'name': 'Moore Dataset',
                'description': 'Internet Traffic Classification',
                'local': True,
                'description_note': 'Classic dataset for app classification'
            }
        }
    
    def download_file(self, url, dest_path):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    def extract_archive(self, archive_path, extract_to):
        """Extract zip or gz archive"""
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix == '.gz':
            output_path = extract_to / archive_path.stem
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
    def download_cicids2017_sample(self):
        """
        Download CIC-IDS2017 sample data
        Note: Full dataset is large (~8GB), this downloads a sample
        """
        print("\n=== Downloading CIC-IDS2017 Sample ===")
        print("Full dataset: https://www.unb.ca/cic/datasets/ids-2017.html")
        
        dataset_dir = self.data_dir / 'cicids2017'
        dataset_dir.mkdir(exist_ok=True)
        
        # Sample URLs (smaller files for demonstration)
        sample_urls = [
            'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
            'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
        ]
        
        for url in sample_urls:
            filename = url.split('/')[-1]
            dest_path = dataset_dir / filename
            
            if not dest_path.exists():
                print(f"Downloading {filename}...")
                try:
                    self.download_file(url, dest_path)
                    print(f"✓ Downloaded {filename}")
                except Exception as e:
                    print(f"✗ Failed to download {filename}: {e}")
            else:
                print(f"✓ {filename} already exists")
        
        return dataset_dir
    
    def create_manual_download_instructions(self):
        """Create instructions for manually downloading datasets"""
        instructions_file = self.data_dir / 'DOWNLOAD_INSTRUCTIONS.txt'
        
        instructions = """
=================================================================
NETWORK TRAFFIC DATASETS - MANUAL DOWNLOAD INSTRUCTIONS
=================================================================

Due to dataset sizes and licensing, some datasets require manual download.
Follow these instructions to obtain the complete datasets:

1. CIC-IDS2017 Dataset
   -----------------------------------------------------------------
   URL: https://www.unb.ca/cic/datasets/ids-2017.html
   Size: ~8 GB
   
   Steps:
   a) Visit the URL above
   b) Download the CSV files (Monday through Friday)
   c) Extract and place in: traffic_data/cicids2017/
   
   Files needed:
   - Monday-WorkingHours.pcap_ISCX.csv
   - Tuesday-WorkingHours.pcap_ISCX.csv
   - Wednesday-workingHours.pcap_ISCX.csv
   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   - Friday-WorkingHours-Morning.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

2. UNSW-NB15 Dataset
   -----------------------------------------------------------------
   URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset
   Alternative: https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys
   Size: ~2.5 GB
   
   Steps:
   a) Visit the URL above
   b) Download CSV files: UNSW-NB15_1.csv, UNSW-NB15_2.csv, UNSW-NB15_3.csv, UNSW-NB15_4.csv
   c) Place in: traffic_data/unsw_nb15/
   
   Files needed:
   - UNSW-NB15_1.csv
   - UNSW-NB15_2.csv
   - UNSW-NB15_3.csv
   - UNSW-NB15_4.csv
   - UNSW-NB15_features.csv (feature descriptions)

3. ISCX VPN-NonVPN Dataset
   -----------------------------------------------------------------
   URL: https://www.unb.ca/cic/datasets/vpn.html
   Size: ~28 GB (PCAP files)
   
   Steps:
   a) Visit the URL above
   b) Download the dataset
   c) Place PCAP files in: traffic_data/iscx_vpn/
   
   Alternative: Use pre-processed features if available

4. CTU-13 Dataset
   -----------------------------------------------------------------
   URL: https://www.stratosphereips.org/datasets-ctu13
   Size: Variable
   
   Steps:
   a) Visit the URL above
   b) Download scenario files
   c) Place in: traffic_data/ctu13/

5. CICDDOS2019 Dataset
   -----------------------------------------------------------------
   URL: https://www.unb.ca/cic/datasets/ddos-2019.html
   Size: ~4 GB
   
   Steps:
   a) Visit the URL above
   b) Download CSV files
   c) Place in: traffic_data/cicddos2019/

=================================================================
ALTERNATIVE: Use Kaggle API
=================================================================

Some datasets are available on Kaggle. You can use Kaggle API:

1. Install Kaggle API:
   pip install kaggle

2. Setup Kaggle credentials:
   - Go to https://www.kaggle.com/account
   - Create API token
   - Place kaggle.json in ~/.kaggle/

3. Download datasets:
   kaggle datasets download -d cicdataset/cicids2017
   kaggle datasets download -d mrwellsdavid/unsw-nb15

=================================================================
AFTER DOWNLOADING
=================================================================

1. Verify files are in correct directories:
   traffic_data/
   ├── cicids2017/
   ├── unsw_nb15/
   ├── iscx_vpn/
   ├── ctu13/
   └── cicddos2019/

2. Run the dataset processor:
   python3 ml_models/dataset_processor.py

3. Train the model:
   python3 ml_models/train_classifier_real.py

=================================================================
"""
        
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(f"\n✓ Download instructions saved to: {instructions_file}")
        return instructions_file
    
    def setup_kaggle_api(self):
        """Setup Kaggle API for dataset downloads"""
        print("\n=== Setting up Kaggle API ===")
        
        try:
            import kaggle
            print("✓ Kaggle API is installed")
            
            # Test authentication
            kaggle.api.authenticate()
            print("✓ Kaggle API authenticated successfully")
            return True
            
        except ImportError:
            print("✗ Kaggle API not installed")
            print("  Install with: pip install kaggle")
            return False
        except Exception as e:
            print(f"✗ Kaggle authentication failed: {e}")
            print("\nTo setup Kaggle API:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token'")
            print("3. Save kaggle.json to ~/.kaggle/")
            print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
    
    def download_from_kaggle(self, dataset_name, output_dir):
        """Download dataset from Kaggle"""
        try:
            import kaggle
            
            print(f"\nDownloading {dataset_name} from Kaggle...")
            kaggle.api.dataset_download_files(
                dataset_name,
                path=output_dir,
                unzip=True
            )
            print(f"✓ Downloaded to {output_dir}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {e}")
            return False
    
    def download_available_datasets(self):
        """Download all available datasets"""
        print("\n" + "="*60)
        print("NETWORK TRAFFIC DATASET DOWNLOADER")
        print("="*60)
        
        # Create download instructions
        self.create_manual_download_instructions()
        
        # Try to setup Kaggle API
        kaggle_available = self.setup_kaggle_api()
        
        if kaggle_available:
            print("\n=== Attempting Kaggle Downloads ===")
            
            # Try to download from Kaggle
            kaggle_datasets = {
                'mldata/network-intrusion-detection': 'cicids2017',
                'dhoogla/cicids2017': 'cicids2017_alt'
            }
            
            for dataset_name, folder_name in kaggle_datasets.items():
                output_dir = self.data_dir / folder_name
                output_dir.mkdir(exist_ok=True)
                self.download_from_kaggle(dataset_name, output_dir)
        
        # Download sample data
        print("\n=== Downloading Sample Datasets ===")
        self.download_cicids2017_sample()
        
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"\nData directory: {self.data_dir.absolute()}")
        print("\nFor complete datasets, please follow the instructions in:")
        print(f"  {self.data_dir}/DOWNLOAD_INSTRUCTIONS.txt")
        print("\nAfter downloading, run:")
        print("  python3 ml_models/dataset_processor.py")
        print("="*60)

def main():
    """Main function"""
    downloader = DatasetDownloader()
    downloader.download_available_datasets()

if __name__ == '__main__':
    main()