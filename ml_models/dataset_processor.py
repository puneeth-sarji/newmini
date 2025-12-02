#!/usr/bin/env python3
"""
Dataset Processor for Multiple Network Traffic Datasets
Processes and unifies different dataset formats
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

warnings.filterwarnings('ignore')

class DatasetProcessor:
    """Process multiple network traffic datasets into unified format"""
    
    def __init__(self, data_dir='../traffic_data'):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Unified feature names
        self.unified_features = [
            'duration',
            'protocol',
            'src_port',
            'dst_port',
            'total_fwd_packets',
            'total_bwd_packets',
            'total_length_fwd_packets',
            'total_length_bwd_packets',
            'fwd_packet_length_mean',
            'bwd_packet_length_mean',
            'flow_bytes_per_sec',
            'flow_packets_per_sec',
            'flow_iat_mean',
            'flow_iat_std',
            'fwd_iat_mean',
            'bwd_iat_mean',
            'packet_length_mean',
            'packet_length_std'
        ]
        
        # Application mapping for classification
        self.app_mapping = {
            # Web browsing
            'http': 'HTTP', 'https': 'HTTP', 'web': 'HTTP', 'ssl': 'HTTP',
            # Video streaming
            'youtube': 'Video', 'netflix': 'Video', 'vimeo': 'Video', 
            'streaming': 'Video', 'video': 'Video',
            # VoIP
            'skype': 'VoIP', 'voip': 'VoIP', 'hangouts': 'VoIP',
            'voice': 'VoIP', 'sip': 'VoIP',
            # File transfer
            'ftp': 'FTP', 'sftp': 'FTP', 'ftps': 'FTP', 'bittorrent': 'FTP',
            'torrent': 'FTP', 'file_transfer': 'FTP',
            # Gaming
            'gaming': 'Gaming', 'game': 'Gaming', 'dota': 'Gaming',
            'wow': 'Gaming', 'counterstrike': 'Gaming',
            # Email
            'smtp': 'Email', 'pop3': 'Email', 'imap': 'Email', 'email': 'Email',
            # P2P
            'p2p': 'P2P', 'peer': 'P2P',
            # Others
            'dns': 'DNS', 'ntp': 'NTP', 'ssh': 'SSH'
        }
    
    def process_cicids2017(self, file_path):
        """Process CIC-IDS2017 dataset"""
        print(f"\nProcessing CIC-IDS2017: {file_path.name}")
        
        try:
            # Read CSV
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            
            # Remove leading/trailing spaces from column names
            df.columns = df.columns.str.strip()
            
            print(f"  Original shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()[:5]}...")
            
            # Map to unified features
            feature_mapping = {
                'Flow Duration': 'duration',
                'Protocol': 'protocol',
                'Source Port': 'src_port',
                'Destination Port': 'dst_port',
                'Total Fwd Packets': 'total_fwd_packets',
                'Total Backward Packets': 'total_bwd_packets',
                'Total Length of Fwd Packets': 'total_length_fwd_packets',
                'Total Length of Bwd Packets': 'total_length_bwd_packets',
                'Fwd Packet Length Mean': 'fwd_packet_length_mean',
                'Bwd Packet Length Mean': 'bwd_packet_length_mean',
                'Flow Bytes/s': 'flow_bytes_per_sec',
                'Flow Packets/s': 'flow_packets_per_sec',
                'Flow IAT Mean': 'flow_iat_mean',
                'Flow IAT Std': 'flow_iat_std',
                'Fwd IAT Mean': 'fwd_iat_mean',
                'Bwd IAT Mean': 'bwd_iat_mean',
                'Packet Length Mean': 'packet_length_mean',
                'Packet Length Std': 'packet_length_std',
                'Label': 'label'
            }
            
            # Select and rename columns
            available_cols = [col for col in feature_mapping.keys() if col in df.columns]
            df_processed = df[available_cols].copy()
            df_processed.rename(columns=feature_mapping, inplace=True)
            
            # Map labels to application types
            df_processed['application'] = 'HTTP'  # Default
            df_processed.loc[df_processed['label'].str.lower().str.contains('benign|normal', na=False), 'application'] = 'HTTP'
            
            # Remove attack labels, keep only normal traffic
            df_processed = df_processed[df_processed['label'].str.lower().str.contains('benign', na=False)]
            
            print(f"  Processed shape: {df_processed.shape}")
            return df_processed
            
        except Exception as e:
            print(f"  Error processing CIC-IDS2017: {e}")
            return None
    
    def process_unsw_nb15(self, file_path):
        """Process UNSW-NB15 dataset"""
        print(f"\nProcessing UNSW-NB15: {file_path.name}")
        
        try:
            # Read CSV
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            
            print(f"  Original shape: {df.shape}")
            
            # Map to unified features
            feature_mapping = {
                'dur': 'duration',
                'proto': 'protocol',
                'sport': 'src_port',
                'dport': 'dst_port',
                'spkts': 'total_fwd_packets',
                'dpkts': 'total_bwd_packets',
                'sbytes': 'total_length_fwd_packets',
                'dbytes': 'total_length_bwd_packets',
                'smeansz': 'fwd_packet_length_mean',
                'dmeansz': 'bwd_packet_length_mean',
                'rate': 'flow_packets_per_sec',
                'sjit': 'fwd_iat_mean',
                'djit': 'bwd_iat_mean',
                'attack_cat': 'label'
            }
            
            # Select and rename columns
            available_cols = [col for col in feature_mapping.keys() if col in df.columns]
            df_processed = df[available_cols].copy()
            df_processed.rename(columns=feature_mapping, inplace=True)
            
            # Calculate derived features
            if 'total_length_fwd_packets' in df_processed.columns and 'total_length_bwd_packets' in df_processed.columns:
                df_processed['flow_bytes_per_sec'] = (
                    df_processed['total_length_fwd_packets'] + 
                    df_processed['total_length_bwd_packets']
                ) / (df_processed['duration'] + 1e-6)
            
            # Map service to application
            if 'service' in df.columns:
                df_processed['service'] = df['service']
                df_processed['application'] = df_processed['service'].map(
                    lambda x: self.app_mapping.get(str(x).lower(), 'Other')
                )
            else:
                df_processed['application'] = 'HTTP'
            
            # Keep only normal traffic
            df_processed = df_processed[df_processed['label'].str.lower() == 'normal']
            
            print(f"  Processed shape: {df_processed.shape}")
            return df_processed
            
        except Exception as e:
            print(f"  Error processing UNSW-NB15: {e}")
            return None
    
    def process_kdd_dataset(self, file_path):
        """Process KDD Cup / NSL-KDD dataset"""
        print(f"\nProcessing KDD dataset: {file_path.name}")
        
        try:
            # KDD column names
            kdd_columns = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate', 'label', 'difficulty'
            ]
            
            # Read CSV
            df = pd.read_csv(file_path, names=kdd_columns, low_memory=False)
            
            print(f"  Original shape: {df.shape}")
            
            # Create unified features
            df_processed = pd.DataFrame()
            df_processed['duration'] = df['duration']
            df_processed['protocol'] = df['protocol_type']
            df_processed['total_length_fwd_packets'] = df['src_bytes']
            df_processed['total_length_bwd_packets'] = df['dst_bytes']
            df_processed['total_fwd_packets'] = df['count']
            df_processed['src_port'] = 0  # Not available
            df_processed['dst_port'] = 0  # Not available
            
            # Calculate derived features
            df_processed['flow_bytes_per_sec'] = (
                df_processed['total_length_fwd_packets'] + 
                df_processed['total_length_bwd_packets']
            ) / (df_processed['duration'] + 1e-6)
            
            # Map service to application
            df_processed['application'] = df['service'].map(
                lambda x: self.app_mapping.get(str(x).lower(), 'HTTP')
            )
            
            # Keep only normal traffic
            df_processed['label'] = df['label']
            df_processed = df_processed[df_processed['label'].str.lower() == 'normal']
            
            print(f"  Processed shape: {df_processed.shape}")
            return df_processed
            
        except Exception as e:
            print(f"  Error processing KDD dataset: {e}")
            return None
    
    def create_synthetic_balanced_dataset(self, n_samples_per_class=10000):
        """Create synthetic balanced dataset when real data is limited"""
        print("\nCreating synthetic balanced dataset...")
        
        data = []
        labels = []
        
        # Define application profiles
        app_profiles = {
            'HTTP': {
                'duration': (1, 60), 'packets': (10, 100),
                'packet_size': (100, 1500), 'iat': (0.1, 0.5),
                'protocol': 6, 'port': 80
            },
            'Video': {
                'duration': (60, 600), 'packets': (500, 5000),
                'packet_size': (1000, 1500), 'iat': (0.03, 0.05),
                'protocol': 17, 'port': 554
            },
            'VoIP': {
                'duration': (30, 300), 'packets': (500, 3000),
                'packet_size': (50, 200), 'iat': (0.02, 0.03),
                'protocol': 17, 'port': 5060
            },
            'FTP': {
                'duration': (10, 600), 'packets': (100, 10000),
                'packet_size': (1200, 1500), 'iat': (0.001, 0.01),
                'protocol': 6, 'port': 21
            },
            'Gaming': {
                'duration': (60, 600), 'packets': (1000, 10000),
                'packet_size': (50, 150), 'iat': (0.01, 0.05),
                'protocol': 17, 'port': 27015
            }
        }
        
        for app, profile in app_profiles.items():
            for _ in range(n_samples_per_class):
                duration = np.random.uniform(*profile['duration'])
                fwd_packets = np.random.randint(*profile['packets'])
                bwd_packets = int(fwd_packets * np.random.uniform(0.3, 0.9))
                
                packet_size = np.random.uniform(*profile['packet_size'])
                fwd_bytes = int(fwd_packets * packet_size)
                bwd_bytes = int(bwd_packets * packet_size * np.random.uniform(0.5, 1.5))
                
                iat = np.random.uniform(*profile['iat'])
                
                features = [
                    duration,
                    profile['protocol'],
                    np.random.randint(1024, 65535),  # src_port
                    profile['port'],  # dst_port
                    fwd_packets,
                    bwd_packets,
                    fwd_bytes,
                    bwd_bytes,
                    packet_size,
                    packet_size * np.random.uniform(0.8, 1.2),
                    (fwd_bytes + bwd_bytes) / duration if duration > 0 else 0,
                    (fwd_packets + bwd_packets) / duration if duration > 0 else 0,
                    iat,
                    iat * np.random.uniform(0.5, 1.5),
                    iat * np.random.uniform(0.8, 1.2),
                    iat * np.random.uniform(0.8, 1.2),
                    (fwd_bytes + bwd_bytes) / (fwd_packets + bwd_packets) if (fwd_packets + bwd_packets) > 0 else 0,
                    packet_size * np.random.uniform(0.2, 0.5)
                ]
                
                data.append(features)
                labels.append(app)
        
        df = pd.DataFrame(data, columns=self.unified_features)
        df['application'] = labels
        
        print(f"  Created synthetic dataset: {df.shape}")
        print(f"  Classes: {df['application'].value_counts().to_dict()}")
        
        return df
    
    def combine_and_balance_datasets(self, datasets, min_samples_per_class=5000):
        """Combine multiple datasets and balance classes"""
        print("\n" + "="*60)
        print("COMBINING AND BALANCING DATASETS")
        print("="*60)
        
        # Combine all datasets
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"\nCombined dataset shape: {combined_df.shape}")
        print(f"Total samples: {len(combined_df)}")
        
        # Show class distribution
        print("\nClass distribution:")
        class_counts = combined_df['application'].value_counts()
        for app, count in class_counts.items():
            print(f"  {app}: {count}")
        
        # Balance classes
        balanced_dfs = []
        for app in class_counts.index:
            app_df = combined_df[combined_df['application'] == app]
            
            if len(app_df) < min_samples_per_class:
                # Oversample
                app_df = app_df.sample(n=min_samples_per_class, replace=True, random_state=42)
            else:
                # Downsample
                app_df = app_df.sample(n=min_samples_per_class, replace=False, random_state=42)
            
            balanced_dfs.append(app_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        print(f"\nBalanced dataset shape: {balanced_df.shape}")
        print("\nBalanced class distribution:")
        for app, count in balanced_df['application'].value_counts().items():
            print(f"  {app}: {count}")
        
        return balanced_df
    
    def process_all_datasets(self):
        """Process all available datasets"""
        print("\n" + "="*60)
        print("DATASET PROCESSOR")
        print("="*60)
        
        all_datasets = []
        
        # Process CIC-IDS2017
        cicids_dir = self.data_dir / 'cicids2017'
        if cicids_dir.exists():
            for csv_file in cicids_dir.glob('*.csv'):
                df = self.process_cicids2017(csv_file)
                if df is not None and len(df) > 0:
                    all_datasets.append(df)
        
        # Process UNSW-NB15
        unsw_dir = self.data_dir / 'unsw_nb15'
        if unsw_dir.exists():
            for csv_file in unsw_dir.glob('UNSW*.csv'):
                df = self.process_unsw_nb15(csv_file)
                if df is not None and len(df) > 0:
                    all_datasets.append(df)
        
        # Process KDD
        for kdd_dir in [self.data_dir / 'kdd', self.data_dir / 'cicids2017']:
            if kdd_dir.exists():
                for txt_file in kdd_dir.glob('*.txt'):
                    df = self.process_kdd_dataset(txt_file)
                    if df is not None and len(df) > 0:
                        all_datasets.append(df)
        
        # If no real datasets found or insufficient data, add synthetic data
        if len(all_datasets) == 0:
            print("\n⚠ No real datasets found. Creating synthetic dataset...")
            synthetic_df = self.create_synthetic_balanced_dataset(10000)
            all_datasets.append(synthetic_df)
        else:
            # Add synthetic data to balance
            print("\n✓ Adding synthetic data for balance...")
            synthetic_df = self.create_synthetic_balanced_dataset(5000)
            all_datasets.append(synthetic_df)
        
        # Combine and balance
        final_df = self.combine_and_balance_datasets(all_datasets)
        
        # Clean data
        final_df = self.clean_dataset(final_df)
        
        # Save processed dataset
        output_file = self.processed_dir / 'unified_traffic_dataset.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved processed dataset to: {output_file}")
        
        # Save dataset statistics
        self.save_statistics(final_df)
        
        return final_df
    
    def clean_dataset(self, df):
        """Clean and preprocess dataset"""
        print("\nCleaning dataset...")
        
        original_size = len(df)
        
        # Replace inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Remove outliers (optional)
        # Using IQR method for key features
        # This is commented out to preserve more data
        # for col in numeric_cols:
        #     Q1 = df[col].quantile(0.25)
        #     Q3 = df[col].quantile(0.75)
        #     IQR = Q3 - Q1
        #     df = df[(df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)]
        
        print(f"  Removed {original_size - len(df)} rows during cleaning")
        print(f"  Final dataset size: {len(df)}")
        
        return df
    
    def save_statistics(self, df):
        """Save dataset statistics"""
        stats = {
            'total_samples': len(df),
            'num_features': len(self.unified_features),
            'classes': df['application'].unique().tolist(),
            'class_distribution': df['application'].value_counts().to_dict(),
            'feature_statistics': {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats['feature_statistics'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
        
        stats_file = self.processed_dir / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Saved dataset statistics to: {stats_file}")

def main():
    """Main function"""
    processor = DatasetProcessor()
    df = processor.process_all_datasets()
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nProcessed dataset saved with {len(df)} samples")
    print("\nNext step:")
    print("  python3 ml_models/train_classifier_real.py")
    print("="*60)

if __name__ == '__main__':
    main()