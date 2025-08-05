"""
Data Download Script for M5 Competition Dataset
Automatically downloads data from Dropbox for the Dairy Analytics Platform
"""

import os
import requests
import pandas as pd
from pathlib import Path
import streamlit as st

class DataDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dropbox direct download URLs (converted from share links)
        self.data_urls = {
            "calendar.csv": "https://www.dropbox.com/scl/fi/vsaku007rzkxoz70k89h2/calendar.csv?rlkey=u052cdmwft2n55c0ti859ox9l&st=n9a251z6&dl=1",
            "sales_train_evaluation.csv": "https://www.dropbox.com/scl/fi/xicsguhot8lsvlkt5zmcf/sales_train_evaluation.csv?rlkey=ndcw02wvjmlvtaltcpc80onz1&st=006ovwr9&dl=1",
            "sample_submission.csv": "https://www.dropbox.com/scl/fi/s92q5k5a7dr14l3jglav8/sample_submission.csv?rlkey=ue7w36bojyrjyj1axsxu0py6a&st=6aos4y5w&dl=1",
            "sales_train_validation.csv": "https://www.dropbox.com/scl/fi/617tvkpy27ig07r0qfe9h/sales_train_validation.csv?rlkey=l5ziqku01simgk8rhsrhdj8ex&st=5dgql42a&dl=1",
            "sell_prices.csv": "https://www.dropbox.com/scl/fi/w2gdk100n2dgrnlug920q/sell_prices.csv?rlkey=bd90o6sadnwxbwz8noiuawtwc&st=gykf3ds2&dl=1"
        }
        
        # File sizes for progress tracking (approximate)
        self.file_sizes = {
            "calendar.csv": 103000,  # ~103 KB
            "sales_train_evaluation.csv": 122000000,  # ~122 MB
            "sample_submission.csv": 5200000,  # ~5.2 MB
            "sales_train_validation.csv": 120000000,  # ~120 MB
            "sell_prices.csv": 203000000  # ~203 MB
        }

    def check_data_exists(self):
        """Check if essential data files already exist"""
        required_files = ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv"]
        existing_files = [f for f in required_files if (self.data_dir / f).exists()]
        
        if len(existing_files) == len(required_files):
            return True
        elif len(existing_files) > 0:
            st.info(f"Found {len(existing_files)}/{len(required_files)} data files. Missing files will be downloaded.")
            return False
        else:
            return False

    def download_file(self, url, filename, use_cache=True):
        """Download a single file with progress tracking"""
        file_path = self.data_dir / filename
        
        if use_cache and file_path.exists():
            file_size = os.path.getsize(file_path)
            if file_size > 1000:  # File exists and has content
                st.success(f"✅ {filename} already exists ({file_size:,} bytes)")
                return True
            else:
                st.warning(f"⚠️ {filename} exists but appears corrupted, re-downloading...")
                os.remove(file_path)
        
        try:
            st.info(f"📥 Downloading {filename}...")
            
            # Create progress placeholder
            progress_placeholder = st.empty()
            
            # Start download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get content length
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                total_size = self.file_sizes.get(filename, 0)
            
            downloaded = 0
            chunk_size = 8192
            
            with open(file_path, 'wb') as f:
                if total_size > 0:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Update progress
                            progress = min(downloaded / total_size, 1.0)
                            progress_bar.progress(progress)
                            
                            # Update status
                            downloaded_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            status_text.text(f"Downloaded: {downloaded_mb:.1f}/{total_mb:.1f} MB ({progress*100:.1f}%)")
                    
                    progress_bar.empty()
                    status_text.empty()
                else:
                    # No content length, download without progress
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                    
                    st.info(f"Downloaded {downloaded:,} bytes (size unknown)")
            
            # Verify download
            final_size = os.path.getsize(file_path)
            if final_size < 1000:
                st.error(f"❌ {filename} download failed - file too small ({final_size} bytes)")
                os.remove(file_path)
                return False
            
            st.success(f"✅ Downloaded {filename} successfully ({final_size:,} bytes)")
            progress_placeholder.empty()
            return True
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error downloading {filename}: {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Failed to download {filename}: {str(e)}")
            return False

    def validate_data_files(self):
        """Validate that downloaded files are proper CSV files"""
        validation_results = {}
        
        for filename in ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv"]:
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                validation_results[filename] = {"valid": False, "error": "File not found"}
                continue
            
            try:
                # Try to read a few rows to validate CSV format
                df = pd.read_csv(file_path, nrows=5)
                
                # Basic validation checks
                if len(df) == 0:
                    validation_results[filename] = {"valid": False, "error": "Empty file"}
                elif len(df.columns) < 2:
                    validation_results[filename] = {"valid": False, "error": "Invalid CSV structure"}
                else:
                    validation_results[filename] = {
                        "valid": True, 
                        "rows_sample": len(df),
                        "columns": len(df.columns),
                        "size_mb": os.path.getsize(file_path) / (1024 * 1024)
                    }
                    
            except Exception as e:
                validation_results[filename] = {"valid": False, "error": f"Read error: {str(e)}"}
        
        return validation_results

    def setup_data(self):
        """Main method to setup data"""
        st.info("🔄 Setting up data files from Dropbox...")
        
        if self.check_data_exists():
            st.success("✅ All required data files already exist!")
            
            # Validate existing files
            validation = self.validate_data_files()
            all_valid = all(result["valid"] for result in validation.values())
            
            if all_valid:
                st.success("✅ All data files validated successfully!")
                return True
            else:
                st.warning("⚠️ Some data files may be corrupted. Re-downloading...")
        
        # Download missing or corrupted files
        download_success = True
        essential_files = ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv"]
        
        for filename in essential_files:
            file_path = self.data_dir / filename
            if not file_path.exists() or os.path.getsize(file_path) < 1000:
                url = self.data_urls[filename]
                success = self.download_file(url, filename, use_cache=False)
                if not success:
                    download_success = False
                    st.error(f"❌ Failed to download {filename}")
        
        # Download optional files
        optional_files = ["sales_train_evaluation.csv", "sample_submission.csv"]
        for filename in optional_files:
            file_path = self.data_dir / filename
            if not file_path.exists():
                url = self.data_urls[filename]
                st.info(f"📥 Downloading optional file: {filename}")
                self.download_file(url, filename, use_cache=True)
        
        if download_success:
            # Final validation
            validation = self.validate_data_files()
            all_valid = all(result["valid"] for result in validation.values())
            
            if all_valid:
                st.success("🎉 Data setup completed successfully!")
                
                # Show data summary
                st.subheader("📊 Data Files Summary")
                summary_data = []
                for filename, result in validation.items():
                    if result["valid"]:
                        summary_data.append({
                            "File": filename,
                            "Size (MB)": f"{result['size_mb']:.1f}",
                            "Columns": result['columns'],
                            "Status": "✅ Ready"
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                return True
            else:
                st.error("❌ Data validation failed after download")
                return False
        else:
            st.error("❌ Data setup failed - some essential files could not be downloaded")
            return False

def download_data_if_needed():
    """Function to be called from Streamlit app"""
    downloader = DataDownloader()
    
    if not downloader.check_data_exists():
        st.warning("📂 Data files not found. Downloading from Dropbox...")
        
        with st.spinner("Downloading data files from Dropbox..."):
            success = downloader.setup_data()
        
        if success:
            st.success("✅ Data setup complete! Refreshing app...")
            st.rerun()
        else:
            st.error("❌ Data setup failed! Please check your internet connection and try again.")
            st.stop()
            return False
    else:
        # Quick validation of existing files
        validation = downloader.validate_data_files()
        all_valid = all(result.get("valid", False) for result in validation.values())
        
        if not all_valid:
            st.warning("⚠️ Data files may be corrupted. Re-downloading...")
            with st.spinner("Re-downloading data files..."):
                success = downloader.setup_data()
            if success:
                st.rerun()
            else:
                st.error("❌ Data repair failed!")
                return False
    
    return True

if __name__ == "__main__":
    # For testing
    downloader = DataDownloader()
    print("Testing data downloader...")
    success = downloader.setup_data()
    print(f"Setup result: {success}")
