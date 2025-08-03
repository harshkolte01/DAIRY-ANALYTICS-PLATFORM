"""
Enhanced Pre-train ML Models Script with Progress Tracking
This script trains all ML models offline with comprehensive progress bars and timing
"""

import sys
import os
import time
from datetime import datetime, timedelta
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ml_data_loader import MLEnhancedDataLoader
from utils.data_loader import load_sales_data, load_calendar_data, load_prices_data, get_available_stores_and_states


def print_banner():
    """Print training banner"""
    print("=" * 70)
    print("🤖 DAIRY ANALYTICS ML MODEL TRAINING WITH PROGRESS TRACKING")
    print("=" * 70)
    print()


def print_progress_bar(progress, message, elapsed_time, width=50):
    """Print enhanced console progress bar with timing"""
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    
    # Calculate ETA
    if progress > 0:
        total_time = elapsed_time / progress
        eta_seconds = total_time - elapsed_time
        eta = timedelta(seconds=int(max(0, eta_seconds)))
        time_str = f"Elapsed: {timedelta(seconds=int(elapsed_time))} | ETA: {eta}"
    else:
        time_str = f"Elapsed: {timedelta(seconds=int(elapsed_time))}"
    
    print(f"\r[{bar}] {progress*100:.1f}% | {message:<35} | {time_str}", end="", flush=True)
    if progress >= 1.0:
        print()  # New line when complete


def main():
    """Enhanced main function with comprehensive progress tracking"""
    
    parser = argparse.ArgumentParser(description='Pre-train ML models with progress tracking')
    parser.add_argument('--store_id', type=str, default=None, help='Specific store ID to train on')
    parser.add_argument('--item_id', type=str, default=None, help='Specific item ID to train on')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--quick_train', action='store_true', help='Quick training (3 stores, ~3 min)')
    parser.add_argument('--full_train', action='store_true', help='Full training (all stores, ~10 min)')
    
    args = parser.parse_args()
    
    # Default to full train if no option specified
    if not args.quick_train and not args.full_train:
        print("No training mode specified. Using --full_train by default...")
        args.full_train = True
    
    print_banner()
    
    # Training mode info
    if args.quick_train:
        print("🚀 QUICK TRAINING MODE")
        print("   • Training on 3 representative stores")
        print("   • Essential features only")
        print("   • Estimated time: 3-5 minutes")
    else:
        print("🎯 FULL TRAINING MODE")
        print("   • Training on all available stores")
        print("   • 100+ feature engineering pipeline")
        print("   • Estimated time: 8-12 minutes")
    
    print(f"\n📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Output Directory: {args.output_dir}")
    print()
    
    overall_start_time = time.time()
    
    try:
        # Step 1: Initialize and verify data
        print("� STEP 1/4: Data Verification")
        print("-" * 50)
        
        step_start = time.time()
        print_progress_bar(0, "Loading data overview...", 0)
        
        sales_df = load_sales_data()
        calendar_df = load_calendar_data()
        prices_df = load_prices_data()
        stores_states = get_available_stores_and_states(sales_df)
        
        print_progress_bar(1, "Data verification complete ✓", time.time() - step_start)
        
        print(f"\n✅ Sales records: {len(sales_df):,}")
        print(f"✅ Calendar days: {len(calendar_df):,}")
        print(f"✅ Price records: {len(prices_df):,}")
        print(f"✅ Available stores: {len(stores_states)}")
        
        # Validate store_id if provided
        if args.store_id and args.store_id not in stores_states:
            print(f"\n❌ Error: Store ID '{args.store_id}' not found")
            print(f"Available stores: {list(stores_states.keys())[:10]}...")
            return 1
        
        # Step 2: Setup training parameters
        print(f"\n🔄 STEP 2/4: Training Setup")
        print("-" * 50)
        
        if args.quick_train:
            # Select representative stores for quick training
            store_list = ['CA_1', 'TX_1', 'WI_1']  # Different states
            print(f"🏪 Selected stores for quick training: {store_list}")
        else:
            store_list = [args.store_id] if args.store_id else [None]
            print(f"🏪 Training scope: {'Specific store' if args.store_id else 'All stores'}")
        
        # Step 3: Model Training
        print(f"\n🔄 STEP 3/4: ML Model Training")
        print("-" * 50)
        
        training_results = {}
        total_stores = len(store_list)
        
        for i, store_id in enumerate(store_list):
            store_start_time = time.time()
            
            if store_id:
                print(f"\n📍 Training Store {i+1}/{total_stores}: {store_id}")
            else:
                print(f"\n🏭 Training All Stores")
            
            print("-" * 30)
            
            # Initialize ML loader
            ml_loader = MLEnhancedDataLoader()
            
            # Progress callback for this store
            def progress_callback(progress, message):
                store_elapsed = time.time() - store_start_time
                print_progress_bar(progress, message, store_elapsed)
            
            # Train models for this store
            try:
                result = ml_loader.train_ml_models(
                    store_id=store_id,
                    item_id=args.item_id,
                    save_models=False,  # We'll save manually
                    use_streamlit=False
                )
                
                store_time = time.time() - store_start_time
                print(f"\n✅ Training completed in {timedelta(seconds=int(store_time))}")
                
                # Store results
                store_key = store_id if store_id else 'all_stores'
                training_results[store_key] = {
                    'result': result,
                    'training_time': store_time,
                    'ml_loader': ml_loader
                }
                
            except Exception as e:
                print(f"\n❌ Training failed for {store_id}: {str(e)}")
                continue
        
        if not training_results:
            print("\n❌ No models were successfully trained!")
            return 1
        
        # Step 4: Save Models and Generate Report
        print(f"\n🔄 STEP 4/4: Saving Models & Generating Report")
        print("-" * 50)
        
        save_start_time = time.time()
        print_progress_bar(0, "Preparing to save models...", 0)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save models from the best performing or last trained model
        best_ml_loader = list(training_results.values())[-1]['ml_loader']
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_suffix = "_quick" if args.quick_train else "_full"
        store_suffix = f"_store_{args.store_id}" if args.store_id else ""
        
        # Save with timestamp
        timestamped_file = os.path.join(args.output_dir, f"ml_models_{timestamp}{mode_suffix}{store_suffix}.pkl")
        best_ml_loader.save_models(timestamped_file)
        
        print_progress_bar(0.5, "Saving timestamped models...", time.time() - save_start_time)
        
        # Save as latest for easy loading
        latest_file = os.path.join(args.output_dir, 'latest_trained_models.pkl')
        best_ml_loader.save_models(latest_file)
        
        print_progress_bar(1, "Model saving complete ✓", time.time() - save_start_time)
        
        # Generate comprehensive report
        total_time = time.time() - overall_start_time
        
        print(f"\n" + "=" * 70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\n⏱️  TIMING SUMMARY:")
        print(f"   Total Training Time: {timedelta(seconds=int(total_time))}")
        print(f"   Average per Store: {timedelta(seconds=int(total_time/len(training_results)))}")
        print(f"   Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n� SAVED FILES:")
        print(f"   📁 Directory: {args.output_dir}/")
        print(f"   📄 Timestamped: {os.path.basename(timestamped_file)}")
        print(f"   📄 Latest: {os.path.basename(latest_file)}")
        
        print(f"\n� TRAINING SUMMARY:")
        total_accuracy = 0
        total_r2 = 0
        model_count = 0
        
        for store_name, data in training_results.items():
            print(f"   🏪 {store_name}:")
            result = data['result']
            
            if 'training_results' in result:
                for model_name, metrics in result['training_results'].items():
                    if 'accuracy' in metrics:
                        acc = metrics['accuracy']
                        total_accuracy += acc
                        model_count += 1
                        print(f"      • {model_name}: {acc:.1%} accuracy")
                    
                    if 'test_r2' in metrics:
                        r2 = metrics['test_r2']
                        total_r2 += r2
                        print(f"      • {model_name}: R² = {r2:.3f}")
        
        if model_count > 0:
            avg_accuracy = total_accuracy / model_count
            print(f"\n🎯 AVERAGE PERFORMANCE:")
            print(f"   � Average Accuracy: {avg_accuracy:.1%}")
            if total_r2 > 0:
                print(f"   📈 Average R²: {total_r2/model_count:.3f}")
        
        print(f"\n🚀 NEXT STEPS:")
        print("   1. Run: streamlit run app.py")
        print("   2. Navigate to: 🤖 Advanced ML Analytics")
        print("   3. Models will load instantly from saved files!")
        
        print("\n" + "=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {str(e)}")
        print(f"\n🔧 TROUBLESHOOTING:")
        print("   1. Check data files exist in data/ directory")
        print("   2. Ensure sufficient memory (4GB+ recommended)")
        print("   3. Use --quick_train for faster testing (3 stores)")
        print("   4. Verify store_id and item_id are valid")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
