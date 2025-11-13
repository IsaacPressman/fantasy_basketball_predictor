"""
Baseline Models
Establishes simple baseline predictions to benchmark ML models against
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

class BaselineModels:
    """Create and evaluate baseline prediction models"""
    
    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
        
        # Target columns we're predicting
        self.target_cols = [
            'NEXT_PTS', 'NEXT_AST', 'NEXT_REB', 'NEXT_BLK', 
            'NEXT_STL', 'NEXT_TOV', 'NEXT_MIN', 'NEXT_FG3M',
            'NEXT_FGA', 'NEXT_FTA', 'NEXT_FG_PCT', 'NEXT_FT_PCT'
        ]
        
        # Corresponding current stat columns (what we use to predict)
        self.current_stat_cols = [
            'PTS', 'AST', 'REB', 'BLK', 
            'STL', 'TOV', 'MIN', 'FG3M',
            'FGA', 'FTA', 'FG_PCT', 'FT_PCT'
        ]
    
    def load_data(self):
        """Load validation data and metadata"""
        print("="*60)
        print("Loading Data for Baseline Evaluation")
        print("="*60)
        
        # Load validation targets
        y_val_path = os.path.join(self.data_dir, 'prepared', 'y_val.csv')
        if not os.path.exists(y_val_path):
            print(f"✗ Validation targets not found at {y_val_path}")
            return None, None, None
        
        y_val = pd.read_csv(y_val_path)
        print(f"✓ Loaded validation targets: {y_val.shape}")
        
        # Load validation metadata (contains current stats)
        metadata_val_path = os.path.join(self.data_dir, 'prepared', 'metadata_val.csv')
        metadata_val = pd.read_csv(metadata_val_path) if os.path.exists(metadata_val_path) else None
        
        # Load full training data to get current stats and position info
        train_path = os.path.join(self.data_dir, 'final_training_data.csv')
        full_train = pd.read_csv(train_path) if os.path.exists(train_path) else None
        
        # Load validation data (with current stats)
        val_path = os.path.join(self.data_dir, 'final_training_data.csv')
        if os.path.exists(val_path):
            full_data = pd.read_csv(val_path)
            # Get only validation season (2023-24)
            val_data = full_data[full_data['SEASON'] == '2023-24'].copy()
            print(f"✓ Loaded validation data with features: {val_data.shape}")
        else:
            val_data = None
        
        return y_val, val_data, full_train
    
    def baseline_last_season(self, val_data, y_val):
        """
        Baseline 1: Predict same as last season (LAG1)
        
        Args:
            val_data (DataFrame): Validation data with current stats
            y_val (DataFrame): Actual target values
            
        Returns:
            DataFrame: Predictions
        """
        print("\n" + "="*60)
        print("Baseline 1: Last Season's Stats")
        print("="*60)
        
        predictions = pd.DataFrame(index=y_val.index)
        
        for target, current in zip(self.target_cols, self.current_stat_cols):
            if target in y_val.columns and current in val_data.columns:
                # Use LAG1 if available, otherwise use current season
                lag_col = f'{current}_LAG1'
                
                if lag_col in val_data.columns:
                    predictions[target] = val_data[lag_col].values
                    print(f"  {target}: Using {lag_col}")
                else:
                    # Fallback: use current season stats
                    predictions[target] = val_data[current].values
                    print(f"  {target}: Using {current} (LAG1 not available)")
        
        print(f"\n✓ Created predictions using last season's stats")
        return predictions
    
    def baseline_three_year_average(self, val_data, y_val):
        """
        Baseline 2: Predict 3-year average
        
        Args:
            val_data (DataFrame): Validation data
            y_val (DataFrame): Actual target values
            
        Returns:
            DataFrame: Predictions
        """
        print("\n" + "="*60)
        print("Baseline 2: 3-Year Average")
        print("="*60)
        
        predictions = pd.DataFrame(index=y_val.index)
        
        for target, current in zip(self.target_cols, self.current_stat_cols):
            if target in y_val.columns:
                # Try to use 3-year average column if it exists
                avg3_col = f'{current}_AVG3'
                avg2_col = f'{current}_AVG2'
                lag1_col = f'{current}_LAG1'
                
                if avg3_col in val_data.columns:
                    predictions[target] = val_data[avg3_col].values
                    print(f"  {target}: Using {avg3_col}")
                elif avg2_col in val_data.columns:
                    predictions[target] = val_data[avg2_col].values
                    print(f"  {target}: Using {avg2_col} (3-yr not available)")
                elif lag1_col in val_data.columns:
                    predictions[target] = val_data[lag1_col].values
                    print(f"  {target}: Using {lag1_col} (multi-year avg not available)")
                elif current in val_data.columns:
                    predictions[target] = val_data[current].values
                    print(f"  {target}: Using {current} (fallback)")
                else:
                    # Last resort: use median from training data
                    predictions[target] = y_val[target].median()
                    print(f"  {target}: Using median (no historical data)")
        
        print(f"\n✓ Created predictions using multi-year averages")
        return predictions
    
    def baseline_position_average(self, val_data, full_train, y_val):
        """
        Baseline 3: Predict league average by position
        
        Args:
            val_data (DataFrame): Validation data
            full_train (DataFrame): Full training data
            y_val (DataFrame): Actual target values
            
        Returns:
            DataFrame: Predictions
        """
        print("\n" + "="*60)
        print("Baseline 3: Position Average")
        print("="*60)
        
        predictions = pd.DataFrame(index=y_val.index)
        
        if full_train is None or 'POSITION' not in val_data.columns:
            print("⚠ Position data not available, using overall average")
            for target in self.target_cols:
                if target in y_val.columns:
                    predictions[target] = y_val[target].median()
            return predictions
        
        # Calculate position averages from training data (2020-23)
        train_seasons = ['2020-21', '2021-22', '2022-23']
        train_data = full_train[full_train['SEASON'].isin(train_seasons)].copy()
        
        # For each target, calculate position averages
        position_avgs = {}
        for target, current in zip(self.target_cols, self.current_stat_cols):
            if current in train_data.columns and 'POSITION' in train_data.columns:
                position_avgs[target] = train_data.groupby('POSITION')[current].mean().to_dict()
        
        # Apply position averages to validation data
        for target in self.target_cols:
            if target in y_val.columns and target in position_avgs:
                # Map position to average
                predictions[target] = val_data['POSITION'].map(position_avgs[target])
                
                # Fill any missing positions with overall average
                overall_avg = y_val[target].median()
                predictions[target] = predictions[target].fillna(overall_avg)
                
                print(f"  {target}: Using position averages")
        
        print(f"\n✓ Created predictions using position averages")
        return predictions
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """
        Calculate evaluation metrics
        
        Args:
            y_true (DataFrame): Actual values
            y_pred (DataFrame): Predicted values
            model_name (str): Name of the model
            
        Returns:
            dict: Metrics for each target
        """
        metrics = {}
        
        for target in self.target_cols:
            if target not in y_true.columns or target not in y_pred.columns:
                continue
            
            # Remove any NaN values
            mask = ~(y_true[target].isna() | y_pred[target].isna())
            y_t = y_true[target][mask]
            y_p = y_pred[target][mask]
            
            if len(y_t) == 0:
                continue
            
            # Calculate metrics
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            r2 = r2_score(y_t, y_p)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero
            mape = np.mean(np.abs((y_t - y_p) / y_t.replace(0, np.nan))) * 100
            
            metrics[target] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
        
        return metrics
    
    def compare_baselines(self, metrics_dict):
        """
        Compare all baseline models
        
        Args:
            metrics_dict (dict): Dictionary of metrics for each model
        """
        print("\n" + "="*60)
        print("BASELINE MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        comparison = []
        
        for target in self.target_cols:
            row = {'Target': target.replace('NEXT_', '')}
            
            for model_name, metrics in metrics_dict.items():
                if target in metrics:
                    row[f'{model_name}_MAE'] = metrics[target]['MAE']
            
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        print("\nMean Absolute Error (MAE) Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        for model_name, metrics in metrics_dict.items():
            print(f"\n{model_name}:")
            
            all_maes = [m['MAE'] for m in metrics.values()]
            all_r2s = [m['R2'] for m in metrics.values()]
            
            print(f"  Average MAE: {np.mean(all_maes):.3f}")
            print(f"  Average R²: {np.mean(all_r2s):.3f}")
            print(f"  MAE Range: {np.min(all_maes):.3f} to {np.max(all_maes):.3f}")
    
    def save_baseline_results(self, metrics_dict):
        """Save baseline results to file"""
        
        results_dir = os.path.join('models', 'baseline_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(results_dir, 'baseline_metrics.json')
        
        # Convert to serializable format
        serializable_metrics = {}
        for model_name, metrics in metrics_dict.items():
            serializable_metrics[model_name] = {}
            for target, target_metrics in metrics.items():
                serializable_metrics[model_name][target] = {
                    k: float(v) for k, v in target_metrics.items()
                }
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\n✓ Saved baseline results to: {metrics_path}")
        
        # Create summary report
        summary_path = os.path.join(results_dir, 'baseline_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("BASELINE MODEL RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for model_name, metrics in metrics_dict.items():
                f.write(f"\n{model_name}:\n")
                f.write("-" * 40 + "\n")
                
                for target, target_metrics in metrics.items():
                    stat_name = target.replace('NEXT_', '')
                    f.write(f"\n{stat_name}:\n")
                    f.write(f"  MAE:  {target_metrics['MAE']:.3f}\n")
                    f.write(f"  RMSE: {target_metrics['RMSE']:.3f}\n")
                    f.write(f"  R²:   {target_metrics['R2']:.3f}\n")
                    f.write(f"  MAPE: {target_metrics['MAPE']:.1f}%\n")
                
                # Average metrics
                all_maes = [m['MAE'] for m in metrics.values()]
                all_r2s = [m['R2'] for m in metrics.values()]
                
                f.write(f"\nAverage Metrics:\n")
                f.write(f"  MAE: {np.mean(all_maes):.3f}\n")
                f.write(f"  R²:  {np.mean(all_r2s):.3f}\n")
        
        print(f"✓ Saved summary report to: {summary_path}")
        
        return results_dir


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Projections - Baseline Models")
    print("="*60)
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    # Load data
    y_val, val_data, full_train = baseline.load_data()
    
    if y_val is None or val_data is None:
        print("\n✗ Failed to load data")
        return None
    
    print(f"\nValidation samples: {len(y_val)}")
    print(f"Targets to predict: {len(baseline.target_cols)}")
    
    # Create baseline predictions
    baseline1_preds = baseline.baseline_last_season(val_data, y_val)
    baseline2_preds = baseline.baseline_three_year_average(val_data, y_val)
    baseline3_preds = baseline.baseline_position_average(val_data, full_train, y_val)
    
    # Calculate metrics for each baseline
    print("\n" + "="*60)
    print("Evaluating Baseline Models")
    print("="*60)
    
    metrics1 = baseline.calculate_metrics(y_val, baseline1_preds, "Last Season")
    print("\n✓ Evaluated Baseline 1: Last Season")
    
    metrics2 = baseline.calculate_metrics(y_val, baseline2_preds, "3-Year Average")
    print("✓ Evaluated Baseline 2: 3-Year Average")
    
    metrics3 = baseline.calculate_metrics(y_val, baseline3_preds, "Position Average")
    print("✓ Evaluated Baseline 3: Position Average")
    
    # Compare baselines
    metrics_dict = {
        'Last Season': metrics1,
        '3-Year Average': metrics2,
        'Position Average': metrics3
    }
    
    baseline.compare_baselines(metrics_dict)
    
    # Save results
    results_dir = baseline.save_baseline_results(metrics_dict)
    
    print("="*60)
    print("\nBaseline Performance Summary:")
    print(f"  Best overall: Check the comparison table above")
    print(f"  These are the benchmarks the ML models must beat!")

    
    return metrics_dict


if __name__ == "__main__":
    # Run the main function
    metrics = main()