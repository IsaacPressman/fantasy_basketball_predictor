"""
Random Forest Models
Train Random Forest models with hyperparameter tuning for each target stat
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

class RandomForestModels:
    """Train and evaluate Random Forest models"""
    
    def __init__(self, data_dir='data/processed', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(os.path.join(models_dir, 'random_forest'), exist_ok=True)
        
        # Target columns
        self.target_cols = [
            'NEXT_PTS', 'NEXT_AST', 'NEXT_REB', 'NEXT_BLK', 
            'NEXT_STL', 'NEXT_TOV', 'NEXT_MIN', 'NEXT_FG3M',
            'NEXT_FGA', 'NEXT_FTA', 'NEXT_FG_PCT', 'NEXT_FT_PCT'
        ]
    
    def load_prepared_data(self):
        """Load prepared training and validation data"""
        print("="*60)
        print("Loading Prepared Data")
        print("="*60)
        
        prepared_dir = os.path.join(self.data_dir, 'prepared')
        
        # Load training data
        X_train = pd.read_csv(os.path.join(prepared_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(prepared_dir, 'y_train.csv'))
        
        # Load validation data
        X_val = pd.read_csv(os.path.join(prepared_dir, 'X_val.csv'))
        y_val = pd.read_csv(os.path.join(prepared_dir, 'y_val.csv'))
        
        print(f"✓ Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"✓ Validation: {X_val.shape[0]} samples, {X_val.shape[1]} features")
        print(f"✓ Targets: {y_train.shape[1]}")
        
        return X_train, y_train, X_val, y_val
    
    def train_random_forest(self, X_train, y_train, target):
        """
        Train Random Forest model with hyperparameter tuning
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training target
            target (str): Target column name
            
        Returns:
            tuple: (best_model, best_params, cv_results)
        """
        print(f"\nTraining Random Forest for {target}...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [10, 20, 50],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt', 'log2']
        }
        
        # Create model
        rf = RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        print(f"  Running grid search (this may take a few minutes)...")
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=3,  # 3-fold CV to save time
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  ✓ Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"    {param}: {value}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_
    
    def extract_feature_importance(self, model, feature_names, target, top_n=20):
        """
        Extract and display feature importances
        
        Args:
            model: Trained model
            feature_names (list): Feature names
            target (str): Target name
            top_n (int): Number of top features to display
            
        Returns:
            DataFrame: Feature importances
        """
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top {top_n} most important features for {target}:")
        for idx, row in feature_importance.head(top_n).iterrows():
            print(f"    {row['feature']:30s}: {row['importance']:.4f}")
        
        return feature_importance
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """
        Train Random Forest for all targets
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            dict: Trained models, metrics, and feature importances
        """
        print("\n" + "="*60)
        print("Training Random Forest Models")
        print("="*60)
        
        all_models = {}
        all_metrics = {}
        all_importances = {}
        
        for target in self.target_cols:
            if target not in y_train.columns:
                print(f"\n⚠ Skipping {target} (not in training data)")
                continue
            
            print(f"\n{'='*60}")
            print(f"Target: {target}")
            print(f"{'='*60}")
            
            # Train model
            model, best_params, cv_results = self.train_random_forest(
                X_train, y_train[target], target
            )
            
            # Extract feature importance
            feature_importance = self.extract_feature_importance(
                model, X_train.columns.tolist(), target
            )
            
            # Evaluate on validation set
            print(f"\n  Evaluating on validation set...")
            y_pred = model.predict(X_val)
            
            mae = mean_absolute_error(y_val[target], y_pred)
            rmse = np.sqrt(mean_squared_error(y_val[target], y_pred))
            r2 = r2_score(y_val[target], y_pred)
            
            print(f"    MAE:  {mae:.3f}")
            print(f"    RMSE: {rmse:.3f}")
            print(f"    R²:   {r2:.3f}")
            
            # Store results
            all_models[target] = model
            all_metrics[target] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'best_params': best_params
            }
            all_importances[target] = feature_importance
        
        return all_models, all_metrics, all_importances
    
    def save_models(self, all_models):
        """Save trained models"""
        print("\n" + "="*60)
        print("Saving Models")
        print("="*60)
        
        models_dir = os.path.join(self.models_dir, 'random_forest')
        
        for target, model in all_models.items():
            model_path = os.path.join(models_dir, f'{target}_rf.pkl')
            joblib.dump(model, model_path)
            print(f"✓ Saved {target} to {model_path}")
        
        print(f"\n✓ All models saved to: {models_dir}")
    
    def save_feature_importances(self, all_importances):
        """Save feature importances"""
        importances_dir = os.path.join(self.models_dir, 'random_forest', 'feature_importances')
        os.makedirs(importances_dir, exist_ok=True)
        
        for target, importance_df in all_importances.items():
            # Save to CSV
            csv_path = os.path.join(importances_dir, f'{target}_importance.csv')
            importance_df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Feature importances saved to: {importances_dir}")
    
    def compare_to_baseline_and_linear(self, rf_metrics):
        """
        Compare RF models to baseline and linear models
        
        Args:
            rf_metrics (dict): Metrics from RF models
        """
        print("\n" + "="*60)
        print("COMPARISON TO BASELINE AND LINEAR MODELS")
        print("="*60)
        
        # Load baseline metrics
        baseline_path = os.path.join(self.models_dir, 'baseline_results', 'baseline_metrics.json')
        baseline_metrics = {}
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                all_baselines = json.load(f)
                baseline_metrics = all_baselines.get('3-Year Average', {})
        
        # Load linear metrics
        linear_path = os.path.join(self.models_dir, 'linear_results', 'linear_metrics.json')
        linear_metrics = {}
        if os.path.exists(linear_path):
            with open(linear_path, 'r') as f:
                all_linear = json.load(f)
                # Get best linear model for each target
                for target, models in all_linear.items():
                    best_linear = min(models.values(), key=lambda x: x['MAE'])
                    linear_metrics[target] = best_linear
        
        # Create comparison table
        comparison = []
        
        for target in self.target_cols:
            if target not in rf_metrics:
                continue
            
            row = {
                'Target': target.replace('NEXT_', ''),
                'Baseline_MAE': baseline_metrics.get(target, {}).get('MAE', np.nan),
                'Linear_MAE': linear_metrics.get(target, {}).get('MAE', np.nan),
                'RF_MAE': rf_metrics[target]['MAE'],
                'RF_R2': rf_metrics[target]['R2']
            }
            
            # Calculate improvements
            if not np.isnan(row['Baseline_MAE']):
                row['vs_Baseline'] = ((row['Baseline_MAE'] - row['RF_MAE']) / row['Baseline_MAE']) * 100
            
            if not np.isnan(row['Linear_MAE']):
                row['vs_Linear'] = ((row['Linear_MAE'] - row['RF_MAE']) / row['Linear_MAE']) * 100
            
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        
        print("\nMAE Comparison (Lower is better):")
        print(comparison_df[['Target', 'Baseline_MAE', 'Linear_MAE', 'RF_MAE', 'RF_R2']].to_string(index=False))
        
        print("\nImprovement over previous models:")
        print(comparison_df[['Target', 'vs_Baseline', 'vs_Linear']].to_string(index=False))
        
        # Summary
        avg_vs_baseline = comparison_df['vs_Baseline'].mean()
        avg_vs_linear = comparison_df['vs_Linear'].mean()
        avg_r2 = comparison_df['RF_R2'].mean()
        
        print(f"\nSummary:")
        print(f"  Average improvement vs Baseline: {avg_vs_baseline:.1f}%")
        print(f"  Average improvement vs Linear:   {avg_vs_linear:.1f}%")
        print(f"  Average R²: {avg_r2:.3f}")
        
        if avg_vs_linear > 0:
            print(f"\n✓ Random Forest outperforms linear models!")
        else:
            print(f"\n⚠ Random Forest underperforms linear models")
    
    def save_results(self, all_metrics, all_importances):
        """Save results to file"""
        results_dir = os.path.join(self.models_dir, 'rf_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(results_dir, 'rf_metrics.json')
        
        # Convert to serializable format
        serializable_metrics = {}
        for target, metrics in all_metrics.items():
            serializable_metrics[target] = {
                'MAE': float(metrics['MAE']),
                'RMSE': float(metrics['RMSE']),
                'R2': float(metrics['R2']),
                'best_params': metrics['best_params']
            }
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\n✓ Saved RF model results to: {metrics_path}")
        
        # Create summary report
        summary_path = os.path.join(results_dir, 'rf_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("RANDOM FOREST MODEL RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for target, metrics in all_metrics.items():
                stat_name = target.replace('NEXT_', '')
                f.write(f"\n{stat_name}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  MAE:  {metrics['MAE']:.3f}\n")
                f.write(f"  RMSE: {metrics['RMSE']:.3f}\n")
                f.write(f"  R²:   {metrics['R2']:.3f}\n")
                f.write(f"\n  Best Parameters:\n")
                for param, value in metrics['best_params'].items():
                    f.write(f"    {param}: {value}\n")
                
                # Top 10 features
                if target in all_importances:
                    f.write(f"\n  Top 10 Features:\n")
                    top_features = all_importances[target].head(10)
                    for idx, row in top_features.iterrows():
                        f.write(f"    {row['feature']:30s}: {row['importance']:.4f}\n")
            
            # Overall summary
            avg_mae = np.mean([m['MAE'] for m in all_metrics.values()])
            avg_r2 = np.mean([m['R2'] for m in all_metrics.values()])
            
            f.write(f"\n{'='*60}\n")
            f.write(f"OVERALL SUMMARY\n")
            f.write(f"{'='*60}\n")
            f.write(f"Average MAE: {avg_mae:.3f}\n")
            f.write(f"Average R²:  {avg_r2:.3f}\n")
        
        print(f"✓ Saved summary report to: {summary_path}")


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Projections - Random Forest Models")
    print("="*60)
    
    # Initialize
    rf_models = RandomForestModels()
    
    # Load data
    X_train, y_train, X_val, y_val = rf_models.load_prepared_data()
    
    # Train all models
    all_models, all_metrics, all_importances = rf_models.train_all_models(
        X_train, y_train, X_val, y_val
    )
    
    # Save models
    rf_models.save_models(all_models)
    
    # Save feature importances
    rf_models.save_feature_importances(all_importances)
    
    # Compare to baseline and linear
    rf_models.compare_to_baseline_and_linear(all_metrics)
    
    # Save results
    rf_models.save_results(all_metrics, all_importances)
    
    print("\n" + "="*60)
    print("✓ STEP 18 COMPLETE!")
    print("="*60)
    print("\nRandom Forest models trained with hyperparameter tuning")
    print("Feature importances extracted for each target")
    print("\nNext: Step 19 - XGBoost/LightGBM Models (The powerhouses!)")
    
    return all_models, all_metrics, all_importances


if __name__ == "__main__":
    # Run the main function
    models, metrics, importances = main()