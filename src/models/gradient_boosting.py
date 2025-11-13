"""
XGBoost Models
Train XGBoost models with hyperparameter tuning for each target stat
"""

import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

class XGBoostModels:
    """Train and evaluate XGBoost models"""
    
    def __init__(self, data_dir='data/processed', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(os.path.join(models_dir, 'xgboost'), exist_ok=True)
        
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
    
    def train_xgboost(self, X_train, y_train, X_val, y_val_target, target):
        """
        Train XGBoost model with hyperparameter tuning
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training target
            X_val (DataFrame): Validation features (for early stopping)
            y_val_target (Series): Validation target (for early stopping)
            target (str): Target column name
            
        Returns:
            tuple: (best_model, best_params)
        """
        print(f"\nTraining XGBoost for {target}...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1]
        }
        
        # Create model
        xgb_model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # Faster training
        )
        
        # Grid search with cross-validation
        print(f"  Running grid search with early stopping...")
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val_target)],
            verbose=False
        )
        
        print(f"  ✓ Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"    {param}: {value}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
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
        Train XGBoost for all targets
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            dict: Trained models, metrics, and feature importances
        """
        print("\n" + "="*60)
        print("Training XGBoost Models")
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
            
            # Train model with early stopping
            model, best_params = self.train_xgboost(
                X_train, y_train[target], X_val, y_val[target], target
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
        
        models_dir = os.path.join(self.models_dir, 'xgboost')
        
        for target, model in all_models.items():
            model_path = os.path.join(models_dir, f'{target}_xgb.pkl')
            joblib.dump(model, model_path)
            print(f"✓ Saved {target} to {model_path}")
        
        print(f"\n✓ All models saved to: {models_dir}")
    
    def save_feature_importances(self, all_importances):
        """Save feature importances"""
        importances_dir = os.path.join(self.models_dir, 'xgboost', 'feature_importances')
        os.makedirs(importances_dir, exist_ok=True)
        
        for target, importance_df in all_importances.items():
            csv_path = os.path.join(importances_dir, f'{target}_importance.csv')
            importance_df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Feature importances saved to: {importances_dir}")
    
    def compare_all_models(self, xgb_metrics):
        """
        Compare XGBoost to all previous models
        
        Args:
            xgb_metrics (dict): Metrics from XGBoost models
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*60)
        
        # Load all previous metrics
        baseline_metrics = self._load_baseline_metrics()
        linear_metrics = self._load_linear_metrics()
        rf_metrics = self._load_rf_metrics()
        
        # Create comparison table
        comparison = []
        
        for target in self.target_cols:
            if target not in xgb_metrics:
                continue
            
            row = {
                'Target': target.replace('NEXT_', ''),
                'Baseline': baseline_metrics.get(target, {}).get('MAE', np.nan),
                'Linear': linear_metrics.get(target, {}).get('MAE', np.nan),
                'RF': rf_metrics.get(target, {}).get('MAE', np.nan),
                'XGBoost': xgb_metrics[target]['MAE'],
                'XGB_R2': xgb_metrics[target]['R2']
            }
            
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        
        print("\nMAE Comparison Across All Models (Lower is better):")
        print(comparison_df.to_string(index=False))
        
        # Find best model for each target
        print("\n" + "="*60)
        print("BEST MODEL PER TARGET")
        print("="*60)
        
        for _, row in comparison_df.iterrows():
            target = row['Target']
            models_mae = {
                'Baseline': row['Baseline'],
                'Linear': row['Linear'],
                'RF': row['RF'],
                'XGBoost': row['XGBoost']
            }
            
            # Remove NaN values
            models_mae = {k: v for k, v in models_mae.items() if not np.isnan(v)}
            
            if models_mae:
                best_model = min(models_mae, key=models_mae.get)
                best_mae = models_mae[best_model]
                print(f"{target:10s}: {best_model:10s} (MAE: {best_mae:.3f})")
        
        # Overall summary
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        
        avg_xgb_mae = comparison_df['XGBoost'].mean()
        avg_xgb_r2 = comparison_df['XGB_R2'].mean()
        
        print(f"\nXGBoost Performance:")
        print(f"  Average MAE: {avg_xgb_mae:.3f}")
        print(f"  Average R²:  {avg_xgb_r2:.3f}")
        
        # Count wins
        xgb_wins = 0
        for _, row in comparison_df.iterrows():
            models_mae = [row['Baseline'], row['Linear'], row['RF'], row['XGBoost']]
            models_mae = [m for m in models_mae if not np.isnan(m)]
            if models_mae and row['XGBoost'] == min(models_mae):
                xgb_wins += 1
        
        print(f"\nXGBoost wins: {xgb_wins}/{len(comparison_df)} targets")
        
        if xgb_wins >= len(comparison_df) * 0.7:
            print("\n🎉 XGBoost is the clear winner!")
        elif xgb_wins >= len(comparison_df) * 0.5:
            print("\n✓ XGBoost performs well overall")
        else:
            print("\n⚠ XGBoost doesn't dominate - consider ensemble approach")
    
    def _load_baseline_metrics(self):
        """Load baseline metrics"""
        path = os.path.join(self.models_dir, 'baseline_results', 'baseline_metrics.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                all_baselines = json.load(f)
                return all_baselines.get('3-Year Average', {})
        return {}
    
    def _load_linear_metrics(self):
        """Load linear model metrics"""
        path = os.path.join(self.models_dir, 'linear_results', 'linear_metrics.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                all_linear = json.load(f)
                # Get best for each target
                metrics = {}
                for target, models in all_linear.items():
                    best = min(models.values(), key=lambda x: x['MAE'])
                    metrics[target] = best
                return metrics
        return {}
    
    def _load_rf_metrics(self):
        """Load RF metrics"""
        path = os.path.join(self.models_dir, 'rf_results', 'rf_metrics.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_results(self, all_metrics, all_importances):
        """Save results to file"""
        results_dir = os.path.join(self.models_dir, 'xgb_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(results_dir, 'xgb_metrics.json')
        
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
        
        print(f"\n✓ Saved XGBoost results to: {metrics_path}")


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Projections - XGBoost Models")
    print("="*60)
    
    # Initialize
    xgb_models = XGBoostModels()
    
    # Load data
    X_train, y_train, X_val, y_val = xgb_models.load_prepared_data()
    
    # Train all models
    all_models, all_metrics, all_importances = xgb_models.train_all_models(
        X_train, y_train, X_val, y_val
    )
    
    # Save models
    xgb_models.save_models(all_models)
    
    # Save feature importances
    xgb_models.save_feature_importances(all_importances)
    
    # Compare to all previous models
    xgb_models.compare_all_models(all_metrics)
    
    # Save results
    xgb_models.save_results(all_metrics, all_importances)
    
    print("\n" + "="*60)
    print("✓ STEP 19 COMPLETE!")
    print("="*60)
    print("\nXGBoost models trained - should be the best performers!")
    print("Next: Step 20 - Ensemble Model (combine all approaches)")
    
    return all_models, all_metrics, all_importances


if __name__ == "__main__":
    # Run the main function
    models, metrics, importances = main()