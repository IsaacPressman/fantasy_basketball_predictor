"""
Linear Regression Models
Train linear models with regularization (Ridge, Lasso, ElasticNet) for each target stat
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

class LinearModels:
    """Train and evaluate linear regression models"""
    
    def __init__(self, data_dir='data/processed', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(os.path.join(models_dir, 'linear'), exist_ok=True)
        
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
    
    def train_ridge_model(self, X_train, y_train, target):
        """
        Train Ridge regression model with cross-validation
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training target
            target (str): Target column name
            
        Returns:
            Ridge model
        """
        print(f"\n  Training Ridge model for {target}...")
        
        # Define parameter grid
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
        
        # Create model
        ridge = Ridge(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            ridge, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"    Best alpha: {grid_search.best_params_['alpha']}")
        
        return grid_search.best_estimator_
    
    def train_lasso_model(self, X_train, y_train, target):
        """
        Train Lasso regression model with cross-validation
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training target
            target (str): Target column name
            
        Returns:
            Lasso model
        """
        print(f"\n  Training Lasso model for {target}...")
        
        # Define parameter grid
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
        
        # Create model
        lasso = Lasso(random_state=42, max_iter=10000)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            lasso, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"    Best alpha: {grid_search.best_params_['alpha']}")
        
        return grid_search.best_estimator_
    
    def train_elasticnet_model(self, X_train, y_train, target):
        """
        Train ElasticNet regression model with cross-validation
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training target
            target (str): Target column name
            
        Returns:
            ElasticNet model
        """
        print(f"\n  Training ElasticNet model for {target}...")
        
        # Define parameter grid
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95]
        }
        
        # Create model
        elasticnet = ElasticNet(random_state=42, max_iter=10000)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            elasticnet, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"    Best alpha: {grid_search.best_params_['alpha']}, "
              f"l1_ratio: {grid_search.best_params_['l1_ratio']}")
        
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """
        Train all three model types for each target
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            dict: Trained models and metrics
        """
        print("\n" + "="*60)
        print("Training Linear Models")
        print("="*60)
        
        all_models = {}
        all_metrics = {}
        
        for target in self.target_cols:
            if target not in y_train.columns:
                print(f"\n⚠ Skipping {target} (not in training data)")
                continue
            
            print(f"\n{'='*60}")
            print(f"Target: {target}")
            print(f"{'='*60}")
            
            models = {}
            metrics = {}
            
            # Train Ridge
            ridge_model = self.train_ridge_model(X_train, y_train[target], target)
            models['Ridge'] = ridge_model
            
            # Train Lasso
            lasso_model = self.train_lasso_model(X_train, y_train[target], target)
            models['Lasso'] = lasso_model
            
            # Train ElasticNet
            elasticnet_model = self.train_elasticnet_model(X_train, y_train[target], target)
            models['ElasticNet'] = elasticnet_model
            
            # Evaluate all models on validation set
            print(f"\n  Evaluating on validation set...")
            for model_name, model in models.items():
                y_pred = model.predict(X_val)
                
                mae = mean_absolute_error(y_val[target], y_pred)
                rmse = np.sqrt(mean_squared_error(y_val[target], y_pred))
                r2 = r2_score(y_val[target], y_pred)
                
                metrics[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                }
                
                print(f"    {model_name:12s}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
            
            # Select best model based on MAE
            best_model_name = min(metrics, key=lambda x: metrics[x]['MAE'])
            best_model = models[best_model_name]
            
            print(f"\n  ✓ Best model: {best_model_name}")
            
            all_models[target] = {
                'models': models,
                'best_model': best_model,
                'best_model_name': best_model_name
            }
            all_metrics[target] = metrics
        
        return all_models, all_metrics
    
    def save_models(self, all_models):
        """Save trained models"""
        print("\n" + "="*60)
        print("Saving Models")
        print("="*60)
        
        models_dir = os.path.join(self.models_dir, 'linear')
        
        for target, model_dict in all_models.items():
            # Save best model
            best_model = model_dict['best_model']
            best_name = model_dict['best_model_name']
            
            model_path = os.path.join(models_dir, f'{target}_linear.pkl')
            joblib.dump(best_model, model_path)
            
            print(f"✓ Saved {target} ({best_name}) to {model_path}")
        
        print(f"\n✓ All models saved to: {models_dir}")
    
    def compare_to_baseline(self, linear_metrics):
        """
        Compare linear models to baseline
        
        Args:
            linear_metrics (dict): Metrics from linear models
        """
        print("\n" + "="*60)
        print("COMPARISON TO BASELINE")
        print("="*60)
        
        # Load baseline metrics
        baseline_path = os.path.join(self.models_dir, 'baseline_results', 'baseline_metrics.json')
        
        if not os.path.exists(baseline_path):
            print("⚠ Baseline metrics not found, skipping comparison")
            return
        
        with open(baseline_path, 'r') as f:
            baseline_metrics = json.load(f)
        
        # Get best baseline (3-Year Average)
        baseline = baseline_metrics.get('3-Year Average', {})
        
        # Create comparison table
        comparison = []
        
        for target in self.target_cols:
            if target not in linear_metrics:
                continue
            
            # Get best linear model metrics
            linear_best = min(linear_metrics[target].values(), key=lambda x: x['MAE'])
            
            row = {
                'Target': target.replace('NEXT_', ''),
                'Baseline_MAE': baseline.get(target, {}).get('MAE', np.nan),
                'Linear_MAE': linear_best['MAE'],
                'Improvement': np.nan
            }
            
            if not np.isnan(row['Baseline_MAE']):
                improvement = ((row['Baseline_MAE'] - row['Linear_MAE']) / row['Baseline_MAE']) * 100
                row['Improvement'] = improvement
            
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        
        print("\nMAE Comparison (Lower is better):")
        print(comparison_df.to_string(index=False))
        
        # Summary
        avg_improvement = comparison_df['Improvement'].mean()
        wins = (comparison_df['Improvement'] > 0).sum()
        total = len(comparison_df)
        
        print(f"\nSummary:")
        print(f"  Linear models beat baseline: {wins}/{total} targets")
        print(f"  Average improvement: {avg_improvement:.1f}%")
        
        if avg_improvement > 0:
            print(f"\n✓ Linear models outperform baseline!")
        else:
            print(f"\n⚠ Linear models underperform baseline (consider more features/tuning)")
    
    def save_results(self, all_metrics):
        """Save results to file"""
        results_dir = os.path.join(self.models_dir, 'linear_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(results_dir, 'linear_metrics.json')
        
        # Convert to serializable format
        serializable_metrics = {}
        for target, models_metrics in all_metrics.items():
            serializable_metrics[target] = {}
            for model_name, metrics in models_metrics.items():
                serializable_metrics[target][model_name] = {
                    k: float(v) for k, v in metrics.items()
                }
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\n✓ Saved linear model results to: {metrics_path}")
        
        # Create summary report
        summary_path = os.path.join(results_dir, 'linear_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("LINEAR REGRESSION MODEL RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for target, models_metrics in all_metrics.items():
                f.write(f"\n{target}:\n")
                f.write("-" * 40 + "\n")
                
                for model_name, metrics in models_metrics.items():
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  MAE:  {metrics['MAE']:.3f}\n")
                    f.write(f"  RMSE: {metrics['RMSE']:.3f}\n")
                    f.write(f"  R²:   {metrics['R2']:.3f}\n")
                
                # Best model
                best_model = min(models_metrics, key=lambda x: models_metrics[x]['MAE'])
                f.write(f"\nBest: {best_model}\n")
        
        print(f"✓ Saved summary report to: {summary_path}")


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Projections - Linear Regression Models")
    print("="*60)
    
    # Initialize
    linear_models = LinearModels()
    
    # Load data
    X_train, y_train, X_val, y_val = linear_models.load_prepared_data()
    
    # Train all models
    all_models, all_metrics = linear_models.train_all_models(X_train, y_train, X_val, y_val)
    
    # Save models
    linear_models.save_models(all_models)
    
    # Compare to baseline
    linear_models.compare_to_baseline(all_metrics)
    
    # Save results
    linear_models.save_results(all_metrics)
    
    print("\n" + "="*60)
    print("✓ STEP 17 COMPLETE!")
    print("="*60)
    print("\nLinear models trained with Ridge, Lasso, and ElasticNet")
    print("Best performing model selected for each target stat")
    print("\nNext: Step 18 - Random Forest Models")
    
    return all_models, all_metrics


if __name__ == "__main__":
    # Run the main function
    models, metrics = main()