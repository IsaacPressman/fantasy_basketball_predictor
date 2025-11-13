"""
Ensemble Model
Combines predictions from Linear, Random Forest, and XGBoost models
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

class EnsembleModel:
    """Create and evaluate ensemble models"""
    
    def __init__(self, data_dir='data/processed', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(os.path.join(models_dir, 'ensemble'), exist_ok=True)
        
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
        
        X_train = pd.read_csv(os.path.join(prepared_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(prepared_dir, 'y_train.csv'))
        X_val = pd.read_csv(os.path.join(prepared_dir, 'X_val.csv'))
        y_val = pd.read_csv(os.path.join(prepared_dir, 'y_val.csv'))
        
        print(f"✓ Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"✓ Validation: {X_val.shape[0]} samples")
        
        return X_train, y_train, X_val, y_val
    
    def load_all_models(self, target):
        """
        Load trained models for a specific target
        
        Args:
            target (str): Target column name
            
        Returns:
            dict: Dictionary of loaded models
        """
        models = {}
        
        # Load Linear model
        linear_path = os.path.join(self.models_dir, 'linear', f'{target}_linear.pkl')
        if os.path.exists(linear_path):
            models['Linear'] = joblib.load(linear_path)
        
        # Load Random Forest model
        rf_path = os.path.join(self.models_dir, 'random_forest', f'{target}_rf.pkl')
        if os.path.exists(rf_path):
            models['RF'] = joblib.load(rf_path)
        
        # Load XGBoost model
        xgb_path = os.path.join(self.models_dir, 'xgboost', f'{target}_xgb.pkl')
        if os.path.exists(xgb_path):
            models['XGBoost'] = joblib.load(xgb_path)
        
        return models
    
    def get_individual_predictions(self, models, X):
        """
        Get predictions from all individual models
        
        Args:
            models (dict): Dictionary of models
            X (DataFrame): Features
            
        Returns:
            DataFrame: Predictions from each model
        """
        predictions = pd.DataFrame()
        
        for model_name, model in models.items():
            predictions[model_name] = model.predict(X)
        
        return predictions
    
    def simple_average_ensemble(self, predictions):
        """
        Simple average of all predictions
        
        Args:
            predictions (DataFrame): Predictions from each model
            
        Returns:
            array: Ensemble predictions
        """
        return predictions.mean(axis=1).values
    
    def weighted_average_ensemble(self, predictions, weights):
        """
        Weighted average based on validation performance
        
        Args:
            predictions (DataFrame): Predictions from each model
            weights (dict): Weights for each model
            
        Returns:
            array: Ensemble predictions
        """
        ensemble_pred = np.zeros(len(predictions))
        
        for model_name in predictions.columns:
            if model_name in weights:
                ensemble_pred += predictions[model_name].values * weights[model_name]
        
        return ensemble_pred
    
    def calculate_weights_from_performance(self, models, X_val, y_val):
        """
        Calculate weights based on validation MAE (inverse weighting)
        Better models (lower MAE) get higher weights
        
        Args:
            models (dict): Dictionary of models
            X_val (DataFrame): Validation features
            y_val (Series): Validation target
            
        Returns:
            dict: Weights for each model
        """
        maes = {}
        
        for model_name, model in models.items():
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            maes[model_name] = mae
        
        # Convert MAE to weights (inverse: lower MAE = higher weight)
        # Use inverse MAE, then normalize to sum to 1
        inverse_maes = {name: 1/mae for name, mae in maes.items()}
        total = sum(inverse_maes.values())
        weights = {name: inv_mae/total for name, inv_mae in inverse_maes.items()}
        
        return weights, maes
    
    def train_stacking_ensemble(self, models, X_train, y_train):
        """
        Train a stacking ensemble with Ridge meta-learner
        
        Args:
            models (dict): Dictionary of base models
            X_train (DataFrame): Training features
            y_train (Series): Training target
            
        Returns:
            StackingRegressor: Trained stacking ensemble
        """
        # Create list of (name, model) tuples for StackingRegressor
        estimators = [(name, model) for name, model in models.items()]
        
        # Create stacking ensemble with Ridge as meta-learner
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3
        )
        
        stacking.fit(X_train, y_train)
        
        return stacking
    
    def evaluate_ensemble(self, y_true, y_pred, ensemble_name):
        """
        Evaluate ensemble performance
        
        Args:
            y_true (array): True values
            y_pred (array): Predicted values
            ensemble_name (str): Name of ensemble method
            
        Returns:
            dict: Metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
    
    def create_all_ensembles(self, X_train, y_train, X_val, y_val):
        """
        Create all ensemble types for all targets
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            dict: All ensemble results
        """
        print("\n" + "="*60)
        print("Creating Ensemble Models")
        print("="*60)
        
        all_results = {}
        
        for target in self.target_cols:
            if target not in y_train.columns:
                continue
            
            print(f"\n{'='*60}")
            print(f"Target: {target}")
            print(f"{'='*60}")
            
            # Load models for this target
            models = self.load_all_models(target)
            
            if len(models) == 0:
                print(f"  ⚠ No models found for {target}")
                continue
            
            print(f"  Loaded {len(models)} models: {list(models.keys())}")
            
            # Get individual model predictions on validation set
            val_predictions = self.get_individual_predictions(models, X_val)
            
            # Evaluate individual models
            print(f"\n  Individual Model Performance:")
            individual_metrics = {}
            for model_name in val_predictions.columns:
                metrics = self.evaluate_ensemble(
                    y_val[target], 
                    val_predictions[model_name],
                    model_name
                )
                individual_metrics[model_name] = metrics
                print(f"    {model_name:12s}: MAE={metrics['MAE']:.3f}, R²={metrics['R2']:.3f}")
            
            # Create ensembles
            ensemble_results = {}
            
            # 1. Simple Average
            print(f"\n  Creating Simple Average Ensemble...")
            simple_avg_pred = self.simple_average_ensemble(val_predictions)
            simple_avg_metrics = self.evaluate_ensemble(
                y_val[target], simple_avg_pred, "Simple Average"
            )
            ensemble_results['Simple_Average'] = simple_avg_metrics
            print(f"    MAE={simple_avg_metrics['MAE']:.3f}, R²={simple_avg_metrics['R2']:.3f}")
            
            # 2. Weighted Average (performance-based)
            print(f"\n  Creating Weighted Average Ensemble...")
            weights, maes = self.calculate_weights_from_performance(
                models, X_val, y_val[target]
            )
            print(f"    Weights: {', '.join([f'{k}={v:.3f}' for k, v in weights.items()])}")
            
            weighted_avg_pred = self.weighted_average_ensemble(val_predictions, weights)
            weighted_avg_metrics = self.evaluate_ensemble(
                y_val[target], weighted_avg_pred, "Weighted Average"
            )
            ensemble_results['Weighted_Average'] = weighted_avg_metrics
            print(f"    MAE={weighted_avg_metrics['MAE']:.3f}, R²={weighted_avg_metrics['R2']:.3f}")
            
            # 3. Stacking
            print(f"\n  Creating Stacking Ensemble...")
            stacking_model = self.train_stacking_ensemble(
                models, X_train, y_train[target]
            )
            stacking_pred = stacking_model.predict(X_val)
            stacking_metrics = self.evaluate_ensemble(
                y_val[target], stacking_pred, "Stacking"
            )
            ensemble_results['Stacking'] = stacking_metrics
            print(f"    MAE={stacking_metrics['MAE']:.3f}, R²={stacking_metrics['R2']:.3f}")
            
            # Save stacking model
            stacking_path = os.path.join(
                self.models_dir, 'ensemble', f'{target}_stacking.pkl'
            )
            joblib.dump(stacking_model, stacking_path)
            
            # Find best approach
            all_approaches = {**individual_metrics, **ensemble_results}
            best_approach = min(all_approaches, key=lambda x: all_approaches[x]['MAE'])
            
            print(f"\n  ✓ Best approach: {best_approach} (MAE={all_approaches[best_approach]['MAE']:.3f})")
            
            # Store results
            all_results[target] = {
                'individual': individual_metrics,
                'ensembles': ensemble_results,
                'weights': weights,
                'best_approach': best_approach
            }
        
        return all_results
    
    def compare_all_approaches(self, all_results):
        """
        Create comprehensive comparison of all approaches
        
        Args:
            all_results (dict): Results from all targets
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE COMPARISON")
        print("="*60)
        
        comparison = []
        
        for target, results in all_results.items():
            row = {'Target': target.replace('NEXT_', '')}
            
            # Individual models
            for model_name, metrics in results['individual'].items():
                row[model_name] = metrics['MAE']
            
            # Ensembles
            for ensemble_name, metrics in results['ensembles'].items():
                row[ensemble_name] = metrics['MAE']
            
            # Best
            row['Best'] = results['best_approach']
            
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        
        print("\nMAE Comparison (Lower is Better):")
        print(comparison_df.to_string(index=False))
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        approach_cols = [col for col in comparison_df.columns if col not in ['Target', 'Best']]
        
        for col in approach_cols:
            if col in comparison_df.columns:
                avg_mae = comparison_df[col].mean()
                print(f"{col:20s}: Avg MAE = {avg_mae:.3f}")
        
        # Count wins
        print(f"\n{'='*60}")
        print("WINS BY APPROACH")
        print(f"{'='*60}")
        
        wins = comparison_df['Best'].value_counts()
        for approach, count in wins.items():
            print(f"{approach:20s}: {count}/{len(comparison_df)} targets")
        
        # Overall recommendation
        print(f"\n{'='*60}")
        print("RECOMMENDATION")
        print(f"{'='*60}")
        
        # Calculate average MAE for each approach
        avg_maes = {}
        for col in approach_cols:
            if col in comparison_df.columns:
                avg_maes[col] = comparison_df[col].mean()
        
        best_overall = min(avg_maes, key=avg_maes.get)
        print(f"\nBest overall approach: {best_overall}")
        print(f"Average MAE: {avg_maes[best_overall]:.3f}")
        
        return comparison_df
    
    def save_results(self, all_results, comparison_df):
        """Save ensemble results"""
        results_dir = os.path.join(self.models_dir, 'ensemble_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save comparison table
        comparison_df.to_csv(os.path.join(results_dir, 'ensemble_comparison.csv'), index=False)
        
        # Save detailed results as JSON
        serializable_results = {}
        for target, results in all_results.items():
            serializable_results[target] = {
                'individual': {k: {m: float(v) for m, v in metrics.items()} 
                              for k, metrics in results['individual'].items()},
                'ensembles': {k: {m: float(v) for m, v in metrics.items()} 
                             for k, metrics in results['ensembles'].items()},
                'weights': {k: float(v) for k, v in results['weights'].items()},
                'best_approach': results['best_approach']
            }
        
        json_path = os.path.join(results_dir, 'ensemble_results.json')
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_dir}")


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Projections - Ensemble Models")
    print("="*60)
    
    # Initialize
    ensemble = EnsembleModel()
    
    # Load data
    X_train, y_train, X_val, y_val = ensemble.load_prepared_data()
    
    # Create all ensembles
    all_results = ensemble.create_all_ensembles(X_train, y_train, X_val, y_val)
    
    # Compare all approaches
    comparison_df = ensemble.compare_all_approaches(all_results)
    
    # Save results
    ensemble.save_results(all_results, comparison_df)
    
    print("\n" + "="*60)
    print("✓ STEP 20 COMPLETE!")
    print("="*60)
    print("\nEnsemble models created combining Linear, RF, and XGBoost")
    print("Check the comparison table to see the best approach!")
    print("\nYour ML pipeline is now COMPLETE! 🎉")
    
    return all_results, comparison_df


if __name__ == "__main__":
    # Run the main function
    results, comparison = main()