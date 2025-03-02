import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_ranking_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "XGBoost",
    test_size: float = 0.2,
    random_state: int = 42
) -> Any:
    """
    Train a machine learning model for ranking repositories.
    
    Args:
        X: Feature DataFrame
        y: Target Series (importance scores)
        model_type: Type of model to train
        test_size: Fraction of data to use for testing
        random_state: Random seed
        
    Returns:
        Trained model
    """
    logger.info(f"Training {model_type} ranking model...")
    
    # Split data if sufficient samples
    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model based on type
    if model_type == "XGBoost":
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
    elif model_type == "Linear":
        model = Ridge(alpha=1.0, random_state=random_state)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    if len(X_test) > 0:
        test_score = model.score(X_test_scaled, y_test)
    else:
        test_score = None
    
    # Fix string formatting
    test_score_str = f"{test_score:.4f}" if test_score is not None else "N/A"
    logger.info(f"Model training complete: Train R² = {train_score:.4f}, Test R² = {test_score_str}")
    
    return model

def predict_funding_allocation(
    model: Any,
    project_data: pd.DataFrame,
    total_funding: float = 1.0  # Normalized to 1.0 for percentage allocation
) -> pd.DataFrame:
    """
    Predict funding allocation based on model and project data.
    
    Args:
        model: Trained model
        project_data: DataFrame with project features
        total_funding: Total funding amount to allocate
        
    Returns:
        DataFrame with predicted funding allocations
    """
    logger.info("Predicting funding allocations...")
    
    # Prepare features
    X = project_data[['importance_score']]
    
    # Make predictions
    predicted_scores = model.predict(X)
    
    # Ensure non-negative values
    predicted_scores = np.maximum(predicted_scores, 0)
    
    # Normalize to sum to total_funding
    total_predicted = np.sum(predicted_scores)
    if total_predicted > 0:
        normalized_allocations = (predicted_scores / total_predicted) * total_funding
    else:
        # If all predictions are 0, distribute equally
        normalized_allocations = np.ones_like(predicted_scores) * (total_funding / len(predicted_scores))
    
    # Create results DataFrame
    funding_df = project_data.copy()
    funding_df['predicted_funding'] = normalized_allocations
    funding_df['funding_percent'] = normalized_allocations * 100
    
    # Sort by funding allocation
    funding_df = funding_df.sort_values('predicted_funding', ascending=False)
    
    logger.info("Funding allocation prediction complete")
    return funding_df

def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    # Calculate cross-validation score if sufficient data
    if len(X) >= 10:
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 2), scoring='r2')
        mean_cv_score = np.mean(cv_scores)
    else:
        mean_cv_score = r2
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "CV R²": mean_cv_score
    }
    
    logger.info(f"Model evaluation complete: R² = {r2:.4f}")
    return metrics

def generate_pairwise_comparisons(
    importance_scores: Dict[str, float],
    n_pairs: int = 100
) -> List[Tuple[str, str, int]]:
    """
    Generate pairwise comparisons for model training.
    
    Args:
        importance_scores: Dictionary mapping projects to importance scores
        n_pairs: Number of pairs to generate
        
    Returns:
        List of (project1, project2, preference) tuples
    """
    logger.info(f"Generating {n_pairs} pairwise comparisons...")
    
    projects = list(importance_scores.keys())
    pairs = []
    
    import random
    random.seed(42)
    
    for _ in range(n_pairs):
        # Select two random projects
        p1, p2 = random.sample(projects, 2)
        
        # Determine preference based on importance scores
        score1 = importance_scores[p1]
        score2 = importance_scores[p2]
        
        if score1 > score2:
            preference = 1  # p1 is preferred
        elif score2 > score1:
            preference = -1  # p2 is preferred
        else:
            preference = 0  # tie
        
        pairs.append((p1, p2, preference))
    
    logger.info("Pairwise comparison generation complete")
    return pairs

def train_pairwise_ranking_model(
    pairs: List[Tuple[str, str, int]],
    features: Dict[str, Dict[str, float]]
) -> Any:
    """
    Train a pairwise ranking model.
    
    Args:
        pairs: List of (project1, project2, preference) tuples
        features: Dictionary mapping projects to feature dictionaries
        
    Returns:
        Trained pairwise ranking model
    """
    # This is a simplified implementation
    # In practice, you would use a ranking-specific algorithm like LambdaMART
    
    logger.info("Training pairwise ranking model...")
    
    # Convert pairs to features
    X = []
    y = []
    
    for p1, p2, preference in pairs:
        if p1 in features and p2 in features:
            # Extract features for both projects
            f1 = features[p1]
            f2 = features[p2]
            
            # Create feature differences
            feature_diff = {f"diff_{k}": f1.get(k, 0) - f2.get(k, 0) 
                          for k in set(f1.keys()).union(f2.keys())}
            
            X.append(list(feature_diff.values()))
            y.append(preference)
    
    # Convert to arrays
    X = np.array(X)
    y = np.array(y)
    
    # Train model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X, y)
    
    logger.info("Pairwise ranking model training complete")
    return model
