import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Set, Optional, Any
import random  # For generating sample data

# Note: In a real implementation, you would use requests or a GitHub API client
# For this demo, we'll simulate GitHub API calls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_github_metrics(
    node_names: List[str], 
    use_real_api: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Extract GitHub metrics for a list of repositories.
    
    Args:
        node_names: List of repository names
        use_real_api: Whether to use real GitHub API (not implemented here)
        
    Returns:
        Dictionary mapping repository names to metrics dictionaries
    """
    logger.info(f"Extracting GitHub metrics for {len(node_names)} repositories")
    
    # In a real implementation, you would use the GitHub API
    # For this demo, we'll generate sample data
    metrics = {}
    
    for i, node in enumerate(node_names):
        # Simulate API rate limiting
        if i > 0 and i % 50 == 0:
            time.sleep(0.1)
        
        if use_real_api:
            # This would be implemented with actual API calls in production
            # metrics[node] = get_github_metrics_from_api(node)
            pass
        else:
            # Generate sample data
            metrics[node] = generate_sample_github_metrics(node)
    
    logger.info(f"Extracted metrics for {len(metrics)} repositories")
    return metrics

def generate_sample_github_metrics(repo_name: str) -> Dict[str, float]:
    """
    Generate sample GitHub metrics for demo purposes.
    
    Args:
        repo_name: Repository name
        
    Returns:
        Dictionary of sample metrics
    """
    # Use deterministic random seeding based on repo name
    # so same repo always gets same values
    random.seed(hash(repo_name) % 10000)
    
    # Generate more realistic metrics based on typical GitHub values
    # Core repositories tend to have more stars and activity
    is_core = any(keyword in repo_name.lower() for keyword in [
        "ethereum", "eth", "web3", "core", "consensus", "protocol"
    ])
    
    # Base metrics
    if is_core:
        stars_base = random.randint(1000, 50000)
        forks_base = random.randint(200, 5000)
        commits_base = random.randint(500, 20000)
        contributors_base = random.randint(20, 200)
        issues_base = random.randint(100, 2000)
    else:
        stars_base = random.randint(10, 5000)
        forks_base = random.randint(5, 500)
        commits_base = random.randint(50, 2000)
        contributors_base = random.randint(2, 50)
        issues_base = random.randint(10, 500)
    
    # Add some noise
    metrics = {
        "stars": stars_base * (0.9 + 0.2 * random.random()),
        "forks": forks_base * (0.9 + 0.2 * random.random()),
        "commits": commits_base * (0.9 + 0.2 * random.random()),
        "contributors": contributors_base * (0.9 + 0.2 * random.random()),
        "open_issues": issues_base * (0.9 + 0.2 * random.random()),
        "closed_issues": issues_base * (1.5 + random.random()) * (0.9 + 0.2 * random.random()),
        "last_updated_days": random.randint(1, 365),
        "age_days": random.randint(365, 365 * 5)
    }
    
    # Calculate derived metrics
    metrics["issue_closing_rate"] = metrics["closed_issues"] / (metrics["open_issues"] + metrics["closed_issues"])
    metrics["commit_frequency"] = metrics["commits"] / metrics["age_days"]
    metrics["contributor_engagement"] = metrics["commits"] / (metrics["contributors"] * metrics["age_days"]) * 100
    
    return metrics

def normalize_github_features(
    github_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Normalize GitHub metrics to [0,1] scale for feature consistency.
    
    Args:
        github_metrics: Dictionary mapping repositories to metrics
        
    Returns:
        Dictionary mapping repositories to normalized metrics
    """
    logger.info("Normalizing GitHub metrics...")
    
    # Extract features to normalize
    feature_columns = [
        "stars", "forks", "commits", "contributors", 
        "issue_closing_rate", "commit_frequency", "contributor_engagement"
    ]
    
    # Collect values for each feature
    feature_values = {col: [] for col in feature_columns}
    for repo, metrics in github_metrics.items():
        for col in feature_columns:
            if col in metrics:
                feature_values[col].append(metrics[col])
    
    # Calculate min-max for each feature
    feature_ranges = {}
    for col, values in feature_values.items():
        if values:
            feature_ranges[col] = (min(values), max(values))
        else:
            feature_ranges[col] = (0, 1)  # Default range
    
    # Normalize features
    normalized_metrics = {}
    for repo, metrics in github_metrics.items():
        normalized_metrics[repo] = {}
        for col in feature_columns:
            if col in metrics:
                min_val, max_val = feature_ranges[col]
                if max_val > min_val:
                    normalized_metrics[repo][col] = (metrics[col] - min_val) / (max_val - min_val)
                else:
                    normalized_metrics[repo][col] = 0.5  # Default to middle value if range is zero
    
    # Calculate combined significance score
    for repo in normalized_metrics:
        features = normalized_metrics[repo]
        # Weighted average
        weights = {
            "stars": 0.25,
            "forks": 0.15,
            "commits": 0.2,
            "contributors": 0.15,
            "issue_closing_rate": 0.1,
            "commit_frequency": 0.1,
            "contributor_engagement": 0.05
        }
        
        total_weight = 0
        weighted_sum = 0
        for feature, value in features.items():
            if feature in weights:
                weighted_sum += value * weights[feature]
                total_weight += weights[feature]
        
        if total_weight > 0:
            normalized_metrics[repo]["significance_score"] = weighted_sum / total_weight
        else:
            normalized_metrics[repo]["significance_score"] = 0.5
    
    logger.info("GitHub metrics normalization complete")
    return normalized_metrics

def get_repository_dependencies(
    repo_name: str,
    use_real_api: bool = False
) -> List[str]:
    """
    Get dependencies for a GitHub repository.
    
    Args:
        repo_name: Repository name
        use_real_api: Whether to use real GitHub API
        
    Returns:
        List of dependency repository names
    """
    # This would use GitHub API to fetch package.json, requirements.txt, etc.
    # For demo purposes, we'll generate random dependencies
    random.seed(hash(repo_name) % 10000)
    
    common_eth_deps = [
        "ethereum/go-ethereum", "ethereum/solidity", "web3j/web3j", 
        "OpenZeppelin/openzeppelin-contracts", "trufflesuite/truffle",
        "ethers-io/ethers.js", "ethereum/py-evm", "ethereum/web3.py",
        "ethereum/evmc", "ethereum/solc-js"
    ]
    
    # Generate 1-5 dependencies
    num_deps = random.randint(1, 5)
    
    # Higher chance of common dependencies
    if random.random() < 0.7:
        return random.sample(common_eth_deps, min(num_deps, len(common_eth_deps)))
    else:
        return [f"repo{random.randint(1, 1000)}/dep{random.randint(1, 100)}" for _ in range(num_deps)]
