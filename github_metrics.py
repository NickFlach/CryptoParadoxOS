import pandas as pd
import numpy as np
import time
import logging
import csv
import os
import datetime
from typing import Dict, List, Set, Optional, Any, Tuple
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
        "ethereum/evmc", "ethereum/solc-js", "ethereum/consensus-specs",
        "ethereum/EIPs", "ethereum/remix", "ethereum/execution-apis",
        "ethereum/ethereumjs-monorepo", "ethereum/eth-json-rpc-filters"
    ]
    
    # Generate 1-5 dependencies
    num_deps = random.randint(1, 5)
    
    # Higher chance of common dependencies
    if random.random() < 0.7:
        return random.sample(common_eth_deps, min(num_deps, len(common_eth_deps)))
    else:
        return [f"repo{random.randint(1, 1000)}/dep{random.randint(1, 100)}" for _ in range(num_deps)]


def generate_sample_dependency_csv(output_path: str = "data/ethereum_dependencies.csv") -> str:
    """
    Generate sample Ethereum dependency graph data and save to CSV.
    
    Args:
        output_path: Path to save CSV file
        
    Returns:
        Path to the generated CSV file
    """
    logger.info(f"Generating sample dependency graph data to {output_path}")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Core Ethereum repositories to use as starting points
    ethereum_core_repos = [
        "ethereum/go-ethereum",
        "ethereum/solidity",
        "ethereum/py-evm",
        "ethereum/consensus-specs",
        "ethereum/execution-apis",
        "ethereum/remix",
        "ethereum/EIPs"
    ]
    
    # Dependencies between core repositories (parent -> child)
    dependencies = []
    
    # Generate first-level dependencies for core repos
    for repo in ethereum_core_repos:
        # Generate 3-8 direct dependencies for each core repo
        num_deps = random.randint(3, 8)
        for _ in range(num_deps):
            child = get_random_ethereum_repo()
            dependencies.append((repo, child))
    
    # Add some second-level dependencies
    level1_repos = set([dep[1] for dep in dependencies])
    for repo in list(level1_repos)[:20]:  # Take first 20 level-1 repos
        # Generate 1-4 dependencies for each level-1 repo
        num_deps = random.randint(1, 4)
        for _ in range(num_deps):
            child = get_random_ethereum_repo()
            dependencies.append((repo, child))
    
    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['parent', 'child'])
        for dep in dependencies:
            writer.writerow(dep)
    
    logger.info(f"Generated {len(dependencies)} dependencies to {output_path}")
    return output_path


def get_random_ethereum_repo() -> str:
    """
    Generate a random Ethereum-related repository name.
    
    Returns:
        Repository name in format "owner/repo"
    """
    common_owners = [
        "ethereum", "OpenZeppelin", "trufflesuite", "ethers-io", 
        "web3j", "ChainSafe", "ConsenSys", "status-im", "paritytech",
        "ethereum-optimism", "matter-labs", "OffchainLabs", "AztecProtocol"
    ]
    
    common_repos = [
        "web3.js", "hardhat", "contracts", "plasma", "rollup", 
        "solidity", "eth2.0-specs", "evm", "geth", "zk-sync", 
        "optimism", "arbitrum", "node", "client", "protocol",
        "starknet", "layer2", "utils", "core", "beacon-chain"
    ]
    
    # 70% chance to use common names, 30% to generate random
    if random.random() < 0.7:
        owner = random.choice(common_owners)
        repo = random.choice(common_repos)
        return f"{owner}/{repo}"
    else:
        return f"repo{random.randint(1, 1000)}/eth-{random.randint(1, 100)}"


def ensure_sample_data_exists():
    """
    Ensure sample data exists, generating it if needed.
    """
    # Check and generate dependency graph data
    dep_graph_path = "data/ethereum_dependencies.csv"
    if not os.path.exists(dep_graph_path):
        generate_sample_dependency_csv(dep_graph_path)
    
    # Check and generate GitHub data
    github_data_path = "data/github_data.csv"
    if not os.path.exists(github_data_path):
        generate_sample_github_data_csv(github_data_path)


def generate_sample_github_data_csv(output_path: str = "data/github_data.csv") -> str:
    """
    Generate sample GitHub repository data and save to CSV.
    
    Args:
        output_path: Path to save CSV file
        
    Returns:
        Path to the generated CSV file
    """
    logger.info(f"Generating sample GitHub data to {output_path}")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use a mix of real and generated repository names
    ethereum_repos = [
        "ethereum/go-ethereum",
        "ethereum/solidity",
        "ethereum/py-evm",
        "ethereum/consensus-specs",
        "ethereum/execution-apis",
        "ethereum/remix",
        "ethereum/EIPs",
        "OpenZeppelin/openzeppelin-contracts",
        "trufflesuite/truffle",
        "ethers-io/ethers.js"
    ]
    
    # Add more generated repos
    for _ in range(30):
        ethereum_repos.append(get_random_ethereum_repo())
    
    # Generate data for each repo
    repo_data = []
    for repo_name in ethereum_repos:
        owner, repo = repo_name.split('/')
        
        # Generate dates
        now = datetime.datetime.now()
        created_date = now - datetime.timedelta(days=random.randint(365, 365*5))
        updated_date = now - datetime.timedelta(days=random.randint(1, 90))
        last_push_date = now - datetime.timedelta(days=random.randint(1, 30))
        
        # Core repositories tend to have more stars and activity
        is_core = "ethereum" in owner.lower()
        
        # Generate metrics
        if is_core:
            stars = random.randint(5000, 50000)
            forks = random.randint(1000, 10000)
            open_issues = random.randint(100, 500)
            size = random.randint(10000, 100000)
        else:
            stars = random.randint(100, 20000)
            forks = random.randint(20, 2000)
            open_issues = random.randint(5, 200)
            size = random.randint(1000, 50000)
        
        # License types
        license_types = ["MIT", "GPL-3.0", "LGPL-3.0", "Apache-2.0", "BSD-3-Clause"]
        license_type = random.choice(license_types)
        
        # Languages
        languages = ["Go", "Solidity", "JavaScript", "Python", "TypeScript", "Rust", "C++"]
        language = random.choice(languages)
        
        # Generate a detailed description
        descriptions = [
            f"{repo} - A {language} implementation for Ethereum",
            f"Official {language} library for Ethereum",
            f"Ethereum {repo} implementation",
            f"{repo} utility for blockchain development",
            f"Smart contract development tools for {language}",
            f"Layer 2 scaling solution for Ethereum",
            f"Ethereum Virtual Machine implementation in {language}",
            f"Zero-knowledge proof implementation for Ethereum"
        ]
        description = random.choice(descriptions)
        
        # Compile repo data
        repo_data.append({
            "owner": owner,
            "repo_name": repo,
            "description": description,
            "language": language,
            "stars": stars,
            "forks": forks,
            "open_issues": open_issues,
            "watchers_count": int(stars * 0.2),  # 20% of stars are watchers
            "license": license_type,
            "created_at": created_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "updated_at": updated_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_push_date": last_push_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "size": size
        })
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = repo_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in repo_data:
            writer.writerow(data)
    
    logger.info(f"Generated data for {len(repo_data)} repositories to {output_path}")
    return output_path
