"""
Blockchain Data Generator Module

This module provides functions for generating sample blockchain dependency data
for different blockchain ecosystems.
"""

import os
import pandas as pd
import random
import networkx as nx
from typing import List, Dict, Optional

from blockchain_manager import BlockchainManager, BlockchainConfig


def ensure_blockchain_sample_data_exists(blockchain_id: str = "ethereum") -> str:
    """
    Ensure that sample dependency data exists for the specified blockchain.
    Creates the data if it doesn't exist.
    
    Args:
        blockchain_id: ID of the blockchain to generate sample data for
        
    Returns:
        Path to the sample data file
    """
    sample_file = f"data/{blockchain_id}_dependencies.csv"
    if not os.path.exists(sample_file):
        generate_blockchain_dependency_csv(blockchain_id, sample_file)
    return sample_file


def generate_blockchain_dependency_csv(
    blockchain_id: str = "ethereum", 
    output_path: Optional[str] = None
) -> str:
    """
    Generate sample dependency data for a specific blockchain and save to CSV.
    
    Args:
        blockchain_id: ID of the blockchain to generate sample data for
        output_path: Path to save the CSV file (default: data/{blockchain_id}_dependencies.csv)
        
    Returns:
        Path to the generated CSV file
    """
    if output_path is None:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        output_path = f"data/{blockchain_id}_dependencies.csv"
    
    # Get blockchain configuration
    blockchain_manager = BlockchainManager()
    blockchain_config = blockchain_manager.get_blockchain(blockchain_id)
    
    if blockchain_config is None:
        raise ValueError(f"Blockchain configuration not found for ID: {blockchain_id}")
    
    # Generate dependency graph based on blockchain configuration
    dependency_df = _generate_dependency_graph(blockchain_config)
    
    # Save to CSV
    dependency_df.to_csv(output_path, index=False)
    return output_path


def _generate_dependency_graph(blockchain_config: BlockchainConfig) -> pd.DataFrame:
    """
    Generate a dependency graph for a blockchain based on its configuration.
    
    Args:
        blockchain_config: BlockchainConfig object with blockchain details
        
    Returns:
        DataFrame with dependency relationships
    """
    # Extract root and seed repositories
    root_repo = blockchain_config.root_repository
    seed_repos = blockchain_config.seed_repositories
    
    # Create an empty graph
    G = nx.DiGraph()
    
    # Add root node
    G.add_node(root_repo)
    
    # Add seed repositories as direct dependencies of root
    for repo in seed_repos:
        G.add_node(repo)
        G.add_edge(root_repo, repo)
    
    # Generate additional repositories (simulated dependencies)
    additional_repos = _generate_additional_repos(blockchain_config, 75)  # Generate 75 additional repositories
    
    # Add additional repositories
    for repo in additional_repos:
        G.add_node(repo)
    
    # Add edges between seed repositories and additional repositories
    # Each seed repo will be a parent to some additional repos
    for seed_repo in seed_repos:
        # Select a random number of children for this seed repo
        num_children = random.randint(2, 8)
        children = random.sample(additional_repos, min(num_children, len(additional_repos)))
        
        for child in children:
            G.add_edge(seed_repo, child)
    
    # Add some relationships between additional repositories
    for _ in range(min(40, len(additional_repos))):  # Add 40 more edges or less if fewer repos
        source = random.choice(list(G.nodes()))
        targets = [n for n in additional_repos if n != source and not G.has_edge(source, n)]
        
        if targets:
            target = random.choice(targets)
            G.add_edge(source, target)
    
    # Convert graph to DataFrame
    edges = list(G.edges())
    dependency_df = pd.DataFrame({
        'parent': [e[0] for e in edges],
        'child': [e[1] for e in edges]
    })
    
    return dependency_df


def _generate_additional_repos(blockchain_config: BlockchainConfig, num_repos: int) -> List[str]:
    """
    Generate additional repository names based on blockchain characteristics.
    
    Args:
        blockchain_config: BlockchainConfig object with blockchain details
        num_repos: Number of repositories to generate
        
    Returns:
        List of repository names
    """
    # Common prefixes and suffixes based on blockchain ecosystem
    blockchain_name = blockchain_config.name
    
    common_prefixes = {
        'ethereum': ['eth-', 'web3-', 'defi-', 'evm-', 'solidity-', ''],
        'solana': ['sol-', 'anchor-', 'solana-', 'serum-', 'spl-', ''],
        'polkadot': ['substrate-', 'polkadot-', 'dot-', 'kusama-', 'parachain-', ''],
    }
    
    common_suffixes = {
        'ethereum': ['-dao', '-protocol', '-dapp', '-contract', '-finance', '-token', '-bridge', ''],
        'solana': ['-program', '-dapp', '-wallet', '-app', '-protocol', '-nft', '-bridge', ''],
        'polkadot': ['-node', '-pallet', '-parachain', '-runtime', '-sdk', '-bridge', '-api', ''],
    }
    
    common_orgs = {
        'ethereum': ['ethereum', 'consensys', 'openzeppelin', 'aave', 'uniswap', 'compound', 'chainlink', 'gnosis'],
        'solana': ['solana-labs', 'project-serum', 'coral-xyz', 'metaplex', 'solflare', 'solend', 'mango'],
        'polkadot': ['paritytech', 'polkadot-js', 'edgeware', 'acala', 'moonbeam', 'centrifuge', 'phala'],
    }
    
    repo_types = {
        'ethereum': ['wallet', 'bridge', 'token', 'dao', 'protocol', 'dex', 'oracle', 'contract', 'standard', 'pool', 'staking'],
        'solana': ['program', 'wallet', 'dapp', 'nft', 'pool', 'staking', 'sdk', 'farm', 'vault', 'swap'],
        'polkadot': ['parachain', 'node', 'runtime', 'sdk', 'pallet', 'bridge', 'api', 'module', 'client', 'tools'],
    }
    
    # Get the appropriate lists for this blockchain, falling back to ethereum for unknown blockchains
    prefixes = common_prefixes.get(blockchain_name, common_prefixes['ethereum'])
    suffixes = common_suffixes.get(blockchain_name, common_suffixes['ethereum'])
    orgs = common_orgs.get(blockchain_name, common_orgs['ethereum'])
    types = repo_types.get(blockchain_name, repo_types['ethereum'])
    
    additional_repos = []
    
    # Generate repositories from common organizations
    for _ in range(num_repos // 3):
        org = random.choice(orgs)
        repo_type = random.choice(types)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        repo_name = f"{org}/{prefix}{repo_type}{suffix}"
        if repo_name not in additional_repos:
            additional_repos.append(repo_name)
    
    # Generate repositories from random community organizations
    community_orgs = [
        "dev", "team", f"{blockchain_name}-community", 
        f"{blockchain_name}-labs", f"open-{blockchain_name}"
    ] 
    
    while len(additional_repos) < num_repos:
        org = f"{random.choice(community_orgs)}-{random.randint(1, 100)}"
        repo_type = random.choice(types)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        repo_name = f"{org}/{prefix}{repo_type}{suffix}"
        if repo_name not in additional_repos:
            additional_repos.append(repo_name)
    
    return additional_repos


def generate_all_blockchain_samples() -> Dict[str, str]:
    """
    Generate sample data for all registered blockchains.
    
    Returns:
        Dictionary mapping blockchain IDs to their sample data file paths
    """
    blockchain_manager = BlockchainManager()
    blockchains = blockchain_manager.get_all_blockchains()
    
    sample_paths = {}
    for blockchain_id in blockchains:
        path = ensure_blockchain_sample_data_exists(blockchain_id)
        sample_paths[blockchain_id] = path
    
    return sample_paths


if __name__ == "__main__":
    # Generate samples for all blockchains when run directly
    samples = generate_all_blockchain_samples()
    print(f"Generated sample data for {len(samples)} blockchains:")
    for blockchain_id, path in samples.items():
        print(f"  - {blockchain_id}: {path}")