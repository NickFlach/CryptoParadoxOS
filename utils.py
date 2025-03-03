import pandas as pd
import numpy as np
import io
import logging
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_results_to_csv(
    funding_allocation: pd.DataFrame,
    include_metrics: bool = True
) -> str:
    """
    Export funding allocation results to CSV format.
    
    Args:
        funding_allocation: DataFrame with funding allocation results
        include_metrics: Whether to include additional metrics
        
    Returns:
        CSV string data
    """
    logger.info("Exporting results to CSV...")
    
    # Create a copy to avoid modifying the original
    export_df = funding_allocation.copy()
    
    # Ensure that the DataFrame has the expected columns
    column_map = {
        'Repository': 'Repository',
        'Importance Score': 'Importance_Score',
        'Allocation (ETH)': 'Allocation_ETH',
        'Allocation (%)': 'Allocation_Percent',
        'Score': 'Importance_Score'
    }
    
    # Rename columns using the mapping, but only for columns that exist
    columns_to_rename = {col: column_map[col] for col in export_df.columns if col in column_map}
    export_df = export_df.rename(columns=columns_to_rename)
    
    # Export to CSV string
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    logger.info(f"Exported {len(export_df)} rows to CSV")
    return csv_data

def compute_validation_metrics(
    predicted_scores: Dict[str, float],
    reference_scores: Optional[Dict[str, float]] = None,
    G: Optional[nx.DiGraph] = None
) -> Dict[str, float]:
    """
    Compute validation metrics for the funding allocation.
    
    Args:
        predicted_scores: Dictionary mapping projects to predicted scores
        reference_scores: Optional reference scores for comparison
        G: Optional NetworkX graph for structural validation
        
    Returns:
        Dictionary of validation metrics
    """
    logger.info("Computing validation metrics...")
    
    metrics = {}
    
    # 1. Total allocation check
    total_allocation = sum(predicted_scores.values())
    metrics['total_allocation'] = total_allocation
    
    # 2. If reference scores are provided, compute correlation
    if reference_scores is not None:
        # Get common projects
        common_projects = set(predicted_scores.keys()) & set(reference_scores.keys())
        
        if common_projects:
            # Extract scores for common projects
            pred_values = [predicted_scores[p] for p in common_projects]
            ref_values = [reference_scores[p] for p in common_projects]
            
            # Compute correlation
            correlation = np.corrcoef(pred_values, ref_values)[0, 1]
            metrics['reference_correlation'] = correlation
            
            # Compute mean absolute error
            mae = np.mean(np.abs(np.array(pred_values) - np.array(ref_values)))
            metrics['reference_mae'] = mae
    
    # 3. If graph is provided, perform structural validation
    if G is not None:
        # Check consistency: connected nodes should have somewhat related scores
        consistency_scores = []
        
        for node1, node2 in G.edges():
            if node1 in predicted_scores and node2 in predicted_scores:
                # Check relative consistency, higher dependencies should generally
                # have higher scores than their dependents
                score1 = predicted_scores[node1]
                score2 = predicted_scores[node2]
                
                # We expect parent nodes to have somewhat higher scores than children
                # but not drastically higher (soft consistency)
                if score1 > 0 and score2 > 0:
                    ratio = max(score1, score2) / min(score1, score2)
                    # Penalize very large ratios
                    consistency = 1 / (1 + np.log1p(ratio))
                    consistency_scores.append(consistency)
        
        if consistency_scores:
            metrics['structural_consistency'] = np.mean(consistency_scores)
    
    # 4. Check score distribution
    score_values = list(predicted_scores.values())
    metrics['min_score'] = min(score_values)
    metrics['max_score'] = max(score_values)
    metrics['mean_score'] = np.mean(score_values)
    metrics['median_score'] = np.median(score_values)
    metrics['std_score'] = np.std(score_values)
    
    logger.info("Validation metrics computation complete")
    return metrics

def generate_sample_dependency_graph(
    num_nodes: int = 50,
    num_seed_projects: int = 10,
    max_depth: int = 3,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate a sample dependency graph for demo/testing purposes.
    
    Args:
        num_nodes: Total number of nodes in the graph
        num_seed_projects: Number of direct dependencies of the root
        max_depth: Maximum depth of the graph
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with dependency graph data
    """
    logger.info(f"Generating sample dependency graph with {num_nodes} nodes...")
    
    import random
    random.seed(random_seed)
    
    # Create nodes
    root_node = "ethereum"
    seed_projects = [f"seed{i}" for i in range(1, num_seed_projects + 1)]
    
    remaining_nodes = num_nodes - num_seed_projects - 1
    child_projects = [f"repo{i}" for i in range(1, remaining_nodes + 1)]
    
    all_nodes = [root_node] + seed_projects + child_projects
    
    # Create edges
    edges = []
    
    # Connect root to all seed projects
    for seed in seed_projects:
        edges.append((root_node, seed))
    
    # Assign remaining nodes as children
    current_tier = seed_projects
    next_tier = []
    
    nodes_to_assign = child_projects.copy()
    current_depth = 1
    
    while nodes_to_assign and current_depth <= max_depth:
        # For each node in current tier
        for parent in current_tier:
            # Determine number of children for this parent
            num_children = random.randint(0, min(5, len(nodes_to_assign)))
            
            if num_children > 0:
                # Select children
                children = random.sample(nodes_to_assign, num_children)
                
                # Create edges
                for child in children:
                    edges.append((parent, child))
                
                # Update next tier
                next_tier.extend(children)
                
                # Remove assigned nodes
                for child in children:
                    nodes_to_assign.remove(child)
        
        # Move to next tier
        current_tier = next_tier
        next_tier = []
        current_depth += 1
    
    # Ensure all nodes are connected
    for node in nodes_to_assign:
        # Connect to a random existing node
        parent = random.choice(all_nodes)
        if parent != node:  # Avoid self-loops
            edges.append((parent, node))
    
    # Create DataFrame
    df = pd.DataFrame(edges, columns=['parent', 'child'])
    
    logger.info(f"Generated sample graph with {len(df)} edges")
    return df

def parse_github_repo_name(repo_url: str) -> str:
    """
    Parse GitHub repository name from URL or path string.
    
    Args:
        repo_url: GitHub repository URL or path
        
    Returns:
        Normalized repository name in format 'owner/repo'
    """
    # Strip URLs to get the owner/repo format
    parts = repo_url.strip('/').split('/')
    
    # Handle different URL formats
    if 'github.com' in parts:
        github_index = parts.index('github.com')
        if len(parts) >= github_index + 3:
            return f"{parts[github_index+1]}/{parts[github_index+2]}"
    elif len(parts) >= 2:
        # Assume it's already in owner/repo format
        return f"{parts[-2]}/{parts[-1]}"
    
    # Return original if we can't parse it
    return repo_url
