import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Optional, Tuple, Set, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dependency_graph(dependency_df: pd.DataFrame) -> nx.DiGraph:
    """
    Load dependency data into a NetworkX directed graph.
    
    Args:
        dependency_df: DataFrame with dependency data (parent, child columns)
        
    Returns:
        NetworkX DiGraph representing the dependency structure
    """
    logger.info("Loading dependency graph from DataFrame...")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges from parent to child
    for _, row in dependency_df.iterrows():
        G.add_edge(row['parent'], row['child'])
    
    logger.info(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

def calculate_pagerank(G: nx.DiGraph, alpha: float = 0.85) -> Dict[str, float]:
    """
    Calculate PageRank scores for all nodes in the graph.
    
    Args:
        G: NetworkX directed graph
        alpha: Damping parameter for PageRank
        
    Returns:
        Dictionary mapping node names to PageRank scores
    """
    logger.info(f"Calculating PageRank with alpha={alpha}...")
    
    # Get reversed graph for PageRank (since importance flows in reverse of dependencies)
    G_reversed = G.reverse()
    
    # Calculate PageRank
    pagerank_scores = nx.pagerank(G_reversed, alpha=alpha)
    
    logger.info("PageRank calculation complete")
    return pagerank_scores

def calculate_weighted_contribution(
    G: nx.DiGraph, 
    pagerank_scores: Dict[str, float],
    github_features: Optional[Dict[str, Dict[str, float]]] = None,
    contribution_weight: float = 0.7
) -> Dict[str, float]:
    """
    Calculate weighted contribution scores combining PageRank and GitHub metrics.
    
    Args:
        G: NetworkX directed graph
        pagerank_scores: Dictionary of PageRank scores
        github_features: Dictionary of GitHub metrics for each project
        contribution_weight: Weight to assign to PageRank vs GitHub metrics
        
    Returns:
        Dictionary mapping node names to weighted scores
    """
    logger.info("Calculating weighted contribution scores...")
    
    weighted_scores = {}
    
    # Calculate weighted scores
    for node in G.nodes():
        pagerank_score = pagerank_scores.get(node, 0.0)
        
        if github_features and node in github_features:
            # Combine PageRank with GitHub metrics
            github_score = sum(github_features[node].values()) / len(github_features[node])
            weighted_score = (contribution_weight * pagerank_score) + ((1 - contribution_weight) * github_score)
        else:
            # Use PageRank only
            weighted_score = pagerank_score
        
        weighted_scores[node] = weighted_score
    
    # Normalize scores
    total_score = sum(weighted_scores.values())
    if total_score > 0:
        for node in weighted_scores:
            weighted_scores[node] = weighted_scores[node] / total_score
    
    logger.info("Weighted contribution calculation complete")
    return weighted_scores

def get_node_tiers(G: nx.DiGraph, root_node: str = "ethereum") -> Dict[str, int]:
    """
    Determine the tier level of each node in the graph.
    
    Args:
        G: NetworkX directed graph
        root_node: The root node of the graph
        
    Returns:
        Dictionary mapping node names to tier levels
    """
    logger.info(f"Calculating node tiers from root: {root_node}")
    
    # BFS to find tiers
    tiers = {root_node: 0}
    visited = {root_node}
    current_tier = 1
    current_level = list(G.successors(root_node))
    
    while current_level:
        next_level = []
        for node in current_level:
            if node not in visited:
                tiers[node] = current_tier
                visited.add(node)
                next_level.extend([n for n in G.successors(node) if n not in visited])
        
        current_level = next_level
        current_tier += 1
    
    # Assign maximum tier to any unvisited nodes
    for node in G.nodes():
        if node not in tiers:
            tiers[node] = current_tier
    
    logger.info(f"Identified {len(tiers)} nodes in {current_tier} tiers")
    return tiers

def apply_tiered_weighting(
    G: nx.DiGraph, 
    base_scores: Dict[str, float],
    max_tier_level: int = 5
) -> Dict[str, float]:
    """
    Apply tiered weighting to give more credit to deeper dependencies.
    
    Args:
        G: NetworkX directed graph
        base_scores: Base importance scores
        max_tier_level: Maximum tier level to consider
        
    Returns:
        Dictionary mapping node names to tiered importance scores
    """
    logger.info("Applying tiered weighting to scores...")
    
    # Get tier levels for each node
    try:
        # Find root node (node with highest in-degree)
        root_candidates = sorted(G.nodes(), key=lambda n: G.in_degree(n), reverse=True)
        root_node = root_candidates[0] if root_candidates else list(G.nodes())[0]
        
        tiers = get_node_tiers(G, root_node)
    except Exception as e:
        logger.warning(f"Error determining tiers: {e}. Using flat tiers.")
        tiers = {node: 1 for node in G.nodes()}
    
    # Apply tier weights
    tier_weights = {i: 1 - (i / (max_tier_level + 1)) for i in range(max_tier_level + 1)}
    
    # Calculate weighted scores
    tiered_scores = {}
    for node, base_score in base_scores.items():
        tier = min(tiers.get(node, max_tier_level), max_tier_level)
        tier_weight = tier_weights.get(tier, tier_weights[max_tier_level])
        
        # Apply tier adjustment
        tiered_scores[node] = base_score * (1 + tier_weight)
    
    # Normalize scores
    total_score = sum(tiered_scores.values())
    for node in tiered_scores:
        tiered_scores[node] = tiered_scores[node] / total_score
    
    logger.info("Tiered weighting applied")
    return tiered_scores

def identify_critical_dependencies(
    G: nx.DiGraph, 
    importance_scores: Dict[str, float],
    threshold: float = 0.9
) -> List[str]:
    """
    Identify critical dependencies based on importance scores and graph structure.
    
    Args:
        G: NetworkX directed graph
        importance_scores: Dictionary mapping nodes to importance scores
        threshold: Percentile threshold for critical dependencies
        
    Returns:
        List of critical dependency nodes
    """
    logger.info(f"Identifying critical dependencies with threshold {threshold}...")
    
    # Calculate node centrality metrics
    betweenness = nx.betweenness_centrality(G)
    
    # Combine with importance scores
    combined_scores = {}
    for node in G.nodes():
        if node in importance_scores and node in betweenness:
            combined_scores[node] = importance_scores[node] * betweenness[node]
    
    # Sort by combined score
    sorted_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Take top percentile as critical
    num_critical = max(1, int(len(sorted_nodes) * (1 - threshold)))
    critical_nodes = [node for node, _ in sorted_nodes[:num_critical]]
    
    logger.info(f"Identified {len(critical_nodes)} critical dependencies")
    return critical_nodes
