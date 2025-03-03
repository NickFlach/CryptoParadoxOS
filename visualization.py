import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import Dict, Optional, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dependency_graph_visualization(
    G: nx.DiGraph,
    node_size_map: Optional[Dict[str, float]] = None,
    colorscale: str = 'Viridis',
    max_nodes: int = 200
) -> go.Figure:
    """
    Create an interactive visualization of the dependency graph.
    
    Args:
        G: NetworkX directed graph
        node_size_map: Optional dictionary mapping node names to sizes
        colorscale: Plotly colorscale name
        max_nodes: Maximum number of nodes to display
        
    Returns:
        Plotly figure with graph visualization
    """
    logger.info("Creating dependency graph visualization...")
    
    # Limit nodes for visualization if too many
    nodes_list = list(G.nodes())
    if len(nodes_list) > max_nodes:
        logger.info(f"Limiting graph to {max_nodes} nodes for visualization")
        
        # Take a subset based on most important nodes if node_size_map is provided
        if node_size_map:
            important_nodes = sorted(node_size_map.items(), key=lambda x: x[1], reverse=True)
            nodes_to_keep = [n for n, _ in important_nodes[:max_nodes]]
        else:
            # Otherwise take nodes with highest degree
            degree_list = list(G.degree())
            node_degrees = sorted(degree_list, key=lambda x: x[1], reverse=True)
            nodes_to_keep = [n for n, _ in node_degrees[:max_nodes]]
        
        G = nx.DiGraph(G.subgraph(nodes_to_keep))
    
    # Create position layout using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Extract positions
    node_x = []
    node_y = []
    for node in G.nodes():
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
    
    # Configure node sizes
    if node_size_map:
        # Use provided size map, normalize to reasonable values
        sizes = [node_size_map.get(node, 0) * 50 + 10 for node in G.nodes()]
    else:
        # Use degree as size, convert to list to avoid type issues
        degree_dict = dict(G.degree())
        sizes = [(degree_dict.get(node, 0) + 1) * 5 for node in G.nodes()]
    
    # Create colorscale values based on node metrics
    if node_size_map:
        node_color = [node_size_map.get(node, 0) for node in G.nodes()]
    else:
        # Use degree as color, using the same dictionary conversion for type safety
        node_color = [degree_dict.get(node, 0) for node in G.nodes()]
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=sizes,
            color=node_color,
            colorscale=colorscale,
            colorbar=dict(
                title='Importance',
                thickness=15,
                xanchor='left',
                x=1.02
            ),
            line=dict(width=1, color='#888')
        ),
        text=[f"{node}<br>Score: {node_size_map.get(node, 0):.4f}" if node_size_map else node for node in G.nodes()],
        hoverinfo='text'
    )
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.8, color='#888'),
        hoverinfo='none'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text='Ethereum Dependency Graph',
                           font=dict(size=16)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    logger.info("Dependency graph visualization created")
    return fig

def create_funding_allocation_chart(funding_allocation: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart of funding allocations.
    
    Args:
        funding_allocation: DataFrame with funding allocations
        
    Returns:
        Plotly bar chart
    """
    logger.info("Creating funding allocation chart...")
    
    # Create funding allocation bar chart
    fig = px.bar(
        funding_allocation,
        y='project',
        x='funding_percent',
        orientation='h',
        labels={'project': 'Project', 'funding_percent': 'Funding Allocation (%)'},
        title='Funding Allocation by Project',
        color='funding_percent',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=800,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Funding Allocation (%)',
        yaxis_title='Project',
        coloraxis_colorbar=dict(title='Allocation %')
    )
    
    logger.info("Funding allocation chart created")
    return fig

def create_project_importance_heatmap(importance_df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap visualization of project importance.
    
    Args:
        importance_df: DataFrame with project importance scores
        
    Returns:
        Plotly heatmap figure
    """
    logger.info("Creating project importance heatmap...")
    
    # Reshape data for heatmap
    # We'll create a matrix with projects on both axes
    # and importance scores as values
    projects = importance_df['project'].tolist()
    scores = importance_df['importance_score'].tolist()
    
    # Normalize scores to [0, 1]
    max_score = max(scores)
    if max_score > 0:
        normalized_scores = [s / max_score for s in scores]
    else:
        normalized_scores = scores
    
    # Create matrix data
    matrix = []
    for i, project1 in enumerate(projects):
        row = []
        for j, project2 in enumerate(projects):
            # Diagonal is importance score
            if i == j:
                row.append(normalized_scores[i])
            # Off-diagonal is pairwise relationship
            # (simplistic: just average of the two scores)
            else:
                row.append((normalized_scores[i] + normalized_scores[j]) / 2)
        matrix.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=projects,
        y=projects,
        colorscale='Viridis',
        colorbar=dict(title='Importance Score')
    ))
    
    # Update layout
    fig.update_layout(
        title='Project Importance Heatmap',
        xaxis_title='Project',
        yaxis_title='Project',
        height=800,
        width=800
    )
    
    # Improve readability of axis labels
    fig.update_xaxes(tickangle=45)
    
    logger.info("Project importance heatmap created")
    return fig

def create_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Create a comparison chart between PageRank and final scores.
    
    Args:
        comparison_df: DataFrame with projects and different scoring metrics
        
    Returns:
        Plotly figure with comparison chart
    """
    logger.info("Creating score comparison chart...")
    
    # Melt DataFrame for easier plotting
    melted_df = pd.melt(
        comparison_df,
        id_vars=['project'],
        value_vars=['pagerank', 'final_score'],
        var_name='scoring_method',
        value_name='score'
    )
    
    # Create grouped bar chart
    fig = px.bar(
        melted_df,
        x='project',
        y='score',
        color='scoring_method',
        barmode='group',
        labels={'project': 'Project', 'score': 'Score', 'scoring_method': 'Scoring Method'},
        title='Comparison of Scoring Methods',
        color_discrete_map={
            'pagerank': '#1E88E5',
            'final_score': '#FFC107'
        }
    )
    
    # Update layout
    fig.update_layout(
        xaxis_tickangle=45,
        xaxis_title='Project',
        yaxis_title='Score',
        legend_title='Scoring Method',
        height=600,
        width=900
    )
    
    logger.info("Score comparison chart created")
    return fig

def create_gnn_relationship_visualization(
    G: nx.DiGraph,
    gnn_scores: Dict[str, float],
    pagerank_scores: Dict[str, float],
    unsung_heroes: Optional[List[Dict[str, Any]]] = None,
    colorscale: str = 'Plasma',
    max_nodes: int = 200,
    highlight_heroes: bool = True
) -> go.Figure:
    """
    Create a specialized visualization showing GNN-detected relationships
    with an option to highlight "unsung hero" repositories.
    
    Args:
        G: NetworkX directed graph
        gnn_scores: Dictionary mapping nodes to GNN importance scores
        pagerank_scores: Dictionary mapping nodes to PageRank scores
        unsung_heroes: Optional list of unsung hero dictionaries from identify_unsung_heroes
        colorscale: Plotly colorscale name
        max_nodes: Maximum number of nodes to display
        highlight_heroes: Whether to highlight unsung heroes
        
    Returns:
        Plotly figure with GNN relationship visualization
    """
    logger.info("Creating GNN relationship visualization...")
    
    # Limit nodes for visualization if too many
    nodes_list = list(G.nodes())
    if len(nodes_list) > max_nodes:
        logger.info(f"Limiting graph to {max_nodes} nodes for visualization")
        
        # First, make sure we include unsung heroes
        nodes_to_keep = []
        if unsung_heroes and highlight_heroes:
            hero_nodes = [hero["repository"] for hero in unsung_heroes]
            nodes_to_keep.extend(hero_nodes[:min(len(hero_nodes), max_nodes // 4)])
        
        # Then add top GNN-ranked nodes
        remaining_slots = max_nodes - len(nodes_to_keep)
        gnn_ranked_nodes = sorted(gnn_scores.items(), key=lambda x: x[1], reverse=True)
        for node, _ in gnn_ranked_nodes:
            if node not in nodes_to_keep and remaining_slots > 0:
                nodes_to_keep.append(node)
                remaining_slots -= 1
        
        G = nx.DiGraph(G.subgraph(nodes_to_keep))
    
    # Create position layout using spring layout with more iterations for better spacing
    pos = nx.spring_layout(G, seed=42, iterations=100, k=0.3)
    
    # Extract positions
    node_x = []
    node_y = []
    for node in G.nodes():
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
    
    # Create a node size map based on GNN scores
    node_sizes = []
    node_colors = []
    node_text = []
    is_hero = []
    
    # Track ranges for normalizing
    min_gnn = min(gnn_scores.values()) if gnn_scores else 0
    max_gnn = max(gnn_scores.values()) if gnn_scores else 1
    gnn_range = max_gnn - min_gnn if max_gnn > min_gnn else 1
    
    min_pr = min(pagerank_scores.values()) if pagerank_scores else 0
    max_pr = max(pagerank_scores.values()) if pagerank_scores else 1
    pr_range = max_pr - min_pr if max_pr > min_pr else 1
    
    hero_set = set()
    if unsung_heroes:
        hero_set = {hero["repository"] for hero in unsung_heroes}
    
    for node in G.nodes():
        # Size based on GNN score
        gnn_score = gnn_scores.get(node, 0)
        # Normalizing to [0,1] for consistency
        normalized_gnn = (gnn_score - min_gnn) / gnn_range if gnn_range > 0 else 0
        
        # Value to display is the difference between normalized GNN and PageRank
        pr_score = pagerank_scores.get(node, 0)
        normalized_pr = (pr_score - min_pr) / pr_range if pr_range > 0 else 0
        
        # Color indicates the difference between GNN and PageRank scores
        # Positive = GNN rates higher, Negative = PageRank rates higher
        score_diff = normalized_gnn - normalized_pr
        
        # Track if this is a hero node
        is_hero_node = node in hero_set
        is_hero.append(is_hero_node)
        
        # Sizes 
        node_size = 15 + normalized_gnn * 50
        # Increase size for hero nodes
        if is_hero_node and highlight_heroes:
            node_size *= 1.5
        
        node_sizes.append(node_size)
        node_colors.append(score_diff)
        
        # Prepare hover text
        node_text.append(
            f"<b>{node}</b><br>" +
            f"GNN Score: {gnn_score:.4f}<br>" +
            f"PageRank: {pr_score:.4f}<br>" +
            f"Difference: {score_diff:.4f}" +
            (f"<br><b>UNSUNG HERO!</b>" if is_hero_node else "")
        )
    
    # Create edge traces with directional arrows
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Calculate arrow position (80% along the edge)
        arrow_x = x0 + 0.8 * (x1 - x0)
        arrow_y = y0 + 0.8 * (y1 - y0)
        
        # Edge line
        edge_trace = go.Scatter(
            x=[x0, x1], 
            y=[y0, y1],
            mode='lines',
            line=dict(width=1, color='rgba(100,100,100,0.2)'),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)
        
        # Arrow head (small triangle)
        dx = x1 - x0
        dy = y1 - y0
        angle = np.arctan2(dy, dx)
        
        # Create a small triangle as arrow
        arrow_size = 0.02
        arrow_trace = go.Scatter(
            x=[arrow_x, 
               arrow_x + arrow_size * np.cos(angle + np.pi - np.pi/4), 
               arrow_x + arrow_size * np.cos(angle + np.pi + np.pi/4),
               arrow_x],
            y=[arrow_y, 
               arrow_y + arrow_size * np.sin(angle + np.pi - np.pi/4),
               arrow_y + arrow_size * np.sin(angle + np.pi + np.pi/4),
               arrow_y],
            mode='lines',
            fill='toself',
            line=dict(width=0),
            fillcolor='rgba(100,100,100,0.4)',
            hoverinfo='none'
        )
        edge_traces.append(arrow_trace)
    
    # Create two node traces - one for regular nodes and one for heroes
    if highlight_heroes and any(is_hero):
        # Regular nodes
        regular_indices = [i for i, hero in enumerate(is_hero) if not hero]
        if regular_indices:
            regular_trace = go.Scatter(
                x=[node_x[i] for i in regular_indices],
                y=[node_y[i] for i in regular_indices],
                mode='markers',
                marker=dict(
                    size=[node_sizes[i] for i in regular_indices],
                    color=[node_colors[i] for i in regular_indices],
                    colorscale=colorscale,
                    colorbar=dict(
                        title='GNN vs PageRank',
                        thickness=15,
                        xanchor='left',
                        x=1.02
                    ),
                    line=dict(width=1, color='#888')
                ),
                text=[node_text[i] for i in regular_indices],
                hoverinfo='text',
                name='Regular Projects'
            )
        else:
            regular_trace = None
        
        # Hero nodes with different appearance
        hero_indices = [i for i, hero in enumerate(is_hero) if hero]
        hero_trace = go.Scatter(
            x=[node_x[i] for i in hero_indices],
            y=[node_y[i] for i in hero_indices],
            mode='markers',
            marker=dict(
                size=[node_sizes[i] for i in hero_indices],
                color='rgba(255, 65, 54, 0.9)',  # Red for heroes
                symbol='star',  # Star shape for heroes
                line=dict(width=2, color='rgb(50, 50, 50)')
            ),
            text=[node_text[i] for i in hero_indices],
            hoverinfo='text',
            name='Unsung Heroes'
        )
        
        # Combine traces
        node_traces = [hero_trace]
        if regular_trace:
            node_traces.append(regular_trace)
    else:
        # Standard node trace without hero highlighting
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale=colorscale,
                colorbar=dict(
                    title='GNN vs PageRank',
                    thickness=15,
                    xanchor='left',
                    x=1.02
                ),
                line=dict(width=1, color='#888')
            ),
            text=node_text,
            hoverinfo='text'
        )
        node_traces = [node_trace]
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title=dict(
                text='GNN Relationship Analysis' + 
                     (' (Highlighting Unsung Heroes)' if highlight_heroes and any(is_hero) else ''),
                font=dict(size=16)
            ),
            showlegend=highlight_heroes and any(is_hero),
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                x=0,
                y=1.1,
                orientation='h'
            )
        )
    )
    
    # Add annotation explaining the color scale
    fig.add_annotation(
        x=1.0,
        y=0.0,
        xref='paper',
        yref='paper',
        text='Higher GNN score than PageRank → Warmer colors<br>Higher PageRank than GNN → Cooler colors',
        showarrow=False,
        font=dict(size=10),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='rgba(0,0,0,0.3)',
        borderwidth=1,
        borderpad=4,
        align='left'
    )
    
    logger.info("GNN relationship visualization created")
    return fig


def create_tier_distribution_chart(
    graph_tiers: Dict[str, int],
    importance_scores: Dict[str, float]
) -> go.Figure:
    """
    Create a visualization showing score distribution across tiers.
    
    Args:
        graph_tiers: Dictionary mapping nodes to tier levels
        importance_scores: Dictionary mapping nodes to importance scores
        
    Returns:
        Plotly figure with tier distribution
    """
    logger.info("Creating tier distribution chart...")
    
    # Create DataFrame from tiers and scores
    data = []
    for node, tier in graph_tiers.items():
        if node in importance_scores:
            data.append({
                'project': node,
                'tier': tier,
                'importance_score': importance_scores[node]
            })
    
    tier_df = pd.DataFrame(data)
    
    # Calculate aggregates by tier
    tier_agg = tier_df.groupby('tier').agg(
        total_score=('importance_score', 'sum'),
        avg_score=('importance_score', 'mean'),
        project_count=('project', 'count')
    ).reset_index()
    
    # Create charts
    fig = px.bar(
        tier_agg,
        x='tier',
        y='total_score',
        labels={'tier': 'Tier Level', 'total_score': 'Total Importance Score'},
        title='Importance Score Distribution by Tier',
        color='project_count',
        text='project_count'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Tier Level',
        yaxis_title='Total Importance Score',
        coloraxis_colorbar=dict(title='Project Count'),
        height=500,
        width=800
    )
    
    # Add text above bars showing project count
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    
    logger.info("Tier distribution chart created")
    return fig
