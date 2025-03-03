"""
Graph Neural Network implementation for Ethereum ecosystem analysis.
This module provides GNN-based node representations and importance scoring for GitHub repositories.
"""

import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Set, Optional, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger("gnn_model")

class GraphConvLayer(nn.Module):
    """
    Simple graph convolutional layer.
    This layer applies a linear transformation followed by a non-linear activation,
    with message passing between connected nodes in the graph.
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        Initialize a graph convolutional layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GCN layer.
        
        Args:
            x: Node features tensor of shape [num_nodes, in_features]
            adj: Adjacency matrix tensor of shape [num_nodes, num_nodes]
            
        Returns:
            Updated node features of shape [num_nodes, out_features]
        """
        # Apply linear transformation
        support = self.linear(x)
        
        # Message passing (matrix multiplication with adjacency matrix)
        output = torch.matmul(adj, support)
        
        # Apply non-linear activation
        return F.relu(output)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    This layer applies attention mechanisms to weight the importance
    of neighboring nodes during message passing.
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        """
        Initialize a graph attention layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            dropout: Dropout probability
            alpha: LeakyReLU slope
        """
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        # Transformation matrix W
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention mechanism
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        # Initialize with Glorot (Xavier) method
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        # LeakyReLU activation
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GAT layer.
        
        Args:
            x: Node features tensor of shape [num_nodes, in_features]
            adj: Adjacency matrix tensor of shape [num_nodes, num_nodes]
            
        Returns:
            Updated node features of shape [num_nodes, out_features]
        """
        # Apply linear transformation to get "embeddings"
        h = self.W(x)  # [N, out_features]
        N = h.size()[0]
        
        # Prepare attention mechanism inputs
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        a_input = a_input.view(N, N, 2 * self.out_features)
        
        # Calculate attention coefficients
        e = self.leakyrelu(self.a(a_input).squeeze(2))
        
        # Mask attention coefficients using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax to get normalized attention coefficients
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # Apply attention coefficients to features
        h_prime = torch.matmul(attention, h)
        
        return F.elu(h_prime)


class MultiHeadGraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Layer.
    This advanced GAT implementation uses multiple attention heads,
    allowing the model to jointly attend to information from different
    representation subspaces.
    """
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, 
                 dropout: float = 0.2, alpha: float = 0.2, concat: bool = True):
        """
        Initialize a multi-head graph attention layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features per head
            n_heads: Number of attention heads
            dropout: Dropout probability
            alpha: LeakyReLU slope
            concat: Whether to concatenate or average the multi-head results
        """
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        
        # Create multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha) 
            for _ in range(n_heads)
        ])
        
        # Dropout layer (applied to the input before passing to attention heads)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-head GAT layer.
        
        Args:
            x: Node features tensor of shape [num_nodes, in_features]
            adj: Adjacency matrix tensor of shape [num_nodes, num_nodes]
            
        Returns:
            Updated node features
        """
        x = self.dropout(x)
        
        # Apply each attention head
        if self.concat:
            # Concatenate the outputs from each head along the feature dimension
            return torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            # Average the outputs from each head
            return torch.mean(torch.stack([att(x, adj) for att in self.attentions]), dim=0)


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer inspired by the Transformer architecture.
    This sophisticated layer combines self-attention mechanisms with 
    feed-forward neural networks and residual connections.
    """
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, 
                 dropout: float = 0.1, alpha: float = 0.2):
        """
        Initialize a graph transformer layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features 
            n_heads: Number of attention heads
            dropout: Dropout probability
            alpha: LeakyReLU slope
        """
        super(GraphTransformerLayer, self).__init__()
        
        # Multi-head attention layer
        self.attention = MultiHeadGraphAttentionLayer(
            in_features, 
            out_features // n_heads,  # Split output features among heads
            n_heads=n_heads,
            dropout=dropout,
            alpha=alpha,
            concat=True
        )
        
        # Normalization layers for residual connections
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(out_features, out_features * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 2, out_features)
        )
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # In case input and output dimensions differ
        self.residual_connection = nn.Linear(in_features, out_features) if in_features != out_features else None
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the graph transformer layer.
        
        Args:
            x: Node features tensor of shape [num_nodes, in_features]
            adj: Adjacency matrix tensor of shape [num_nodes, num_nodes]
            
        Returns:
            Updated node features of shape [num_nodes, out_features]
        """
        # Prepare residual connection
        residual = x if self.residual_connection is None else self.residual_connection(x)
        
        # Multi-head attention with residual connection and normalization
        attn_output = self.attention(x, adj)
        x = self.norm1(residual + self.dropout1(attn_output))
        
        # Feed-forward network with residual connection and normalization
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout2(ff_output))
        
        return output

class GNN(nn.Module):
    """
    Graph Neural Network for node importance prediction.
    
    This GNN model takes a graph structure and node features as input,
    and predicts importance scores for each node based on both the
    graph structure and node attributes. It uses a hybrid architecture
    with both standard GCN layers and attention-based GAT layers.
    """
    
    def __init__(
        self, 
        feature_dim: int, 
        hidden_dim: int = 64, 
        output_dim: int = 32, 
        num_layers: int = 2,
        use_attention: bool = True,
        dropout: float = 0.2
    ):
        """
        Initialize the GNN model.
        
        Args:
            feature_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_layers: Number of GNN layers
            use_attention: Whether to use attention mechanism in some layers
            dropout: Dropout rate for attention layers
        """
        super(GNN, self).__init__()
        
        self.use_attention = use_attention
        self.dropout = dropout
        
        # First layer: input features to hidden (always use standard GCN for stability)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvLayer(feature_dim, hidden_dim))
        
        # Hidden layers - alternate between GCN and GAT if attention is enabled
        for i in range(num_layers - 2):
            if use_attention and i % 2 == 1:  # Use attention in odd-numbered layers
                self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout))
            else:
                self.layers.append(GraphConvLayer(hidden_dim, hidden_dim))
        
        # Last layer: hidden to output (use attention for final layer if enabled)
        if num_layers > 1:
            if use_attention:
                self.layers.append(GraphAttentionLayer(hidden_dim, output_dim, dropout=dropout))
            else:
                self.layers.append(GraphConvLayer(hidden_dim, output_dim))
            
        # Prediction head for importance score with dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features tensor of shape [num_nodes, feature_dim]
            adj: Adjacency matrix tensor of shape [num_nodes, num_nodes]
            
        Returns:
            Tuple of (node embeddings, importance scores)
        """
        # Pass through GNN layers
        for layer in self.layers:
            x = layer(x, adj)
        
        # Node embeddings
        embeddings = x
        
        # Predict importance scores
        scores = self.predictor(embeddings).squeeze(-1)
        
        return embeddings, scores


class AdvancedGNN(nn.Module):
    """
    Advanced Graph Neural Network with transformer-like architecture.
    
    This enhanced GNN model uses multi-head attention and transformer layers
    for more sophisticated graph representation learning. It's designed to 
    capture complex relationships in blockchain dependency graphs.
    """
    
    def __init__(
        self, 
        feature_dim: int, 
        hidden_dim: int = 128, 
        output_dim: int = 64, 
        num_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        activation: str = "gelu"
    ):
        """
        Initialize the advanced GNN model.
        
        Args:
            feature_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_layers: Number of transformer layers
            n_heads: Number of attention heads in multi-head attention
            dropout: Dropout rate for regularization
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            activation: Activation function ('relu', 'gelu', or 'elu')
        """
        super(AdvancedGNN, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Feature projection layer
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Choose activation function
        if activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GraphTransformerLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    alpha=0.2
                )
            )
        
        # Output projection layer
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Prediction heads
        self.importance_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 1)
        )
        
        # Explainability head for attention visualization
        self.explainability_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            self.activation,
            nn.Linear(output_dim // 2, 1),
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, 
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the advanced GNN.
        
        Args:
            x: Node features tensor of shape [num_nodes, feature_dim]
            adj: Adjacency matrix tensor of shape [num_nodes, num_nodes]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - embeddings: Node embeddings
                - importance_scores: Node importance scores
                - explainability_scores: Explainability scores
                - attention_weights: Attention weights (if return_attention=True)
        """
        # Initial feature projection
        h = self.feature_proj(x)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Store attention weights for explainability if requested
        attention_weights = []
        
        # Apply transformer layers
        for layer in self.layers:
            h_prev = h
            h = layer(h, adj)
            
            # Optional: Store attention weights
            if return_attention and hasattr(layer, 'attention') and hasattr(layer.attention, 'attentions'):
                attention_weights.append(layer.attention)
        
        # Final projection
        embeddings = self.output_proj(h)
        
        # Compute importance scores
        importance_scores = self.importance_head(embeddings).squeeze(-1)
        
        # Compute explainability scores
        explainability_scores = self.explainability_head(embeddings).squeeze(-1)
        
        # Prepare return dictionary
        result = {
            'embeddings': embeddings,
            'importance_scores': importance_scores,
            'explainability_scores': explainability_scores
        }
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result
    
    def get_attention_maps(self, x: torch.Tensor, adj: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention maps for explainability.
        
        Args:
            x: Node features tensor
            adj: Adjacency matrix tensor
            
        Returns:
            List of attention maps for each layer
        """
        with torch.no_grad():
            result = self.forward(x, adj, return_attention=True)
            return result.get('attention_weights', [])

def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Normalize adjacency matrix for GNN.
    
    Args:
        adj: Adjacency matrix tensor
        
    Returns:
        Normalized adjacency matrix
    """
    # Add self-loops
    identity = torch.eye(adj.size(0), device=adj.device)
    adj_with_self = adj + identity
    
    # Calculate degree matrix
    rowsum = adj_with_self.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_with_self), d_mat_inv_sqrt)
    
    return normalized_adj

def prepare_graph_data(
    G: nx.DiGraph, 
    github_features: Dict[str, Dict[str, float]]
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, str], Dict[str, int]]:
    """
    Prepare graph data for GNN processing.
    
    Args:
        G: NetworkX graph
        github_features: Dictionary mapping nodes to feature dictionaries
        
    Returns:
        Tuple of (adjacency matrix tensor, feature matrix tensor, node index to name mapping, name to index mapping)
    """
    logger.info("Preparing graph data for GNN processing...")
    
    # Get node list and create mappings
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}
    
    # Create adjacency matrix
    adj = nx.adjacency_matrix(G, nodelist=nodes).todense()
    adj = torch.FloatTensor(adj)
    
    # Normalize adjacency matrix
    adj = normalize_adjacency(adj)
    
    # Extract features for each node
    feature_names = set()
    for node in nodes:
        if node in github_features:
            feature_names.update(github_features[node].keys())
    
    feature_names = sorted(list(feature_names))
    feature_idx = {name: i for i, name in enumerate(feature_names)}
    
    # Create feature matrix
    features = torch.zeros((len(nodes), len(feature_names)))
    for i, node in enumerate(nodes):
        if node in github_features:
            node_features = github_features[node]
            for feat_name, feat_value in node_features.items():
                if feat_name in feature_idx:
                    features[i, feature_idx[feat_name]] = feat_value
    
    # If no features, use identity matrix
    if len(feature_names) == 0:
        features = torch.eye(len(nodes))
    
    # Apply feature normalization/scaling
    for j in range(features.shape[1]):
        col = features[:, j]
        if col.max() > col.min():  # Only scale if there's variation in values
            features[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-8)  # Add epsilon to avoid division by zero
    
    logger.info(f"Prepared data: {len(nodes)} nodes, {features.shape[1]} features")
    return adj, features, idx_to_node, node_to_idx

def train_gnn(
    adj: torch.Tensor, 
    features: torch.Tensor, 
    reference_scores: Optional[torch.Tensor] = None,
    epochs: int = 200,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    use_attention: bool = True,
    hidden_dim: int = 64,
    output_dim: int = 32,
    num_layers: int = 2,
    dropout: float = 0.2
) -> GNN:
    """
    Train a GNN model on the graph.
    
    Args:
        adj: Adjacency matrix tensor
        features: Feature matrix tensor
        reference_scores: Optional reference scores for supervised training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        use_attention: Whether to use attention mechanism
        hidden_dim: Size of hidden dimension
        output_dim: Size of output dimension
        num_layers: Number of GNN layers
        dropout: Dropout rate for regularization
        
    Returns:
        Trained GNN model
    """
    # Create model with specified architecture
    model = GNN(
        features.shape[1], 
        hidden_dim=hidden_dim, 
        output_dim=output_dim, 
        num_layers=num_layers,
        use_attention=use_attention,
        dropout=dropout
    )
    
    # Use either supervised or self-supervised training
    if reference_scores is not None:
        logger.info("Training GNN with supervised learning...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            _, scores = model(features, adj)
            loss = F.mse_loss(scores, reference_scores)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")
                
            loss.backward()
            optimizer.step()
    else:
        logger.info("Training GNN with self-supervised learning...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Self-supervised training using graph structure as supervision
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings, _ = model(features, adj)
            
            # Reconstruction loss: node embeddings should predict adjacency
            pred_adj = torch.mm(embeddings, embeddings.t())
            loss = F.binary_cross_entropy_with_logits(pred_adj, adj)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")
                
            loss.backward()
            optimizer.step()
    
    model.eval()
    return model

def train_advanced_gnn(
    adj: torch.Tensor, 
    features: torch.Tensor, 
    reference_scores: Optional[torch.Tensor] = None,
    epochs: int = 300,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    hidden_dim: int = 128,
    output_dim: int = 64,
    num_layers: int = 3,
    n_heads: int = 4,
    dropout: float = 0.1
) -> AdvancedGNN:
    """
    Train the advanced GNN model on the graph.
    
    Args:
        adj: Adjacency matrix tensor
        features: Feature matrix tensor
        reference_scores: Optional reference scores for supervised training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        hidden_dim: Size of hidden dimension
        output_dim: Size of output dimension
        num_layers: Number of transformer layers
        n_heads: Number of attention heads
        dropout: Dropout rate for regularization
        
    Returns:
        Trained AdvancedGNN model
    """
    # Create model with specified architecture
    model = AdvancedGNN(
        feature_dim=features.shape[1],
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        n_heads=n_heads,
        dropout=dropout,
        use_layer_norm=True,
        use_residual=True,
        activation="gelu"
    )
    
    # Configure optimizer with learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # Supervised or self-supervised training
    if reference_scores is not None:
        logger.info("Training Advanced GNN with supervised learning...")
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(features, adj)
            loss = F.mse_loss(outputs['importance_scores'], reference_scores)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")
                
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
    else:
        logger.info("Training Advanced GNN with self-supervised learning...")
        model.train()
        
        # Track best model
        best_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(features, adj)
            
            # Reconstruction loss combined with embedding smoothness loss
            pred_adj = torch.mm(outputs['embeddings'], outputs['embeddings'].t())
            reconstruction_loss = F.binary_cross_entropy_with_logits(pred_adj, adj)
            
            # Smoothness: connected nodes should have similar embeddings
            d_flat = adj.view(-1)
            emb_prod = outputs['embeddings'].mm(outputs['embeddings'].t())
            emb_prod_flat = emb_prod.view(-1)
            mask = (d_flat != 0).float()
            smoothness_loss = F.mse_loss(emb_prod_flat * mask, d_flat * mask) 
            
            # Explainability loss: encourage interpretable attention patterns
            explainability_loss = torch.mean(torch.abs(outputs['explainability_scores']))
            
            # Combined loss
            loss = reconstruction_loss + 0.1 * smoothness_loss + 0.01 * explainability_loss
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f} "
                           f"(Recon: {reconstruction_loss:.4f}, Smooth: {smoothness_loss:.4f})")
                
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # Save best model
            if loss < best_loss:
                best_loss = loss
                best_model_state = model.state_dict().copy()
        
        # Load best model if available
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    
    model.eval()
    return model

def advanced_gnn_node_importance(
    G: nx.DiGraph,
    github_features: Dict[str, Dict[str, float]],
    reference_scores: Optional[Dict[str, float]] = None,
    model_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate node importance using the Advanced GNN model with transformer layers.
    
    Args:
        G: NetworkX graph
        github_features: Dictionary mapping nodes to feature dictionaries
        reference_scores: Optional dictionary of reference scores for supervision
        model_params: Optional dictionary of model parameters
        
    Returns:
        Dictionary containing:
            - importance_scores: Dictionary mapping node names to importance scores
            - explainability_scores: Dictionary mapping node names to explainability scores
    """
    logger.info("Calculating Advanced GNN-based node importance...")
    
    # Prepare data
    adj, features, idx_to_node, node_to_idx = prepare_graph_data(G, github_features)
    
    # Prepare reference scores if provided
    ref_tensor = None
    if reference_scores is not None:
        ref_tensor = torch.zeros(len(node_to_idx))
        for node, score in reference_scores.items():
            if node in node_to_idx:
                ref_tensor[node_to_idx[node]] = score
    
    # Set default model parameters if not provided
    if model_params is None:
        model_params = {
            'hidden_dim': 128,
            'output_dim': 64,
            'num_layers': 3,
            'n_heads': 4,
            'dropout': 0.1,
            'epochs': 300
        }
    
    # Train advanced model
    try:
        model = train_advanced_gnn(
            adj, 
            features, 
            ref_tensor,
            epochs=model_params.get('epochs', 300),
            hidden_dim=model_params.get('hidden_dim', 128),
            output_dim=model_params.get('output_dim', 64),
            num_layers=model_params.get('num_layers', 3),
            n_heads=model_params.get('n_heads', 4),
            dropout=model_params.get('dropout', 0.1)
        )
        
        # Get importance and explainability scores
        with torch.no_grad():
            outputs = model(features, adj)
            importance_scores = torch.sigmoid(outputs['importance_scores']).numpy()
            explainability_scores = torch.sigmoid(outputs['explainability_scores']).numpy()
        
        # Map scores back to node names
        importance_dict = {}
        explainability_dict = {}
        for idx in range(len(importance_scores)):
            node = idx_to_node[idx]
            importance_dict[node] = float(importance_scores[idx])
            explainability_dict[node] = float(explainability_scores[idx])
        
        # Normalize scores
        sum_importance = sum(importance_dict.values())
        if sum_importance > 0:
            importance_dict = {k: v / sum_importance for k, v in importance_dict.items()}
        
        return {
            'importance_scores': importance_dict,
            'explainability_scores': explainability_dict
        }
    
    except Exception as e:
        logger.error(f"Error training advanced GNN: {str(e)}")
        logger.info("Falling back to standard GNN for node importance calculation...")
        # Fall back to standard GNN if the advanced one fails
        return {'importance_scores': gnn_node_importance(G, github_features, reference_scores)}

def gnn_node_importance(
    G: nx.DiGraph,
    github_features: Dict[str, Dict[str, float]],
    reference_scores: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Calculate node importance using a Graph Neural Network.
    
    Args:
        G: NetworkX graph
        github_features: Dictionary mapping nodes to feature dictionaries
        reference_scores: Optional dictionary of reference scores for supervision
        
    Returns:
        Dictionary mapping node names to importance scores
    """
    logger.info("Calculating GNN-based node importance...")
    
    # Prepare data
    adj, features, idx_to_node, node_to_idx = prepare_graph_data(G, github_features)
    
    # Prepare reference scores if provided
    ref_tensor = None
    if reference_scores is not None:
        ref_tensor = torch.zeros(len(node_to_idx))
        for node, score in reference_scores.items():
            if node in node_to_idx:
                ref_tensor[node_to_idx[node]] = score
    
    # Train model
    model = train_gnn(adj, features, ref_tensor)
    
    # Get importance scores
    with torch.no_grad():
        _, scores = model(features, adj)
        scores = torch.sigmoid(scores).numpy()  # Normalize to [0,1]
    
    # Map scores back to node names
    importance_scores = {}
    for idx, score in enumerate(scores):
        node = idx_to_node[idx]
        importance_scores[node] = float(score)
    
    # Normalize scores to sum to 1
    total = sum(importance_scores.values())
    if total > 0:
        for node in importance_scores:
            importance_scores[node] /= total
    
    logger.info("GNN node importance calculation complete")
    return importance_scores

def get_node_embeddings(
    G: nx.DiGraph,
    github_features: Dict[str, Dict[str, float]]
) -> Dict[str, np.ndarray]:
    """
    Get node embeddings from a trained GNN.
    
    Args:
        G: NetworkX graph
        github_features: Dictionary mapping nodes to feature dictionaries
        
    Returns:
        Dictionary mapping node names to embedding vectors
    """
    logger.info("Generating GNN node embeddings...")
    
    # Prepare data
    adj, features, idx_to_node, node_to_idx = prepare_graph_data(G, github_features)
    
    # Train model
    model = train_gnn(adj, features)
    
    # Get embeddings
    with torch.no_grad():
        embeddings, _ = model(features, adj)
        embeddings = embeddings.numpy()
    
    # Map embeddings back to node names
    node_embeddings = {}
    for idx, emb in enumerate(embeddings):
        node = idx_to_node[idx]
        node_embeddings[node] = emb
    
    logger.info(f"Generated embeddings with dimension {embeddings.shape[1]} for {len(node_embeddings)} nodes")
    return node_embeddings

def optimize_gnn_parameters(
    G: nx.DiGraph,
    github_features: Dict[str, Dict[str, float]],
    reference_scores: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Optimize GNN parameters for better performance on the given graph.
    
    Args:
        G: NetworkX graph
        github_features: Dictionary mapping nodes to feature dictionaries
        reference_scores: Optional reference scores for supervised training
        
    Returns:
        Dictionary of optimal parameters
    """
    logger.info("Optimizing GNN parameters...")
    
    # Prepare data
    adj, features, idx_to_node, node_to_idx = prepare_graph_data(G, github_features)
    
    # Prepare reference scores if provided
    ref_tensor = None
    if reference_scores is not None:
        ref_tensor = torch.zeros(len(node_to_idx))
        for node, score in reference_scores.items():
            if node in node_to_idx:
                ref_tensor[node_to_idx[node]] = score
    
    # Parameter configurations to try
    param_configs = [
        {"use_attention": True, "num_layers": 2, "hidden_dim": 64, "dropout": 0.2},
        {"use_attention": False, "num_layers": 2, "hidden_dim": 64, "dropout": 0.1},
        {"use_attention": True, "num_layers": 3, "hidden_dim": 128, "dropout": 0.3},
    ]
    
    best_loss = float('inf')
    best_params = param_configs[0]
    
    # Try different parameter configurations
    for params in param_configs:
        logger.info(f"Trying parameters: {params}")
        try:
            model = GNN(
                features.shape[1], 
                hidden_dim=params["hidden_dim"], 
                output_dim=32, 
                num_layers=params["num_layers"],
                use_attention=params["use_attention"],
                dropout=params["dropout"]
            )
            
            # Training configurations
            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            epochs = 50  # Reduced for optimization search
            
            # Simple training loop
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                if ref_tensor is not None:
                    # Supervised training
                    _, scores = model(features, adj)
                    loss = F.mse_loss(scores, ref_tensor)
                else:
                    # Self-supervised training
                    embeddings, _ = model(features, adj)
                    pred_adj = torch.mm(embeddings, embeddings.t())
                    loss = F.binary_cross_entropy_with_logits(pred_adj, adj)
                
                loss.backward()
                optimizer.step()
            
            # Evaluate final loss
            model.eval()
            with torch.no_grad():
                if ref_tensor is not None:
                    _, scores = model(features, adj)
                    final_loss = F.mse_loss(scores, ref_tensor).item()
                else:
                    embeddings, _ = model(features, adj)
                    pred_adj = torch.mm(embeddings, embeddings.t())
                    final_loss = F.binary_cross_entropy_with_logits(pred_adj, adj).item()
            
            logger.info(f"Config {params} achieved loss: {final_loss:.4f}")
            
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = params
                
        except Exception as e:
            logger.warning(f"Failed to train with parameters {params}: {str(e)}")
    
    logger.info(f"Optimal parameters: {best_params}, Loss: {best_loss:.4f}")
    return best_params


def apply_gnn_funding_allocation(
    G: nx.DiGraph, 
    github_features: Dict[str, Dict[str, float]],
    total_funding: float = 1.0,
    optimize_params: bool = True
) -> Dict[str, float]:
    """
    Apply GNN-based funding allocation.
    
    Args:
        G: NetworkX graph
        github_features: Dictionary mapping nodes to feature dictionaries
        total_funding: Total funding amount to allocate
        optimize_params: Whether to automatically optimize GNN parameters
        
    Returns:
        Dictionary mapping node names to funding amounts
    """
    # Optionally optimize parameters
    if optimize_params and len(list(G.nodes())) < 500:  # Only optimize for reasonably sized graphs
        logger.info("Optimizing GNN parameters for funding allocation...")
        # Use PageRank as reference for supervised optimization
        pagerank_scores = nx.pagerank(G, alpha=0.85)
        optimal_params = optimize_gnn_parameters(G, github_features, pagerank_scores)
        
        # Get importance scores from GNN with optimal parameters
        adj, features, idx_to_node, node_to_idx = prepare_graph_data(G, github_features)
        model = train_gnn(
            adj, 
            features, 
            hidden_dim=optimal_params.get("hidden_dim", 64),
            num_layers=optimal_params.get("num_layers", 2),
            use_attention=optimal_params.get("use_attention", True),
            dropout=optimal_params.get("dropout", 0.2)
        )
        
        with torch.no_grad():
            _, scores = model(features, adj)
            scores = torch.sigmoid(scores).numpy()
        
        # Map scores to nodes
        importance_scores = {}
        for idx, score in enumerate(scores):
            node = idx_to_node[idx]
            importance_scores[node] = float(score)
        
        # Normalize to sum to 1
        total = sum(importance_scores.values())
        if total > 0:
            for node in importance_scores:
                importance_scores[node] /= total
    else:
        # Use standard approach for large graphs or when optimization is disabled
        importance_scores = gnn_node_importance(G, github_features)
    
    # Allocate funding based on importance scores
    funding_allocation = {}
    for node, score in importance_scores.items():
        funding_allocation[node] = score * total_funding
    
    return funding_allocation

def compare_allocation_methods(
    G: nx.DiGraph,
    github_features: Dict[str, Dict[str, float]],
    pagerank_scores: Dict[str, float]
) -> pd.DataFrame:
    """
    Compare different allocation methods.
    
    Args:
        G: NetworkX graph
        github_features: Dictionary mapping nodes to feature dictionaries
        pagerank_scores: PageRank scores for comparison
        
    Returns:
        DataFrame with comparison of different scoring methods
    """
    # Calculate GNN scores
    gnn_scores = gnn_node_importance(G, github_features)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        "repository": list(G.nodes()),
        "pagerank_score": [pagerank_scores.get(node, 0) for node in G.nodes()],
        "gnn_score": [gnn_scores.get(node, 0) for node in G.nodes()]
    })
    
    # Calculate correlation
    correlation = comparison[["pagerank_score", "gnn_score"]].corr().iloc[0, 1]
    logger.info(f"Correlation between PageRank and GNN scores: {correlation:.4f}")
    
    return comparison


def identify_unsung_heroes(
    G: nx.DiGraph,
    github_features: Dict[str, Dict[str, float]],
    pagerank_scores: Dict[str, float],
    threshold_percentile: float = 90
) -> List[Dict[str, Any]]:
    """
    Identify 'unsung hero' repositories that are ranked much higher by GNN than by PageRank.
    These are potentially undervalued projects that contribute significantly to the ecosystem
    but don't receive proportional recognition.
    
    Args:
        G: NetworkX graph
        github_features: Dictionary mapping nodes to feature dictionaries
        pagerank_scores: PageRank scores for comparison
        threshold_percentile: Percentile threshold for difference in rankings
        
    Returns:
        List of dictionaries with information about unsung hero repositories
    """
    logger.info(f"Identifying unsung hero repositories using GNN analysis...")
    
    # Get GNN scores
    gnn_scores = gnn_node_importance(G, github_features)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        "repository": list(G.nodes()),
        "pagerank_score": [pagerank_scores.get(node, 0) for node in G.nodes()],
        "pagerank_rank": 0,  # Will be filled in
        "gnn_score": [gnn_scores.get(node, 0) for node in G.nodes()],
        "gnn_rank": 0,  # Will be filled in
    })
    
    # Compute ranks
    comparison['pagerank_rank'] = comparison['pagerank_score'].rank(ascending=False)
    comparison['gnn_rank'] = comparison['gnn_score'].rank(ascending=False)
    
    # Calculate rank differences (positive means GNN ranks it higher than PageRank)
    comparison['rank_difference'] = comparison['pagerank_rank'] - comparison['gnn_rank']
    
    # Calculate the normalized score difference
    # First normalize each score column to [0, 1] range 
    if len(comparison) > 0:
        comparison['pagerank_normalized'] = (comparison['pagerank_score'] - comparison['pagerank_score'].min()) / \
                                          (comparison['pagerank_score'].max() - comparison['pagerank_score'].min() + 1e-8)
        comparison['gnn_normalized'] = (comparison['gnn_score'] - comparison['gnn_score'].min()) / \
                                     (comparison['gnn_score'].max() - comparison['gnn_score'].min() + 1e-8)
        
        # Calculate score difference (positive means GNN values it more than PageRank)
        comparison['score_difference'] = comparison['gnn_normalized'] - comparison['pagerank_normalized']
    
    # Identify unsung heroes - repositories with high positive rank difference
    threshold = np.percentile(comparison['rank_difference'], threshold_percentile)
    unsung_heroes = comparison[comparison['rank_difference'] > threshold].sort_values('rank_difference', ascending=False)
    
    # Prepare detailed results
    results = []
    for _, row in unsung_heroes.iterrows():
        repo = row['repository']
        # Get GitHub metrics if available
        metrics = {}
        if repo in github_features:
            metrics = github_features[repo]
        
        # Get graph metrics
        in_degree = G.in_degree(repo) if repo in G else 0
        out_degree = G.out_degree(repo) if repo in G else 0
        
        # Create result entry
        result = {
            "repository": repo,
            "pagerank_score": row['pagerank_score'],
            "pagerank_rank": int(row['pagerank_rank']),
            "gnn_score": row['gnn_score'],
            "gnn_rank": int(row['gnn_rank']),
            "rank_difference": int(row['rank_difference']),
            "score_difference": float(row['score_difference']),
            "in_degree": in_degree,
            "out_degree": out_degree,
            "github_metrics": metrics
        }
        results.append(result)
    
    logger.info(f"Identified {len(results)} unsung hero repositories")
    return results

if __name__ == "__main__":
    # Example usage
    import networkx as nx
    
    # Create a sample graph
    G = nx.DiGraph()
    G.add_edges_from([
        ("ethereum/go-ethereum", "ethereum/solidity"),
        ("ethereum/go-ethereum", "ethereum/web3.js"),
        ("ethereum/solidity", "ethereum/tests"),
        ("ethereum/web3.js", "ethereum/ethers.js")
    ])
    
    # Sample features
    features = {
        "ethereum/go-ethereum": {"stars": 0.9, "forks": 0.8, "activity": 0.7},
        "ethereum/solidity": {"stars": 0.7, "forks": 0.6, "activity": 0.9},
        "ethereum/web3.js": {"stars": 0.5, "forks": 0.4, "activity": 0.6},
        "ethereum/tests": {"stars": 0.3, "forks": 0.2, "activity": 0.4},
        "ethereum/ethers.js": {"stars": 0.6, "forks": 0.5, "activity": 0.8}
    }
    
    # Calculate node importance
    importance = gnn_node_importance(G, features)
    print("Node importance:")
    for node, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node}: {score:.4f}")