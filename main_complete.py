import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import base64
import io
import os
import time
from PIL import Image

# Set page config first, before any Streamlit element is shown
st.set_page_config(
    page_title="Crypto_ParadoxOS",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

from graph_processor import (
    load_dependency_graph, 
    calculate_pagerank, 
    calculate_weighted_contribution,
    apply_tiered_weighting,
    identify_critical_dependencies
)
from github_metrics import (
    extract_github_metrics,
    normalize_github_features
)
from blockchain_data_generator import (
    ensure_blockchain_sample_data_exists,
    generate_blockchain_dependency_csv,
    generate_all_blockchain_samples
)
from github_data_builder import GitHubDataBuilder
from model import (
    train_ranking_model,
    predict_funding_allocation,
    evaluate_model
)
from visualization import (
    create_dependency_graph_visualization,
    create_funding_allocation_chart,
    create_project_importance_heatmap,
    create_comparison_chart,
    create_gnn_relationship_visualization
)
from utils import (
    export_results_to_csv,
    compute_validation_metrics
)
from gnn_model import (
    gnn_node_importance,
    advanced_gnn_node_importance,
    get_node_embeddings,
    apply_gnn_funding_allocation,
    compare_allocation_methods,
    identify_unsung_heroes,
    optimize_gnn_parameters
)
from blockchain_manager import (
    BlockchainManager,
    BlockchainConfig,
    BlockchainAdapter,
    EnhancedBlockchainAdapterFactory
)
from web_scraper import (
    scrape_github_repository,
    scrape_github_repositories_batch,
    extract_repository_metrics,
    normalize_scraper_metrics
)

# Initialize blockchain manager
blockchain_manager = BlockchainManager()

# Check if blockchain configurations exist
if len(blockchain_manager.get_all_blockchains()) == 0:
    blockchain_manager.add_default_blockchains()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.1rem;
        font-weight: 400;
        color: #424242;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
    }
    .note {
        padding: 1rem;
        border-left: 4px solid #1E88E5;
        background-color: #E3F2FD;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    .blockchain-selector {
        margin-bottom: 1.5rem;
    }
    .blockchain-logo {
        text-align: center;
        margin-bottom: 1rem;
    }
    .blockchain-info {
        padding: 0.75rem;
        background-color: #f1f8fe;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Get blockchain list for selection
blockchain_options = blockchain_manager.get_blockchain_list()

# Default to Ethereum if available, otherwise first blockchain
default_idx = blockchain_ids.index('ethereum') if 'ethereum' in blockchain_ids else 0

# Initialize session state for selected blockchain if not set
if 'selected_blockchain_id' not in st.session_state:

# Get the selected blockchain config
selected_blockchain = blockchain_manager.get_blockchain(selected_blockchain_id)

# Main header
st.markdown("<h1 class='main-header'>Crypto_ParadoxOS</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-driven funding allocation for blockchain open-source ecosystems powered by Paradoxical OS Innovation Strategy</p>", unsafe_allow_html=True)

# Blockchain selector in main UI
with col2:
    st.markdown("<div class='blockchain-selector'>", unsafe_allow_html=True)
    selected_index = st.selectbox(
        "Select Blockchain Ecosystem",
        options=range(len(blockchain_options)),
        index=blockchain_ids.index(selected_blockchain_id)
    )
    
    # Update the selected blockchain when changed
        selected_blockchain = blockchain_manager.get_blockchain(selected_blockchain_id)
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='blockchain-logo'>", unsafe_allow_html=True)
    st.image(selected_blockchain.logo_url, width=150)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display blockchain info
    st.markdown("<div class='blockchain-info'>", unsafe_allow_html=True)
    st.markdown(f"### {selected_blockchain.display_name}")
    st.markdown(f"*{selected_blockchain.description}*")
    st.markdown(f"**Primary Language:** {selected_blockchain.primary_language}")
    if selected_blockchain.year_founded:
        st.markdown(f"**Founded:** {selected_blockchain.year_founded}")
    if selected_blockchain.website:
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("## Configuration")
    
    # Data upload section
    st.markdown("### Data Input")
    
    use_sample_data = st.checkbox("Use sample data", value=True)
    
    # Parameters section
    st.markdown("### Parameters")
    pagerank_alpha = st.slider("PageRank Alpha", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    contribution_weight = st.slider("Contribution Weight", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    
    tiered_weighting = st.checkbox("Apply Tiered Weighting", value=True)
    max_tier_level = st.number_input("Max Tier Level", min_value=1, max_value=10, value=5)
    
    include_github_metrics = st.checkbox("Include GitHub Metrics", value=True)
    if include_github_metrics:
        github_data_source = st.radio(
            "GitHub Data Source",
            index=0,
            help="Choose data source: Simulated data (offline), GitHub API (requires token), or Web Scraping (no token needed)"
        )
        
        use_web_scraping = github_data_source == "Web Scraping"
        
        # Store in session state
        
        if github_data_source == "GitHub API" and not os.environ.get("GITHUB_TOKEN"):
            st.warning("⚠️ No GitHub token found. You may encounter rate limits. Consider adding a GITHUB_TOKEN to your environment variables.")
            
        if github_data_source == "Web Scraping":
            st.info("ℹ️ Web scraping will extract data directly from GitHub web pages. This is slower but doesn't require an API token.")
            
            # Add scraping limits/options
            max_repos_to_scrape = st.slider(
                "Max Repos to Scrape", 
                min_value=5, 
                max_value=100, 
                value=20,
                help="Limit the number of repositories to scrape to avoid timeouts or potential blocking"
            )
    
    # Model selection
    st.markdown("### Model Selection")
    
    # Run button
    run_analysis = st.button("Run Analysis", use_container_width=True)

# Main content

with tab1:
    st.markdown("<h2 class='section-header'>Data Explorer</h2>", unsafe_allow_html=True)
    
    # Ensure sample data exists for the selected blockchain
    sample_file = ensure_blockchain_sample_data_exists(selected_blockchain_id)
    
    # Load data
    if uploaded_file is not None:
        dependency_df = pd.read_csv(uploaded_file)
        use_sample_data = False
    elif use_sample_data:
        # Use generated sample data for selected blockchain
        dependency_df = pd.read_csv(sample_file)
        st.success(f"Loaded sample dependency data for {selected_blockchain.display_name} with {len(dependency_df)} relationships")
    else:
        st.warning("Please upload a dependency graph file or use sample data.")
        dependency_df = None
    
    if dependency_df is not None:
        # Display data overview
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Dependency Graph Overview")
        
        # Check for expected columns and display appropriate metrics
        if 'child' in dependency_df.columns and 'parent' in dependency_df.columns:
            st.write(f"Number of dependencies: {len(dependency_df)}")
        elif 'source' in dependency_df.columns and 'target' in dependency_df.columns:
            # Alternative column names
            st.write(f"Number of dependencies: {len(dependency_df)}")
        else:
            # Fallback for any column structure
            st.write(f"Number of rows: {len(dependency_df)}")
            st.write(f"Columns: {', '.join(dependency_df.columns)}")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display dataset
        st.markdown("#### Dependency Graph Data")
        st.dataframe(dependency_df, use_container_width=True)
        
        # Create and display graph
        if run_analysis or st.button("Generate Graph Visualization"):
            with st.spinner("Generating graph visualization..."):
                G = load_dependency_graph(dependency_df)
                fig = create_dependency_graph_visualization(G)
                st.plotly_chart(fig, use_container_width=True)
                
                # Graph metrics
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Graph Metrics")
                
                # Convert to lists to avoid type issues
                nodes_list = list(G.nodes())
                edges_list = list(G.edges())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Nodes", len(nodes_list))
                with col2:
                    st.metric("Number of Edges", len(edges_list))
                with col3:
                    st.metric("Average Degree", round(sum(degree_values) / len(nodes_list), 2) if nodes_list else 0)
                st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 class='section-header'>Model & Analysis</h2>", unsafe_allow_html=True)
    
    if dependency_df is not None and run_analysis:
        with st.spinner("Running analysis..."):
            # Build the graph
            G = load_dependency_graph(dependency_df)
            
            # Apply PageRank
            pagerank_scores = calculate_pagerank(G, alpha=pagerank_alpha)
            
            # Get GitHub metrics if enabled
            if include_github_metrics:
                # Initialize the GitHub data builder
                github_token = os.environ.get("GITHUB_TOKEN")
                use_real_github_data = st.session_state.get("use_real_github_data", False)
                use_web_scraping = st.session_state.get("use_web_scraping", False)
                
                # Convert G.nodes() to a list to ensure correct type
                
                # Limit the number of nodes for web scraping if needed
                if use_web_scraping and "max_repos_to_scrape" in st.session_state:
                    if len(nodes_list) > max_repos:
                        st.warning(f"Limiting web scraping to {max_repos} repositories (out of {len(nodes_list)} total)")
                        # Sort by PageRank score to prioritize important repos
                
                if use_real_github_data and use_web_scraping:
                    st.info("Using enhanced web scraping to fetch GitHub data")
                    # Use our custom web scraper
                    with st.spinner("Web scraping GitHub repositories..."):
                        # Get maximum repositories to scrape
                        max_repos = st.session_state.get("max_repos_to_scrape", 20)
                        
                        # Show progress bar
                        progress_bar = st.progress(0)
                        progress_bar.progress(5/100)
                        
                        # Scrape repositories using our custom scraper
                        scraped_data = scrape_github_repositories_batch(nodes_list, max_repos=max_repos)
                        progress_bar.progress(60/100)
                        
                        # Extract metrics from scraped data
                        raw_metrics = extract_repository_metrics(scraped_data)
                        progress_bar.progress(80/100)
                        
                        # Normalize metrics
                        github_metrics = normalize_scraper_metrics(raw_metrics)
                        progress_bar.progress(100/100)
                        
                        # Store in session state for later use
                elif use_real_github_data and github_token:
                    st.info("Using real GitHub API data (with token)")
                    github_builder = GitHubDataBuilder(token=github_token)
                    github_metrics = github_builder.extract_github_metrics_batch(nodes_list, use_cache=True)
                elif use_real_github_data:
                    st.warning("Using real GitHub API data (without token, rate limits apply)")
                    github_builder = GitHubDataBuilder()
                    github_metrics = github_builder.extract_github_metrics_batch(nodes_list, use_cache=True)
                else:
                    st.info("Using simulated GitHub data")
                    github_metrics = extract_github_metrics(nodes_list)
                
                github_features = normalize_github_features(github_metrics)
            else:
                github_features = None
            
            # Calculate weighted contribution
            weighted_scores = calculate_weighted_contribution(
                G, 
                pagerank_scores, 
                github_features, 
                contribution_weight
            )
            
            # Apply tiered weighting if enabled
            if tiered_weighting:
                final_scores = apply_tiered_weighting(G, weighted_scores, max_tier_level)
            else:
                final_scores = weighted_scores
            
            # Convert to DataFrame for display
            results_df = pd.DataFrame({
                'project': list(final_scores.keys()),
                'importance_score': list(final_scores.values())
            }).sort_values('importance_score', ascending=False)
            
            # Display PageRank results
            st.markdown("#### PageRank Results (Top 10)")
            pagerank_df = pd.DataFrame({
                'project': list(pagerank_scores.keys()),
                'pagerank_score': list(pagerank_scores.values())
            }).sort_values('pagerank_score', ascending=False).head(10)
            st.dataframe(pagerank_df, use_container_width=True)
            
            # Display final results
            st.markdown("#### Final Importance Scores (Top 20)")
            st.dataframe(results_df.head(20), use_container_width=True)
            
            # Train and evaluate model
            st.markdown("#### Model Training & Evaluation")
            
            # Simulated model training (in a real scenario, we would use real training data)
            model = train_ranking_model(X, y, model_type=model_type)
            
            # Predict funding allocation
            funding_allocation = predict_funding_allocation(model, results_df)
            
            # Evaluation metrics
            evaluation_metrics = evaluate_model(model, X, y)
            
            # Display metrics
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Model Performance Metrics")
            metrics_cols = st.columns(len(evaluation_metrics))
            for i, (metric, value) in enumerate(evaluation_metrics.items()):
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Advanced Analysis - GNN Implementation
            st.markdown("#### Advanced Graph Neural Network Analysis")
            
            with st.expander("Run GNN Analysis", expanded=True):
                st.markdown(f"""
                🧠 **Graph Neural Network (GNN) Analysis**
                
                This analysis uses a deep learning approach to understand project importance in the {selected_blockchain.display_name} ecosystem.
                Unlike PageRank which primarily considers link structure, GNNs can learn from both graph structure and node features.
                """)
                
                # Add an explanation for any errors the user might encounter
                st.info("""
                ℹ️ **GNN Analysis Steps**:
                
                1. Make sure you've loaded dependency data and clicked "Run Analysis" first
                2. Ensure "Include GitHub Metrics" is enabled for best results
                3. Click the button below to run the GNN model
                4. Go to the Visualizations tab to see detailed results including "Unsung Heroes"
                """)
                
                # Create tabs for different GNN models
                
                    run_gnn = st.button("Run Standard GNN Analysis", key="run_std_gnn_button")
                    
                    if run_gnn:
                        with st.spinner("Training Standard Graph Neural Network..."):
                            if github_features:
                                try:
                                    # Run GNN analysis
                                    st.info("Training Standard GNN model on repository graph and features...")
                                    
                                    # Ensure the Graph is loaded
                                    if G is None:
                                        st.error("Graph is not properly loaded. Please ensure you've loaded dependency data.")
                                        st.stop()
                                        
                                    try:
                                        # Check if graph has nodes
                                        num_nodes = len(list(G.nodes()))
                                        if num_nodes == 0:
                                            st.error("Graph has no nodes. Please ensure dependency data is valid.")
                                            st.stop()
                                    except Exception as e:
                                        st.error(f"Error checking graph: {str(e)}")
                                        st.stop()
                                        
                                    # Make sure we have GitHub features    
                                    if not github_features:
                                        st.warning("No GitHub features available. Using generated sample features.")
                                        # Generate simple features for each node
                                        github_features = {}
                                        for node in G.nodes():
                                            # Create random but consistent features
                                            import hashlib
                                            hash_val = int(hashlib.md5(str(node).encode()).hexdigest(), 16) % 1000
                                                'stars': (hash_val % 100) / 100.0,
                                                'forks': (hash_val % 50) / 100.0,
                                                'activity': (hash_val % 75) / 100.0
                                            }
                                    
                                    # Run GNN node importance analysis
                                    gnn_scores = gnn_node_importance(G, github_features, reference_scores=pagerank_scores)
                                    
                                    # Store in session state
                                    st.session_state.gnn_scores = gnn_scores
                                    
                                    # Compare allocation methods
                                    comparison_df = compare_allocation_methods(G, github_features, pagerank_scores)
                                    st.session_state.comparison_df = comparison_df
                                    
                                    # Display GNN results
                                    gnn_df = pd.DataFrame({
                                        'project': list(gnn_scores.keys()),
                                        'gnn_score': list(gnn_scores.values())
                                    }).sort_values('gnn_score', ascending=False).head(10)
                                    
                                    st.markdown("#### Standard GNN Importance Scores (Top 10)")
                                    st.dataframe(gnn_df, use_container_width=True)
                                    
                                    # Create side-by-side comparison
                                    st.markdown("#### PageRank vs Standard GNN Score Comparison (Top 15)")
                                    comparison_top = comparison_df.sort_values('gnn_score', ascending=False).head(15)
                                    
                                    st.success("Standard GNN analysis complete!")
                                    
                                except Exception as e:
                                    st.error(f"Error in Standard GNN analysis: {str(e)}")
                                    import traceback
                                    st.error(traceback.format_exc())
                            else:
                                st.error("GitHub features not available. Please load data first.")
                
                    advanced_gnn_info = st.expander("About Advanced GNN", expanded=True)
                    with advanced_gnn_info:
                        st.markdown("""
                        ### Advanced GNN with Transformer Architecture
                        
                        This model implements a sophisticated Graph Neural Network with:
                        
                        - **Multi-head attention** - Captures different relationship types simultaneously
                        - **Transformer layers** - Improved information flow across the graph
                        - **Explainable AI** - Provides importance and explainability scores
                        - **Residual connections** - Better gradient flow for deep networks
                        
                        The advanced model provides both importance scores and explainability metrics,
                        helping understand why certain repositories are considered important.
                        """)
                    
                    # Model parameters customization
                    with st.expander("Model Parameters", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            hidden_dim = st.slider("Hidden Dimension", min_value=32, max_value=256, value=128, step=32)
                            num_layers = st.slider("Number of Layers", min_value=1, max_value=5, value=3, step=1)
                        with col2:
                            n_heads = st.slider("Attention Heads", min_value=1, max_value=8, value=4, step=1)
                            dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
                    
                    run_advanced_gnn = st.button("Run Advanced GNN Analysis", key="run_adv_gnn_button")
                    
                    if run_advanced_gnn:
                        with st.spinner("Training Advanced Graph Neural Network..."):
                            if github_features:
                                try:
                                    # Run advanced GNN analysis
                                    st.info("Training Advanced GNN model with transformer architecture...")
                                    
                                    # Ensure the Graph is loaded
                                    if G is None:
                                        st.error("Graph is not properly loaded. Please ensure you've loaded dependency data.")
                                        st.stop()
                                        
                                    # Check graph nodes
                                    try:
                                        num_nodes = len(list(G.nodes()))
                                        if num_nodes == 0:
                                            st.error("Graph has no nodes. Please ensure dependency data is valid.")
                                            st.stop()
                                            
                                        st.info(f"Graph contains {num_nodes} nodes for analysis.")
                                    except Exception as e:
                                        st.error(f"Error checking graph: {str(e)}")
                                        st.stop()
                                        
                                    # Generate features if needed
                                    if not github_features:
                                        st.warning("No GitHub features available. Using generated sample features.")
                                        github_features = {}
                                        for node in G.nodes():
                                            # Create random but consistent features
                                            import hashlib
                                            hash_val = int(hashlib.md5(str(node).encode()).hexdigest(), 16) % 1000
                                                'stars': (hash_val % 100) / 100.0,
                                                'forks': (hash_val % 50) / 100.0,
                                                'activity': (hash_val % 75) / 100.0,
                                                'issues': (hash_val % 30) / 100.0,
                                                'contributors': (hash_val % 20) / 100.0
                                            }
                                    
                                    # Set model parameters from UI inputs
                                    model_params = {
                                        'hidden_dim': hidden_dim,
                                        'output_dim': hidden_dim // 2,  # Half the hidden dimension
                                        'num_layers': num_layers,
                                        'n_heads': n_heads,
                                        'dropout': dropout,
                                        'epochs': 300
                                    }
                                    
                                    # Calculate advanced GNN importance
                                    import time
                                    progress_bar = st.progress(0)
                                    # Show some progress before actual computation starts
                                    for i in range(10):
                                        progress_bar.progress(i/100)
                                        time.sleep(0.05)
                                    
                                    # Calculate advanced GNN scores
                                    advanced_results = advanced_gnn_node_importance(
                                        G, 
                                        github_features,
                                        reference_scores=pagerank_scores,  # Use supervised learning
                                        model_params=model_params
                                    )
                                    
                                    # Update progress
                                    for i in range(10, 101, 5):
                                        progress_bar.progress(i/100)
                                        time.sleep(0.05)
                                    
                                    # Store results in session state
                                    
                                    # Display advanced GNN results
                                    adv_gnn_df = pd.DataFrame({
                                    }).sort_values('importance_score', ascending=False).head(10)
                                    
                                    st.markdown("#### Advanced GNN Importance Scores (Top 10)")
                                    st.dataframe(adv_gnn_df, use_container_width=True)
                                    
                                    # Try to identify critical dependencies
                                    try:
                                        critical_deps = identify_critical_dependencies(
                                        )
                                        
                                        if critical_deps:
                                            st.markdown("#### Critical Dependencies Identified")
                                            if len(critical_deps) > 5:
                                                st.write(f"...and {len(critical_deps) - 5} more")
                                    except Exception as e:
                                        st.warning(f"Could not identify critical dependencies: {str(e)}")
                                    
                                    st.success("Advanced GNN analysis complete!")
                                    
                                except Exception as e:
                                    st.error(f"Error in Advanced GNN analysis: {str(e)}")
                                    import traceback
                                    st.error(traceback.format_exc())
                            else:
                                st.error("GitHub features not available. Please load data first.")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**PageRank Top 5**")
                                    st.dataframe(comparison_df.sort_values('pagerank_score', ascending=False).head(5))
                                with col2:
                                    st.markdown("**GNN Top 5**")
                                    st.dataframe(comparison_df.sort_values('gnn_score', ascending=False).head(5))
                                
                                # Create correlation matrix visualization
                                st.markdown("#### Score Correlation Matrix")
                                st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)
                                
                                # Store GNN results in session state
                                
                                # Set a flag to indicate GNN analysis is complete
                                
                                # Checkbox in Visualizations tab should be pre-selected
                                
                                st.success("GNN analysis complete! GNN prioritizes different projects than PageRank - check the comparison.")
                            except Exception as e:
                                st.error(f"Error running GNN analysis: {str(e)}")
                                st.info("Try running the main analysis first to ensure all required data is loaded properly.")
                        else:
                            st.warning("GitHub features are required for GNN analysis. Please enable 'Include GitHub Metrics'.")
            
            # Store results in session state for other tabs
    else:
        st.info("Upload data and click 'Run Analysis' to see model results.")

with tab3:
    st.markdown("<h2 class='section-header'>Visualizations</h2>", unsafe_allow_html=True)
    
    if 'results_df' in st.session_state and 'funding_allocation' in st.session_state:
        
        # Visualization options
        st.markdown("#### Select Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            show_importance = st.checkbox("Show Project Importance", value=True)
            show_funding = st.checkbox("Show Funding Allocation", value=True)
            show_gnn_results = st.checkbox("Show GNN Analysis Results", 
                                            value=st.session_state.get('show_gnn_results', False))
        with col2:
            show_comparison = st.checkbox("Show Comparison Chart", value=True)
            show_heatmap = st.checkbox("Show Importance Heatmap", value=True)
        
        # Project importance visualization
        if show_importance:
            st.markdown("#### Project Importance by Tier")
            fig = create_dependency_graph_visualization(
                G,
                node_size_map=final_scores,
                colorscale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Funding allocation chart
        if show_funding:
            st.markdown("#### Funding Allocation")
            top_n = st.slider("Show Top N Projects", min_value=5, max_value=50, value=20)
            fig = create_funding_allocation_chart(
                funding_allocation.head(top_n)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparison chart (PageRank vs. Final Score)
        if show_comparison:
            st.markdown("#### PageRank vs. Final Score Comparison")
            comparison_df = pd.DataFrame({
                'project': list(final_scores.keys()),
                'final_score': list(final_scores.values())
            }).sort_values('final_score', ascending=False).head(15)
            
            fig = create_comparison_chart(comparison_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Importance heatmap
        if show_heatmap:
            st.markdown("#### Project Importance Heatmap")
            top_n = st.slider("Number of Projects", min_value=10, max_value=100, value=50)
            fig = create_project_importance_heatmap(results_df.head(top_n))
            st.plotly_chart(fig, use_container_width=True)
            
        # GNN Results Visualization
        if show_gnn_results:
            st.markdown("### Graph Neural Network Analysis")
            
            # Create tabs for standard and advanced GNN visualizations
            
                if 'gnn_scores' in st.session_state:
                    st.markdown("#### Standard GNN Analysis Results")
                    
                    comparison_df = st.session_state.get('comparison_df')
                    
                    # Show standard GNN visualizations
                    st.markdown("##### Top 10 Projects by GNN Score")
                    gnn_top10 = pd.DataFrame({
                        'project': list(gnn_scores.keys()),
                        'gnn_score': list(gnn_scores.values())
                    }).sort_values('gnn_score', ascending=False).head(10)
                    
                    st.dataframe(gnn_top10, use_container_width=True)
                    
                    # Compare PageRank vs GNN rankings
                    if comparison_df is not None:
                        st.markdown("##### PageRank vs GNN Score Comparison")
                        fig = create_comparison_chart(comparison_df)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Run Standard GNN Analysis first to see visualizations.")
            
                if 'advanced_gnn_scores' in st.session_state:
                    st.markdown("#### Advanced GNN Analysis Results")
                    
                    explainability_scores = st.session_state.get('explainability_scores', {})
                    
                    # Show advanced GNN visualizations
                    st.markdown("##### Top 10 Projects by Advanced GNN Score")
                    adv_gnn_top10 = pd.DataFrame({
                        'project': list(advanced_scores.keys()),
                        'importance_score': list(advanced_scores.values()),
                    }).sort_values('importance_score', ascending=False).head(10)
                    
                    st.dataframe(adv_gnn_top10, use_container_width=True)
                    
                    # Display explainability visualization
                    st.markdown("##### Explainability Analysis")
                    st.markdown("""
                    This visualization shows which repositories have the most explainable importance scores.
                    Higher explainability means the model has more confidence in its assessment.
                    """)
                    
                    # Create explainability vs importance visualization
                    explainability_df = pd.DataFrame({
                        'project': list(advanced_scores.keys()),
                        'importance_score': list(advanced_scores.values()),
                    }).sort_values('explainability_score', ascending=False).head(15)
                    
                    # Create a scatter plot of importance vs explainability
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        mode='markers+text',
                        textposition="top center",
                        marker=dict(
                            size=10,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Importance Score")
                        ),
                        hovertemplate="<b>%{text}</b><br>Importance: %{y:.4f}<br>Explainability: %{x:.4f}"
                    ))
                    
                    fig.update_layout(
                        title="Repository Importance vs. Explainability",
                        xaxis_title="Explainability Score",
                        yaxis_title="Importance Score",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add relationship visualization if the graph is not too large
                    if G and len(G.nodes()) < 100:
                        st.markdown("##### Advanced GNN Relationship Analysis")
                        try:
                            # Creating visualization showing GNN-detected relationships
                            gnn_graph_fig = create_gnn_relationship_visualization(
                                G, 
                                advanced_scores,
                                st.session_state.get('pagerank_scores', {}),
                                colorscale='Plasma',
                                max_nodes=75
                            )
                            st.plotly_chart(gnn_graph_fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create GNN relationship visualization: {str(e)}")
                else:
                    st.info("Run Advanced GNN Analysis first to see visualizations.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### GNN Importance Scores (Top 10)")
                gnn_df = pd.DataFrame({
                    'project': list(gnn_scores.keys()),
                    'gnn_score': list(gnn_scores.values())
                }).sort_values('gnn_score', ascending=False).head(10)
                st.dataframe(gnn_df)
            
            with col2:
                st.markdown("##### GNN vs PageRank Priority Differences")
                if comparison_df is not None:
                    # Calculate absolute difference between normalized scores
                    )
                    
                    # Get projects with highest differences
                    diff_df = comparison_df.sort_values('score_diff', ascending=False).head(5)
                else:
                    st.info("Run GNN analysis in the Model & Analysis tab to see comparison.")
                    
            # Show GNN-weighted graph visualization
            st.markdown("##### Dependency Graph with GNN Importance")
            fig = create_dependency_graph_visualization(
                G,
                node_size_map=gnn_scores,
                colorscale='Plasma'  # Different colorscale to distinguish from PageRank
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Unsung Heroes Analysis Section
            st.markdown("#### 🌟 Unsung Heroes Analysis")
            st.markdown(f"""
            This analysis identifies repositories that are ranked much higher by our GNN than by traditional PageRank.
            These are potentially undervalued projects that contribute significantly to the {selected_blockchain.display_name} ecosystem
            but may not receive proportional recognition or funding.
            """)
            
            if st.button("Identify Unsung Heroes"):
                with st.spinner("Analyzing repository relationships to identify unsung heroes..."):
                    # Get github_features from session state
                    github_features = st.session_state.get('github_features')
                    
                    if github_features is not None:
                        # Run unsung heroes identification
                        unsung_heroes = identify_unsung_heroes(
                            G, 
                            github_features, 
                            pagerank_scores,
                            threshold_percentile=90
                        )
                        
                        if unsung_heroes:
                            st.success(f"Identified {len(unsung_heroes)} unsung hero repositories!")
                            
                            # Display unsung heroes in a table
                                {
                                }
                                for hero in unsung_heroes
                            ])
                            
                            st.dataframe(heroes_df, use_container_width=True)
                            
                            # Create specialized GNN relationship visualization with highlighted heroes
                            st.markdown("##### GNN Relationship Visualization (Highlighting Unsung Heroes)")
                            
                            fig = create_gnn_relationship_visualization(
                                G,
                                gnn_scores,
                                pagerank_scores,
                                unsung_heroes=unsung_heroes,
                                highlight_heroes=True,
                                colorscale='Plasma',
                                max_nodes=100
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation about why these are important
                            st.markdown("""
                            #### Why Unsung Heroes Matter
                            
                            These repositories might be critical to the ecosystem in ways that traditional metrics miss:
                            
                            * They may provide **foundational infrastructure** that many other projects build upon
                            * They might serve as **connectors** or **bridges** between different parts of the ecosystem
                            * They could implement **critical security features** that don't attract attention but are essential
                            * They may be **newer innovations** that haven't yet accumulated traditional reputation metrics
                            
                            Funding these projects could provide outsized impact for the ecosystem.
                            """)
                        else:
                            st.info("No significant unsung heroes identified in this dataset.")
                            
                        # Store unsung heroes in session state
                    else:
                        st.error("GitHub features are required for unsung heroes analysis. Please run the full analysis with GitHub metrics enabled.")
            
            st.markdown("""
            **Note on GNN vs PageRank**: 
            
            Graph Neural Networks can identify important projects in the ecosystem that PageRank might miss.
            While PageRank primarily considers link structure, GNNs learn from:
            
            1. Repository graph position
            2. GitHub metrics and popularity
            3. Complex patterns in network connectivity
            
            This often leads to discovering "unsung heroes" - projects with critical importance but low visibility.
            """)
        elif show_gnn_results:
            # Check if GNN was run but results not properly stored
            if st.session_state.get('gnn_analysis_complete', False):
                st.success("GNN analysis completed! Please go to Model & Analysis tab, then rerun the GNN analysis to update the results.")
            else:
                st.info("Run GNN analysis in the Model & Analysis tab first.")
    else:
        st.info("Run the analysis in the 'Model & Analysis' tab to generate visualizations.")

with tab4:
    st.markdown("<h2 class='section-header'>Export & Documentation</h2>", unsafe_allow_html=True)
    
    # Documentation
    st.markdown("#### Methodology Documentation")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ### Methodology Overview
    
    The funding allocation tool implements the Paradoxical OS Innovation Strategy outlined in the whitepaper, addressing three key paradoxes:
    
    1. **The Unsung Hero Paradox**: Using a weighted contribution approach to ensure deep dependencies receive fair funding.
    2. **The Human vs. Machine Judgment Paradox**: Implementing a hybrid model design that balances AI predictions with human reasoning.
    3. **The Open vs. Proprietary Paradox**: Creating a strategic transparency framework that balances collaboration with competitive advantage.
    
    ### Technical Implementation
    
    #### 1. Graph Processing
    - Dependency graph processing using NetworkX
    - PageRank implementation for baseline importance scoring
    - Tiered weighting to credit projects at different dependency depths
    
    #### 2. Feature Extraction
    - GitHub repository metrics extraction (stars, forks, commits)
    - Normalization and feature engineering
    
    #### 3. Model Development
    - Initial importance scoring using PageRank algorithm
    - Enhanced scoring with weighted contribution approach
    - Advanced Graph Neural Network (GNN) analysis for deep structural learning
    - Ranking model using machine learning (XGBoost/Linear/Random Forest)
    
    #### 4. Validation Strategy
    - Cross-validation for model performance
    - Consistency checks across graph levels
    - Performance optimization for large-scale graphs
    
    ### Submission Format
    The tool outputs a CSV file with funding allocations for each project, ready for Cryptopond submission.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Export options
    st.markdown("#### Export Results")
    
    if 'results_df' in st.session_state and 'funding_allocation' in st.session_state:
        
        if st.button("Export Results", use_container_width=True):
            if export_format == "CSV":
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{selected_blockchain_id}_funding_allocation.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            elif export_format == "JSON":
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{selected_blockchain_id}_funding_allocation.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:  # Excel
                output = io.BytesIO()
                excel_data = output.getvalue()
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"{selected_blockchain_id}_funding_allocation.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
    else:
        st.info("Run the analysis in the 'Model & Analysis' tab to export results.")
    
    # Validation metrics
    st.markdown("#### Validation & Performance Metrics")
    if 'results_df' in st.session_state:
        st.markdown("<div class='note'>", unsafe_allow_html=True)
        st.markdown("""
        **Performance Summary**
        
        - Graph Processing Speed: Optimized for 15,000+ nodes
        - Model Accuracy: Cross-validated metrics shown in Model & Analysis tab
        - Memory Usage: Efficient sparse matrix implementations for large graphs
        
        **Data Collection Methods**
        
        - **Simulated Data**: Generated sample data for testing without external dependencies
        - **GitHub API**: Direct API access with higher accuracy (requires token)
        - **Web Scraping**: HTML parsing of GitHub pages for token-free data collection
        
        **Next Steps for Production Use**
        
        1. Enhanced GitHub data collection with combined API and scraping approaches
        2. Implementation of user feedback loop for model refinement
        3. Expanded dependency discovery through package manifest analysis
        4. GNN model optimization with additional training data from ecosystem feedback
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Run the analysis to see validation metrics.")

# Footer
st.markdown("---")
st.markdown(f"Crypto_ParadoxOS | {selected_blockchain.display_name} | Based on Paradoxical OS Innovation Strategy | v1.0")

