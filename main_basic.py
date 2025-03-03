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

# Import non-PyTorch components
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
    create_comparison_chart
)
from utils import (
    export_results_to_csv,
    compute_validation_metrics
)

# Setup page styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .blockchain-card {
        background-color: #F5F8FF;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #4A90E2;
        display: block;
    }
    .blockchain-card h4 {
        color: #4A90E2;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1.2rem;
    }
    .blockchain-card p {
        margin-bottom: 15px;
        color: #333;
    }
    .blockchain-card ul {
        margin-bottom: 0;
        padding-left: 20px;
    }
    .blockchain-card li {
        margin-bottom: 5px;
        color: #666;
    }
    .metric-container {
        background-color: #EFF6FF;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .footer-text {
        text-align: center;
        font-size: 0.8rem;
        color: #666;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and header
st.markdown("<h1 class='main-header'>Crypto_ParadoxOS</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p>A blockchain-agnostic funding allocation system powered by advanced graph analysis and machine learning</p>
</div>
""", unsafe_allow_html=True)

# Main app tabs
tab1, tab2, tab3 = st.tabs(["Blockchain Selection", "Model & Analysis", "Visualizations"])

# Tab 1: Blockchain Selection
with tab1:
    st.header("Select Blockchain Ecosystem")
    
    # Display blockchain options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ethereum Ecosystem")
        st.markdown("""
        <div class="blockchain-card">
            <h4>Ethereum</h4>
            <p>The leading smart contract platform and DeFi ecosystem</p>
            <ul>
                <li>Root repository: ethereum/go-ethereum</li>
                <li>Language: Solidity & Go</li>
                <li>Year founded: 2015</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Ethereum", key="select_ethereum"):
            st.session_state.selected_blockchain = "ethereum"
            st.success("Ethereum ecosystem selected!")
            
    with col2:
        st.subheader("Solana Ecosystem")
        st.markdown("""
        <div class="blockchain-card">
            <h4>Solana</h4>
            <p>High-performance blockchain with low transaction costs</p>
            <ul>
                <li>Root repository: solana-labs/solana</li>
                <li>Language: Rust</li>
                <li>Year founded: 2020</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Solana", key="select_solana"):
            st.session_state.selected_blockchain = "solana"
            st.success("Solana ecosystem selected!")
    
    # Additional blockchains
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Polkadot Ecosystem")
        st.markdown("""
        <div class="blockchain-card">
            <h4>Polkadot</h4>
            <p>Multi-chain network enabling cross-blockchain transfers</p>
            <ul>
                <li>Root repository: paritytech/polkadot</li>
                <li>Language: Rust</li>
                <li>Year founded: 2017</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Polkadot", key="select_polkadot"):
            st.session_state.selected_blockchain = "polkadot"
            st.success("Polkadot ecosystem selected!")
            
    with col4:
        st.subheader("Cosmos Ecosystem")
        st.markdown("""
        <div class="blockchain-card">
            <h4>Cosmos</h4>
            <p>Internet of Blockchains focusing on interoperability</p>
            <ul>
                <li>Root repository: cosmos/cosmos-sdk</li>
                <li>Language: Go</li>
                <li>Year founded: 2016</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Cosmos", key="select_cosmos"):
            st.session_state.selected_blockchain = "cosmos"
            st.success("Cosmos ecosystem selected!")
    
    # Data source selection
    st.header("Data Source Configuration")
    data_source = st.radio(
        "Select data source:",
        ["Simulated Data", "GitHub API (Token Required)", "Web Scraping"],
        index=0
    )
    
    if data_source == "GitHub API (Token Required)":
        github_token = st.text_input("GitHub API Token:", type="password")
        if github_token:
            st.success("GitHub API token provided.")
    
    st.subheader("Advanced Settings")
    with st.expander("Configure Analysis Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Graph analysis parameters
            st.write("Graph Analysis Settings")
            max_depth = st.slider("Max Dependency Depth", min_value=1, max_value=5, value=2)
            pagerank_alpha = st.slider("PageRank Alpha", min_value=0.1, max_value=0.99, value=0.85, step=0.01)
            
        with col2:
            # Funding allocation parameters
            st.write("Funding Allocation Settings")
            github_weight = st.slider("GitHub Metrics Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            funding_amount = st.number_input("Total Funding Amount (ETH)", min_value=1, max_value=1000000, value=1000)

# Tab 2: Model & Analysis
with tab2:
    st.header("Model & Analysis")
    
    # Check if blockchain is selected
    if 'selected_blockchain' not in st.session_state:
        st.warning("Please select a blockchain ecosystem in the first tab.")
    else:
        blockchain_id = st.session_state.selected_blockchain
        
        # Run analysis button
        if st.button("Run Analysis", key="run_analysis"):
            with st.spinner(f"Analyzing {blockchain_id.capitalize()} ecosystem..."):
                # Simulate processing time with a placeholder
                progress_bar = st.progress(0)
                
                # Generate or load sample data
                st.info("Generating sample data...")
                try:
                    data_path = ensure_blockchain_sample_data_exists(blockchain_id)
                    progress_bar.progress(25)
                    
                    # Load the dependency graph
                    st.info("Building dependency graph...")
                    df = pd.read_csv(data_path)
                    G = load_dependency_graph(df)
                    progress_bar.progress(50)
                    
                    # Calculate PageRank and metrics
                    st.info("Calculating repository importance...")
                    pagerank_scores = calculate_pagerank(G, alpha=pagerank_alpha)
                    node_names = list(G.nodes())
                    github_metrics = extract_github_metrics(node_names)
                    github_metrics = normalize_github_features(github_metrics)
                    progress_bar.progress(75)
                    
                    # Calculate weighted contributions
                    weighted_scores = calculate_weighted_contribution(
                        G, 
                        pagerank_scores, 
                        github_metrics, 
                        contribution_weight=1.0 - github_weight
                    )
                    
                    # Apply tiered weighting
                    final_scores = apply_tiered_weighting(G, weighted_scores)
                    
                    # Store results in session state
                    st.session_state.dependency_graph = G
                    st.session_state.pagerank_scores = pagerank_scores
                    st.session_state.github_metrics = github_metrics
                    st.session_state.final_scores = final_scores
                    st.session_state.funding_amount = funding_amount
                    
                    progress_bar.progress(100)
                    st.success("Analysis complete!")
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
        
        # Results section - show if analysis has been run
        if 'final_scores' in st.session_state:
            st.subheader("Analysis Results")
            
            # Convert scores to DataFrame for display
            scores_df = pd.DataFrame({
                'Repository': list(st.session_state.final_scores.keys()),
                'Importance Score': list(st.session_state.final_scores.values())
            })
            scores_df = scores_df.sort_values('Importance Score', ascending=False).reset_index(drop=True)
            
            # Calculate funding allocations
            total_funding = st.session_state.funding_amount
            scores_df['Allocation (ETH)'] = scores_df['Importance Score'] / scores_df['Importance Score'].sum() * total_funding
            scores_df['Allocation (%)'] = scores_df['Importance Score'] / scores_df['Importance Score'].sum() * 100
            
            # Display top 10 repositories
            st.write("Top 10 Repositories by Importance:")
            st.dataframe(scores_df.head(10).style.format({
                'Importance Score': '{:.4f}',
                'Allocation (ETH)': '{:.2f}',
                'Allocation (%)': '{:.2f}%'
            }))
            
            # Download results button
            csv_data = export_results_to_csv(scores_df)
            st.download_button(
                label="Download Full Results",
                data=csv_data,
                file_name=f"{blockchain_id}_funding_allocation.csv",
                mime="text/csv"
            )
            
            # Critical dependencies
            st.subheader("Critical Dependencies")
            critical_deps = identify_critical_dependencies(
                st.session_state.dependency_graph, 
                st.session_state.final_scores,
                threshold=0.9
            )
            
            if critical_deps:
                st.write("The following repositories are identified as critical dependencies in the ecosystem:")
                for dep in critical_deps:
                    st.markdown(f"- **{dep}** (Score: {st.session_state.final_scores.get(dep, 0):.4f})")
            else:
                st.write("No critical dependencies identified.")

# Tab 3: Visualizations
with tab3:
    st.header("Visualizations")
    
    # Check if analysis has been run
    if 'final_scores' not in st.session_state:
        st.warning("Please run the analysis in the Model & Analysis tab first.")
    else:
        # Prepare data for visualizations
        G = st.session_state.dependency_graph
        final_scores = st.session_state.final_scores
        pagerank_scores = st.session_state.pagerank_scores
        
        # Network Graph Visualization
        st.subheader("Dependency Network Graph")
        st.write("The ecosystem dependency network showing connections between projects")
        
        try:
            network_fig = create_dependency_graph_visualization(G, node_size_map=final_scores)
            st.plotly_chart(network_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating network visualization: {str(e)}")
        
        col1, col2 = st.columns(2)
        
        # Funding Allocation Chart
        with col1:
            st.subheader("Funding Allocation")
            
            # Create DataFrame for allocation
            allocation_df = pd.DataFrame({
                'Repository': list(final_scores.keys()),
                'Score': list(final_scores.values())
            })
            allocation_df = allocation_df.sort_values('Score', ascending=False).head(10)
            allocation_df['Allocation'] = allocation_df['Score'] / allocation_df['Score'].sum() * 100
            
            try:
                allocation_chart = create_funding_allocation_chart(allocation_df)
                st.plotly_chart(allocation_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating allocation chart: {str(e)}")
        
        # Comparison Chart
        with col2:
            st.subheader("Scoring Model Comparison")
            
            # Create comparison DataFrame
            common_repos = set(pagerank_scores.keys()) & set(final_scores.keys())
            comparison_df = pd.DataFrame({
                'Repository': list(common_repos),
                'PageRank': [pagerank_scores.get(repo, 0) for repo in common_repos],
                'Final Score': [final_scores.get(repo, 0) for repo in common_repos]
            })
            comparison_df = comparison_df.sort_values('Final Score', ascending=False).head(10)
            
            try:
                comparison_chart = create_comparison_chart(comparison_df)
                st.plotly_chart(comparison_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating comparison chart: {str(e)}")
        
        # Heatmap of Project Importance
        st.subheader("Project Importance Heatmap")
        
        # Create DataFrame for heatmap
        top_repos = pd.DataFrame({
            'Repository': list(final_scores.keys()),
            'Score': list(final_scores.values())
        }).sort_values('Score', ascending=False).head(20)
        
        # Create metrics matrix
        if st.session_state.github_metrics:
            # First, create a better structure for the heatmap
            # We will use Repository column and Score column from top_repos
            metrics_df = pd.DataFrame()
            metrics_df['Repository'] = top_repos['Repository']
            metrics_df['Importance Score'] = top_repos['Score']
            
            # Add the GitHub metrics as additional columns
            for repo in metrics_df['Repository']:
                if repo in st.session_state.github_metrics:
                    for metric, value in st.session_state.github_metrics[repo].items():
                        if metric not in metrics_df.columns:
                            metrics_df[metric] = 0
                        metrics_df.loc[metrics_df['Repository'] == repo, metric] = value
            
            # Reset index to make sure the Repository is a column, not the index
            metrics_df = metrics_df.reset_index(drop=True)
            
            # Fill NaNs with 0
            metrics_df = metrics_df.fillna(0)
            
            if not metrics_df.empty:
                try:
                    heatmap_fig = create_project_importance_heatmap(metrics_df)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating heatmap: {str(e)}")
            else:
                st.warning("Insufficient data for heatmap visualization.")
        else:
            st.warning("GitHub metrics not available for heatmap visualization.")

# Footer
st.markdown("""
<div class="footer-text">
    <p>Crypto_ParadoxOS - Advanced funding allocation for blockchain ecosystems</p>
    <p>Copyright © 2025 - All rights reserved</p>
</div>
""", unsafe_allow_html=True)