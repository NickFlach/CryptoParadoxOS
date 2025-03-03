"""
Web scraper module for retrieving content from websites.
This module provides utilities for fetching and parsing HTML content from web pages.
"""

import logging
import requests
import trafilatura
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup
import json
import re
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_website_text_content(url: str) -> str:
    """
    Extract the main text content from a website.
    
    Args:
        url: URL of the website to scrape
        
    Returns:
        Extracted text content
    """
    try:
        logger.info(f"Fetching content from {url}")
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        
        if not text:
            logger.warning(f"No text extracted from {url}")
            return "No content extracted"
            
        return text
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        return f"Error: {str(e)}"

def scrape_github_repository(repo_name: str) -> Dict[str, Any]:
    """
    Scrape GitHub repository information using Beautiful Soup.
    
    Args:
        repo_name: Repository name in 'owner/repo' format
        
    Returns:
        Dictionary of repository information
    """
    url = f"https://github.com/{repo_name}"
    logger.info(f"Scraping GitHub repository: {url}")
    
    try:
        # Add a random delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic repo information
        result = {
            'repo_name': repo_name,
            'url': url,
            'description': None,
            'stars': 0,
            'forks': 0,
            'watchers': 0,
            'issues': 0,
            'last_updated': None,
            'languages': {},
            'topics': []
        }
        
        # Description
        description_elem = soup.select_one('p[itemprop="description"]')
        if description_elem:
            result['description'] = description_elem.text.strip()
        
        # Stars, forks, watchers
        for counter in soup.select('a.social-count'):
            href = counter.get('href', '')
            count = int(re.sub(r'[^\d]', '', counter.text.strip()) or 0)
            
            if '/stargazers' in href:
                result['stars'] = count
            elif '/network/members' in href:
                result['forks'] = count
            elif '/watchers' in href:
                result['watchers'] = count
        
        # Issues
        issues_elem = soup.select_one('a[href$="/issues"] .Counter')
        if issues_elem:
            result['issues'] = int(re.sub(r'[^\d]', '', issues_elem.text.strip()) or 0)
        
        # Last updated
        time_elem = soup.select_one('relative-time')
        if time_elem:
            result['last_updated'] = time_elem.get('datetime')
        
        # Languages
        language_elems = soup.select('.repository-lang-stats-numbers .lang')
        for lang_elem in language_elems:
            lang_name = lang_elem.text.strip()
            lang_percent_elem = lang_elem.find_next('span', class_='percent')
            if lang_percent_elem:
                percent = float(lang_percent_elem.text.strip().replace('%', ''))
                result['languages'][lang_name] = percent
        
        # Topics
        topic_elems = soup.select('a.topic-tag')
        for topic_elem in topic_elems:
            result['topics'].append(topic_elem.text.strip())
        
        logger.info(f"Successfully scraped repository: {repo_name}")
        return result
    
    except Exception as e:
        logger.error(f"Error scraping GitHub repository {repo_name}: {str(e)}")
        return {
            'repo_name': repo_name,
            'url': url,
            'error': str(e)
        }

def scrape_github_repositories_batch(repo_names: List[str], max_repos: int = 20) -> Dict[str, Dict[str, Any]]:
    """
    Scrape multiple GitHub repositories.
    
    Args:
        repo_names: List of repository names in 'owner/repo' format
        max_repos: Maximum number of repositories to scrape
        
    Returns:
        Dictionary mapping repository names to their scraped information
    """
    logger.info(f"Scraping batch of {len(repo_names)} repositories (max: {max_repos})")
    
    # Limit the number of repos to scrape
    if len(repo_names) > max_repos:
        logger.warning(f"Limiting batch to {max_repos} repositories (from {len(repo_names)})")
        repo_names = repo_names[:max_repos]
    
    results = {}
    for i, repo_name in enumerate(repo_names):
        logger.info(f"Scraping repository {i+1}/{len(repo_names)}: {repo_name}")
        results[repo_name] = scrape_github_repository(repo_name)
        
        # Add a random delay between requests
        if i < len(repo_names) - 1:
            delay = random.uniform(2, 5)
            logger.info(f"Waiting {delay:.2f} seconds before next request...")
            time.sleep(delay)
    
    logger.info(f"Completed batch scraping of {len(results)} repositories")
    return results

def extract_repository_metrics(scrape_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Extract numerical metrics from scraped repository data.
    
    Args:
        scrape_results: Dictionary of scraped repository information
        
    Returns:
        Dictionary mapping repository names to dictionaries of metrics
    """
    metrics = {}
    
    for repo_name, repo_data in scrape_results.items():
        if 'error' in repo_data:
            continue
            
        repo_metrics = {
            'stars': float(repo_data.get('stars', 0)),
            'forks': float(repo_data.get('forks', 0)),
            'issues': float(repo_data.get('issues', 0)),
            'watchers': float(repo_data.get('watchers', 0)),
            'language_count': float(len(repo_data.get('languages', {}))),
            'topic_count': float(len(repo_data.get('topics', []))),
        }
        
        # Calculate additional derived metrics
        if repo_metrics['stars'] > 0:
            repo_metrics['fork_to_star_ratio'] = repo_metrics['forks'] / repo_metrics['stars']
        else:
            repo_metrics['fork_to_star_ratio'] = 0.0
            
        metrics[repo_name] = repo_metrics
    
    return metrics

def scrape_blockchain_project_data(blockchain_name: str, project_list: List[str]) -> Dict[str, Any]:
    """
    Scrape data for blockchain projects from GitHub and other sources.
    
    Args:
        blockchain_name: Name of the blockchain ecosystem
        project_list: List of project repository names
        
    Returns:
        Dictionary with scraped project data
    """
    logger.info(f"Scraping data for {blockchain_name} ecosystem - {len(project_list)} projects")
    
    # Scrape GitHub repositories
    github_data = scrape_github_repositories_batch(project_list)
    
    # Extract metrics
    metrics = extract_repository_metrics(github_data)
    
    # Scrape blockchain-specific information
    blockchain_info = {
        'name': blockchain_name,
        'documentation_scraped': None
    }
    
    # Try to scrape blockchain documentation based on blockchain name
    try:
        if blockchain_name.lower() == 'ethereum':
            doc_url = 'https://ethereum.org/en/developers/docs/'
            blockchain_info['documentation_scraped'] = get_website_text_content(doc_url)
        elif blockchain_name.lower() == 'solana':
            doc_url = 'https://docs.solana.com/'
            blockchain_info['documentation_scraped'] = get_website_text_content(doc_url)
        elif blockchain_name.lower() == 'polkadot':
            doc_url = 'https://wiki.polkadot.network/'
            blockchain_info['documentation_scraped'] = get_website_text_content(doc_url)
    except Exception as e:
        logger.error(f"Error scraping blockchain documentation: {str(e)}")
    
    return {
        'blockchain_info': blockchain_info,
        'github_data': github_data,
        'metrics': metrics
    }

def normalize_scraper_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Normalize metrics from web scraping to [0,1] range for ML models.
    
    Args:
        metrics: Dictionary mapping repositories to metrics
        
    Returns:
        Dictionary of normalized metrics
    """
    if not metrics:
        return {}
        
    # Collect all values for each metric type
    metric_values = {}
    all_metrics = set()
    
    for repo_metrics in metrics.values():
        for metric_name, value in repo_metrics.items():
            if metric_name not in metric_values:
                metric_values[metric_name] = []
            metric_values[metric_name].append(value)
            all_metrics.add(metric_name)
    
    # Calculate min/max for each metric
    metric_ranges = {}
    for metric_name in all_metrics:
        values = metric_values.get(metric_name, [0])
        metric_ranges[metric_name] = {
            'min': min(values),
            'max': max(values)
        }
    
    # Normalize each repository's metrics
    normalized = {}
    for repo_name, repo_metrics in metrics.items():
        normalized_metrics = {}
        
        for metric_name, value in repo_metrics.items():
            min_val = metric_ranges[metric_name]['min']
            max_val = metric_ranges[metric_name]['max']
            
            # Avoid division by zero
            if max_val == min_val:
                normalized_metrics[metric_name] = 0.0 if value == 0 else 1.0
            else:
                normalized_metrics[metric_name] = (value - min_val) / (max_val - min_val)
        
        normalized[repo_name] = normalized_metrics
    
    return normalized