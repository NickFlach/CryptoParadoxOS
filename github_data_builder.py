import os
import requests
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
from urllib.parse import urlparse, quote
import re
from bs4 import BeautifulSoup
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubDataBuilder:
    """
    Class for fetching and building GitHub data for repositories.
    This handles rate limiting, caching, and data transformation.
    """
    
    def __init__(self, token: Optional[str] = None, cache_dir: str = "data/github_cache"):
        """
        Initialize the GitHub data builder.
        
        Args:
            token: GitHub API token (optional but recommended to avoid rate limits)
            cache_dir: Directory to cache GitHub API responses
        """
        self.token = token
        self.cache_dir = cache_dir
        self.headers = {}
        if token:
            self.headers["Authorization"] = f"token {token}"
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Rate limiting tracking
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = 0
    
    def _get_api_url(self, resource_path: str) -> str:
        """Get the full GitHub API URL for a resource path."""
        return f"https://api.github.com/{resource_path}"
    
    def _get_cache_path(self, resource_path: str) -> str:
        """Get the cache file path for a resource path."""
        # Create a valid filename from the resource path
        filename = re.sub(r'[^\w\-_]', '_', resource_path) + '.json'
        return os.path.join(self.cache_dir, filename)
    
    def _check_rate_limit(self):
        """Check and handle GitHub API rate limiting."""
        if self.rate_limit_remaining <= 10:
            current_time = time.time()
            if current_time < self.rate_limit_reset:
                wait_time = self.rate_limit_reset - current_time + 1
                logger.warning(f"Rate limit nearly exhausted. Waiting for {wait_time:.1f} seconds.")
                time.sleep(wait_time)
    
    def _update_rate_limit(self, response):
        """Update rate limit information from response headers."""
        if 'X-RateLimit-Remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in response.headers:
            self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
        
        if self.rate_limit_remaining <= 100:
            logger.warning(f"GitHub API rate limit running low: {self.rate_limit_remaining} requests remaining")
    
    def api_request(self, resource_path: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Make a GitHub API request with rate limiting and caching.
        
        Args:
            resource_path: API resource path (e.g., "repos/ethereum/go-ethereum")
            use_cache: Whether to use cached responses
            
        Returns:
            JSON response as a dictionary
        """
        cache_path = self._get_cache_path(resource_path)
        
        # Check cache first if enabled
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    logger.debug(f"Using cached data for {resource_path}")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading cache for {resource_path}: {e}")
        
        # Check rate limit before making request
        self._check_rate_limit()
        
        # Make API request
        url = self._get_api_url(resource_path)
        logger.info(f"Fetching data from GitHub API: {url}")
        
        try:
            response = requests.get(url, headers=self.headers)
            self._update_rate_limit(response)
            
            if response.status_code == 200:
                data = response.json()
                
                # Cache response
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
                
                return data
            elif response.status_code == 404:
                logger.warning(f"Resource not found: {resource_path}")
                return {}
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {}
        
        except Exception as e:
            logger.error(f"Error making API request to {url}: {e}")
            return {}
    
    def normalize_repo_name(self, repo_identifier: str) -> str:
        """
        Normalize a repository identifier to owner/repo format.
        
        Args:
            repo_identifier: Repository identifier (URL, path, or owner/repo)
            
        Returns:
            Normalized repository name in 'owner/repo' format
        """
        # If it's a URL, extract the path
        if repo_identifier.startswith(('http://', 'https://')):
            parsed_url = urlparse(repo_identifier)
            path = parsed_url.path.lstrip('/')
            # Remove .git extension if present
            if path.endswith('.git'):
                path = path[:-4]
            
            # Handle github.com URLs
            if parsed_url.netloc == 'github.com':
                parts = path.split('/')
                if len(parts) >= 2:
                    return f"{parts[0]}/{parts[1]}"
        
        # If it's already in owner/repo format
        elif '/' in repo_identifier and repo_identifier.count('/') == 1:
            return repo_identifier
        
        # Default fallback - assume it's the repo name with ethereum as owner
        return f"ethereum/{repo_identifier}"
    
    def get_repo_info(self, repo_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get repository information.
        
        Args:
            repo_name: Repository name in 'owner/repo' format
            use_cache: Whether to use cached responses
            
        Returns:
            Repository information dictionary
        """
        normalized_name = self.normalize_repo_name(repo_name)
        resource_path = f"repos/{normalized_name}"
        return self.api_request(resource_path, use_cache)
    
    def get_repo_stats(self, repo_name: str, stat_type: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get repository statistics.
        
        Args:
            repo_name: Repository name in 'owner/repo' format
            stat_type: Type of statistics ('contributors', 'commit_activity', etc.)
            use_cache: Whether to use cached responses
            
        Returns:
            Statistics as a list of dictionaries
        """
        normalized_name = self.normalize_repo_name(repo_name)
        resource_path = f"repos/{normalized_name}/stats/{stat_type}"
        result = self.api_request(resource_path, use_cache)
        
        # Ensure we return a list
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and result:
            return [result]
        else:
            return []
    
    def get_contributors(self, repo_name: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get repository contributors.
        
        Args:
            repo_name: Repository name in 'owner/repo' format
            use_cache: Whether to use cached responses
            
        Returns:
            List of contributors
        """
        normalized_name = self.normalize_repo_name(repo_name)
        resource_path = f"repos/{normalized_name}/contributors"
        result = self.api_request(resource_path, use_cache)
        
        # Ensure we return a list
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and result:
            return [result]
        else:
            return []
    
    def get_issues(self, repo_name: str, state: str = "all", use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get repository issues.
        
        Args:
            repo_name: Repository name in 'owner/repo' format
            state: Issue state ('open', 'closed', 'all')
            use_cache: Whether to use cached responses
            
        Returns:
            List of issues
        """
        normalized_name = self.normalize_repo_name(repo_name)
        resource_path = f"repos/{normalized_name}/issues?state={state}&per_page=100"
        result = self.api_request(resource_path, use_cache)
        
        # Ensure we return a list
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and result:
            return [result]
        else:
            return []
    
    def get_repository_dependencies(self, repo_name: str, use_cache: bool = True) -> List[str]:
        """
        Get dependencies for a GitHub repository.
        
        Args:
            repo_name: Repository name in 'owner/repo' format
            use_cache: Whether to use cached responses
            
        Returns:
            List of dependency repository names
        """
        normalized_name = self.normalize_repo_name(repo_name)
        
        # Check for package.json for Node.js projects
        try:
            # Try to get package.json
            pkg_json_path = f"repos/{normalized_name}/contents/package.json"
            pkg_json_data = self.api_request(pkg_json_path, use_cache)
            
            if 'content' in pkg_json_data:
                import base64
                content = base64.b64decode(pkg_json_data['content']).decode('utf-8')
                pkg_data = json.loads(content)
                
                dependencies = {}
                if 'dependencies' in pkg_data:
                    dependencies.update(pkg_data['dependencies'])
                if 'devDependencies' in pkg_data:
                    dependencies.update(pkg_data['devDependencies'])
                
                # Extract GitHub repositories from dependencies
                # This is a simplified approach - many packages aren't direct GitHub repos
                github_deps = []
                for pkg_name, version in dependencies.items():
                    if 'github' in pkg_name or (isinstance(version, str) and 'github' in version):
                        github_deps.append(pkg_name)
                
                return github_deps
        except Exception as e:
            logger.debug(f"Error parsing package.json for {repo_name}: {e}")
        
        # Check for requirements.txt for Python projects
        try:
            req_txt_path = f"repos/{normalized_name}/contents/requirements.txt"
            req_txt_data = self.api_request(req_txt_path, use_cache)
            
            if 'content' in req_txt_data:
                import base64
                content = base64.b64decode(req_txt_data['content']).decode('utf-8')
                # Parse requirements.txt - very simplified
                dependencies = [line.strip() for line in content.split('\n') 
                               if line.strip() and not line.startswith('#')]
                return dependencies
        except Exception as e:
            logger.debug(f"Error parsing requirements.txt for {repo_name}: {e}")
        
        # As a fallback, return an empty list
        return []
    
    def extract_repo_metrics(self, repo_name: str, use_cache: bool = True) -> Dict[str, float]:
        """
        Extract metrics for a GitHub repository.
        
        Args:
            repo_name: Repository name in 'owner/repo' format
            use_cache: Whether to use cached responses
            
        Returns:
            Dictionary of repository metrics
        """
        # Get basic repository information
        repo_info = self.get_repo_info(repo_name, use_cache)
        
        if not repo_info:
            logger.warning(f"No data found for repository: {repo_name}")
            return {}
        
        # Basic metrics
        metrics = {
            "stars": repo_info.get("stargazers_count", 0),
            "forks": repo_info.get("forks_count", 0),
            "open_issues": repo_info.get("open_issues_count", 0),
            "watchers": repo_info.get("watchers_count", 0),
            "size": repo_info.get("size", 0),
        }
        
        # Handle timestamps with timezone awareness
        current_time = pd.Timestamp.now()
        
        # Calculate age in days
        if repo_info.get("created_at"):
            created_time = pd.Timestamp(repo_info.get("created_at")).tz_localize(None)  # Remove timezone
            metrics["age_days"] = (current_time - created_time).days
        else:
            metrics["age_days"] = 365  # Default to 1 year if no creation date
            
        # Calculate days since last update
        if repo_info.get("updated_at"):
            updated_time = pd.Timestamp(repo_info.get("updated_at")).tz_localize(None)  # Remove timezone
            metrics["last_updated_days"] = (current_time - updated_time).days
        else:
            metrics["last_updated_days"] = 30  # Default to 1 month if no update date
        
        # Get commit stats (a bit more advanced)
        commit_activity = self.get_repo_stats(repo_name, "commit_activity", use_cache)
        if commit_activity and isinstance(commit_activity, list):
            total_commits = sum(week.get("total", 0) for week in commit_activity)
            metrics["commits"] = total_commits
            metrics["commit_frequency"] = total_commits / max(1, metrics["age_days"] / 7)  # per week
        
        # Get contributor stats
        contributors = self.get_contributors(repo_name, use_cache)
        if contributors and isinstance(contributors, list):
            metrics["contributors"] = len(contributors)
            
            # Get total contributions
            total_contributions = sum(contributor.get("contributions", 0) for contributor in contributors)
            metrics["total_contributions"] = total_contributions
            
            # Contributors diversity (Gini coefficient-like measure)
            if contributors and len(contributors) > 1:
                contributions = [c.get("contributions", 0) for c in contributors]
                total = sum(contributions)
                if total > 0:
                    # Sort contributions
                    contributions.sort()
                    # Calculate Gini coefficient
                    n = len(contributions)
                    indices = range(1, n + 1)
                    gini = sum([(2 * i - n - 1) * c for i, c in zip(indices, contributions)]) / (n * total)
                    # Invert so higher is more diverse
                    metrics["contributor_diversity"] = 1 - abs(gini)
                else:
                    metrics["contributor_diversity"] = 0
            else:
                metrics["contributor_diversity"] = 0
        
        # Get issue stats
        issues = self.get_issues(repo_name, "all", use_cache)
        if issues and isinstance(issues, list):
            closed_issues = sum(1 for issue in issues if issue.get("state") == "closed")
            total_issues = len(issues)
            
            if total_issues > 0:
                metrics["issue_closing_rate"] = closed_issues / total_issues
            else:
                metrics["issue_closing_rate"] = 0
        
        # Calculate derived metrics
        if "contributors" in metrics and metrics["contributors"] > 0:
            metrics["contributor_engagement"] = metrics.get("commits", 0) / (metrics["contributors"] * max(1, metrics["age_days"]/30)) * 100
        
        # Activity score (combining multiple factors)
        activity_factors = [
            metrics.get("commit_frequency", 0) / 10,  # scale down commit frequency
            (1 / (metrics.get("last_updated_days", 365) + 1)) * 100,  # inverse of days since update
            metrics.get("issue_closing_rate", 0) * 100  # convert to percentage
        ]
        metrics["activity_score"] = sum(activity_factors) / len(activity_factors)
        
        return metrics
    
    def scrape_github_repo_data(self, repo_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Scrape GitHub repository data using web scraping as an alternative to the API.
        
        Args:
            repo_name: Repository name in 'owner/repo' format
            use_cache: Whether to use cached responses
            
        Returns:
            Dictionary of repository data
        """
        normalized_name = self.normalize_repo_name(repo_name)
        cache_path = self._get_cache_path(f"scrape_{normalized_name}")
        
        # Check cache first if enabled
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    logger.debug(f"Using cached scraped data for {repo_name}")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading cache for scraped data {repo_name}: {e}")
        
        # Base GitHub URL
        github_url = f"https://github.com/{normalized_name}"
        logger.info(f"Scraping GitHub repository: {github_url}")
        
        try:
            # Add a randomized user agent to avoid being blocked
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
            ]
            headers = {
                'User-Agent': random.choice(user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            
            # Make the request to the main repo page
            response = requests.get(github_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract data points
            data = {
                "repo_name": normalized_name,
                "scraped_data": True,
                "url": github_url
            }
            
            # Extract stars, forks, watchers, etc.
            try:
                # Look for the social count elements
                social_counts = soup.select('a.social-count')
                
                # Extract stars and forks
                for count_elem in social_counts:
                    count_text = count_elem.get_text(strip=True).replace(',', '')
                    if 'stargazers' in count_elem.get('href', ''):
                        data['stars'] = int(count_text) if count_text.isdigit() else 0
                    elif 'network/members' in count_elem.get('href', ''):
                        data['forks'] = int(count_text) if count_text.isdigit() else 0
                    elif 'watchers' in count_elem.get('href', ''):
                        data['watchers'] = int(count_text) if count_text.isdigit() else 0
            except Exception as e:
                logger.warning(f"Error extracting social counts for {repo_name}: {e}")
                
            # Try to extract repository description
            try:
                description_elem = soup.select_one('p.f4.my-3')
                if description_elem:
                    data['description'] = description_elem.get_text(strip=True)
            except Exception as e:
                logger.warning(f"Error extracting description for {repo_name}: {e}")
                
            # Extract language information
            try:
                language_elem = soup.select_one('span[itemprop="programmingLanguage"]')
                if language_elem:
                    data['language'] = language_elem.get_text(strip=True)
            except Exception as e:
                logger.warning(f"Error extracting language for {repo_name}: {e}")
                
            # Extract last update time
            try:
                update_elem = soup.select_one('relative-time')
                if update_elem and update_elem.has_attr('datetime'):
                    data['updated_at'] = update_elem['datetime']
            except Exception as e:
                logger.warning(f"Error extracting update time for {repo_name}: {e}")
            
            # Cache the data
            with open(cache_path, 'w') as f:
                json.dump(data, f)
                
            return data
            
        except Exception as e:
            logger.error(f"Error scraping GitHub repository {github_url}: {e}")
            return {"repo_name": normalized_name, "error": str(e)}
    
    def extract_repo_metrics_from_scrape(self, repo_name: str, use_cache: bool = True) -> Dict[str, float]:
        """
        Extract metrics from scraped GitHub repository data.
        
        Args:
            repo_name: Repository name in 'owner/repo' format
            use_cache: Whether to use cached responses
            
        Returns:
            Dictionary of repository metrics
        """
        scraped_data = self.scrape_github_repo_data(repo_name, use_cache)
        
        if not scraped_data or "error" in scraped_data:
            logger.warning(f"No scraped data found for repository: {repo_name}")
            return {}
        
        # Extract metrics from scraped data
        metrics = {
            "stars": scraped_data.get("stars", 0),
            "forks": scraped_data.get("forks", 0),
            "watchers": scraped_data.get("watchers", 0),
            "age_days": 365  # Default age for scraped repositories (1 year)
        }
        
        # Calculate age if possible
        if "updated_at" in scraped_data:
            try:
                # Handle timezone safely
                updated_time = pd.Timestamp(scraped_data["updated_at"]).tz_localize(None)  # Remove timezone
                metrics["last_updated_days"] = (pd.Timestamp.now() - updated_time).days
            except Exception as e:
                logger.warning(f"Error calculating update time for {repo_name}: {e}")
                metrics["last_updated_days"] = 30  # Default to 1 month
        
        # Generate some derived metrics based on what we have
        # Note: These are less accurate than the API metrics but provide something to work with
        if metrics["stars"] > 0:
            # Estimate number of contributors based on stars and forks
            estimated_contributors = max(1, int(metrics["stars"] * 0.01 + metrics["forks"] * 0.1))
            metrics["contributors"] = estimated_contributors
            
            # Estimate commit activity based on stars
            metrics["commit_frequency"] = metrics["stars"] * 0.05
            
            # Estimated contributor engagement
            metrics["contributor_engagement"] = 50  # Default middle value
            
            # Activity score (simplified)
            recency_factor = max(1, 30 / (metrics["last_updated_days"] + 1)) if "last_updated_days" in metrics else 1
            metrics["activity_score"] = (metrics["stars"] * 0.5 + metrics["forks"] * 5) * recency_factor / 100
        
        return metrics

    def extract_github_metrics_batch(self, repo_names: List[str], use_cache: bool = True, use_scraping: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Extract GitHub metrics for a batch of repositories.
        
        Args:
            repo_names: List of repository names
            use_cache: Whether to use cached responses
            use_scraping: Whether to use web scraping instead of API
            
        Returns:
            Dictionary mapping repository names to metrics dictionaries
        """
        logger.info(f"Extracting GitHub metrics for {len(repo_names)} repositories using {'web scraping' if use_scraping else 'API'}")
        
        metrics = {}
        for i, repo_name in enumerate(repo_names):
            logger.info(f"Processing repository {i+1}/{len(repo_names)}: {repo_name}")
            
            if use_scraping:
                # Use web scraping to get metrics
                metrics[repo_name] = self.extract_repo_metrics_from_scrape(repo_name, use_cache)
            else:
                # Use GitHub API to get metrics
                metrics[repo_name] = self.extract_repo_metrics(repo_name, use_cache)
            
            # Add a small delay to be nice to the GitHub/website
            if i < len(repo_names) - 1:
                time.sleep(0.5 if use_scraping else 0.2)  # Longer delay for web scraping
        
        return metrics
    
    def build_dependency_graph(self, root_repo: str, max_depth: int = 2, use_cache: bool = True) -> pd.DataFrame:
        """
        Build a dependency graph starting from a root repository.
        
        Args:
            root_repo: Root repository name
            max_depth: Maximum depth to traverse
            use_cache: Whether to use cached responses
            
        Returns:
            DataFrame with dependency relationships (parent, child)
        """
        logger.info(f"Building dependency graph from {root_repo} with max depth {max_depth}")
        
        # Initialize graph data
        edges = []
        visited = set()
        # Store as list of lists instead of tuples to avoid LSP type errors
        to_visit = [[self.normalize_repo_name(root_repo), 0]]  # [repo, depth]
        
        while to_visit:
            repo, depth = to_visit.pop(0)
            
            if repo in visited or depth > max_depth:
                continue
            
            visited.add(repo)
            logger.info(f"Processing dependencies for {repo} (depth {depth}/{max_depth})")
            
            # Get dependencies
            dependencies = self.get_repository_dependencies(repo, use_cache)
            
            for dep in dependencies:
                # Normalize dependency name
                normalized_dep = self.normalize_repo_name(dep)
                
                # Add edge
                edges.append((repo, normalized_dep))
                
                # Add to visit queue if not at max depth
                if depth < max_depth:
                    # Use list instead of tuple to avoid LSP type error
                    next_depth = depth + 1
                    to_visit.append([normalized_dep, next_depth])
        
        # Create DataFrame
        df = pd.DataFrame(edges, columns=["parent", "child"])
        logger.info(f"Built dependency graph with {len(df)} edges")
        
        return df
    
    def save_dependency_graph(self, df: pd.DataFrame, output_path: str = "data/dependency_graph.csv"):
        """
        Save dependency graph to CSV.
        
        Args:
            df: DataFrame with dependency graph
            output_path: Path to save CSV
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dependency graph saved to {output_path}")
    
    def load_dependency_graph(self, input_path: str = "data/dependency_graph.csv") -> pd.DataFrame:
        """
        Load dependency graph from CSV.
        
        Args:
            input_path: Path to load CSV from
            
        Returns:
            DataFrame with dependency graph
        """
        if os.path.exists(input_path):
            df = pd.read_csv(input_path)
            logger.info(f"Loaded dependency graph from {input_path} with {len(df)} edges")
            return df
        else:
            logger.warning(f"Dependency graph file {input_path} not found")
            return pd.DataFrame(columns=["parent", "child"])


# Example usage
if __name__ == "__main__":
    # Create builder with GitHub token (if available)
    token = os.environ.get("GITHUB_TOKEN")
    builder = GitHubDataBuilder(token=token)
    
    # Build dependency graph from root repository
    root_repo = "ethereum/solidity"
    df = builder.build_dependency_graph(root_repo, max_depth=1)
    
    # Save dependency graph
    builder.save_dependency_graph(df)
    
    # Extract metrics for all repositories in the graph
    all_repos = set(df["parent"].tolist() + df["child"].tolist())
    metrics = builder.extract_github_metrics_batch(list(all_repos))
    
    # Print some sample metrics
    for repo, repo_metrics in list(metrics.items())[:3]:
        print(f"Repository: {repo}")
        for metric, value in repo_metrics.items():
            print(f"  {metric}: {value}")
        print()