"""
Blockchain Ecosystem Manager for CryptoPond

This module provides functionality for managing different blockchain ecosystems,
including configuration management, registration, and adapter interfaces.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class BlockchainConfig:
    """Configuration for a blockchain ecosystem."""
    name: str
    display_name: str
    description: str
    root_repository: str
    seed_repositories: List[str]
    logo_url: str
    primary_language: str  # e.g., Solidity, Rust, Go
    github_org: Optional[str] = None  # Main GitHub organization
    website: Optional[str] = None
    documentation: Optional[str] = None
    chain_id: Optional[int] = None  # For EVM chains
    year_founded: Optional[int] = None
    custom_parameters: Optional[Dict[str, Any]] = None


class BlockchainManager:
    """
    Manager for blockchain ecosystem configurations.
    Provides functionality to add, remove, and modify blockchain ecosystem configs.
    """
    
    def __init__(self, config_dir: str = "data/blockchains"):
        """
        Initialize the blockchain manager.
        
        Args:
            config_dir: Directory to store blockchain configurations
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        self.blockchains = self._load_blockchains()
        
    def _load_blockchains(self) -> Dict[str, BlockchainConfig]:
        """
        Load blockchain configurations from the filesystem.
        
        Returns:
            Dictionary mapping blockchain IDs to their configurations
        """
        blockchains = {}
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                blockchain_id = filename.split('.')[0]
                with open(os.path.join(self.config_dir, filename), 'r') as f:
                    config_data = json.load(f)
                    blockchains[blockchain_id] = BlockchainConfig(**config_data)
        
        # If no blockchains found, create default Ethereum config
        if not blockchains:
            ethereum_config = self._create_default_ethereum_config()
            self.add_blockchain(ethereum_config)
            blockchains["ethereum"] = ethereum_config
            
        return blockchains
    
    def _create_default_ethereum_config(self) -> BlockchainConfig:
        """
        Create a default Ethereum blockchain configuration.
        
        Returns:
            BlockchainConfig for Ethereum
        """
        return BlockchainConfig(
            name="ethereum",
            display_name="Ethereum",
            description="The world's programmable blockchain",
            root_repository="ethereum/go-ethereum",
            seed_repositories=[
                "ethereum/solidity",
                "ethereum/web3.py",
                "ethereum/EIPs",
                "ethereum/pm",
                "ethereum/fe",
                "ethereum/py-evm",
                "ethereum/eth2.0-specs",
                "ethereum/execution-apis",
                "ethereum/evmc",
                "ethereum/sourcify"
            ],
            logo_url="https://ethereum.org/static/a110735dade3f354a46fc2446cd52476/f3a29/eth-home-icon.webp",
            primary_language="Solidity",
            github_org="ethereum",
            website="https://ethereum.org",
            documentation="https://ethereum.org/developers/docs/",
            chain_id=1,
            year_founded=2015,
            custom_parameters={
                "consensus_mechanism": "Proof of Stake",
                "average_block_time": 12,  # in seconds
                "has_smart_contracts": True
            }
        )
    
    def _create_default_solana_config(self) -> BlockchainConfig:
        """
        Create a default Solana blockchain configuration.
        
        Returns:
            BlockchainConfig for Solana
        """
        return BlockchainConfig(
            name="solana",
            display_name="Solana",
            description="High-performance blockchain supporting builders around the world",
            root_repository="solana-labs/solana",
            seed_repositories=[
                "solana-labs/solana-program-library",
                "solana-labs/solana-web3.js",
                "solana-labs/wallet-adapter",
                "solana-labs/anchor",
                "solana-labs/token-list",
                "solana-labs/ecosystem",
                "solana-labs/governance-ui",
                "solana-labs/example-helloworld",
                "solana-labs/dapp-scaffold",
                "solana-labs/solana-pay"
            ],
            logo_url="https://solana.com/src/images/branding/solanaLogoMark.svg",
            primary_language="Rust",
            github_org="solana-labs",
            website="https://solana.com",
            documentation="https://docs.solana.com",
            year_founded=2017,
            custom_parameters={
                "consensus_mechanism": "Proof of History + Proof of Stake",
                "average_block_time": 0.4,  # in seconds
                "has_smart_contracts": True
            }
        )
    
    def _create_default_polkadot_config(self) -> BlockchainConfig:
        """
        Create a default Polkadot blockchain configuration.
        
        Returns:
            BlockchainConfig for Polkadot
        """
        return BlockchainConfig(
            name="polkadot",
            display_name="Polkadot",
            description="A scalable, interoperable & secure network protocol for the next web",
            root_repository="paritytech/polkadot",
            seed_repositories=[
                "paritytech/substrate",
                "paritytech/cumulus",
                "paritytech/parity-bridges-common",
                "paritytech/ink",
                "paritytech/subxt",
                "paritytech/polkadot-sdk",
                "paritytech/parity-signer",
                "paritytech/txwrapper",
                "paritytech/frontier",
                "paritytech/smoldot"
            ],
            logo_url="https://polkadot.network/assets/img/logo-polkadot.svg",
            primary_language="Rust",
            github_org="paritytech",
            website="https://polkadot.network",
            documentation="https://wiki.polkadot.network",
            year_founded=2016,
            custom_parameters={
                "consensus_mechanism": "NPoS (Nominated Proof of Stake)",
                "average_block_time": 6,  # in seconds
                "has_smart_contracts": True
            }
        )
    
    def add_blockchain(self, config: BlockchainConfig) -> None:
        """
        Add a new blockchain configuration.
        
        Args:
            config: BlockchainConfig object with blockchain details
        """
        # Store in memory
        self.blockchains[config.name] = config
        
        # Save to file
        config_path = os.path.join(self.config_dir, f"{config.name}.json")
        with open(config_path, 'w') as f:
            # Convert dataclass to dict for serialization
            config_dict = {
                field: getattr(config, field) 
                for field in [f.name for f in config.__dataclass_fields__.values()]
            }
            json.dump(config_dict, f, indent=2)
    
    def remove_blockchain(self, blockchain_id: str) -> bool:
        """
        Remove a blockchain configuration.
        
        Args:
            blockchain_id: ID of the blockchain to remove
            
        Returns:
            True if removal was successful, False otherwise
        """
        if blockchain_id in self.blockchains:
            # Remove from memory
            del self.blockchains[blockchain_id]
            
            # Remove from filesystem
            config_path = os.path.join(self.config_dir, f"{blockchain_id}.json")
            if os.path.exists(config_path):
                os.remove(config_path)
            return True
        return False
    
    def get_blockchain(self, blockchain_id: str) -> Optional[BlockchainConfig]:
        """
        Get a blockchain configuration by ID.
        
        Args:
            blockchain_id: ID of the blockchain to retrieve
            
        Returns:
            BlockchainConfig if found, None otherwise
        """
        return self.blockchains.get(blockchain_id)
    
    def get_all_blockchains(self) -> Dict[str, BlockchainConfig]:
        """
        Get all blockchain configurations.
        
        Returns:
            Dictionary mapping blockchain IDs to their configurations
        """
        return self.blockchains
    
    def get_blockchain_list(self) -> List[Tuple[str, str]]:
        """
        Get a list of blockchain IDs and display names for UI selection.
        
        Returns:
            List of (id, display_name) tuples
        """
        return [(b_id, config.display_name) for b_id, config in self.blockchains.items()]
    
    def update_blockchain(self, blockchain_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a blockchain configuration.
        
        Args:
            blockchain_id: ID of the blockchain to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update was successful, False otherwise
        """
        if blockchain_id in self.blockchains:
            config = self.blockchains[blockchain_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(config, field):
                    setattr(config, field, value)
            
            # Save updated config
            self.add_blockchain(config)
            return True
        return False
    
    def add_default_blockchains(self) -> None:
        """
        Add default blockchain configurations (Ethereum, Solana, Polkadot).
        """
        ethereum_config = self._create_default_ethereum_config()
        solana_config = self._create_default_solana_config()
        polkadot_config = self._create_default_polkadot_config()
        
        self.add_blockchain(ethereum_config)
        self.add_blockchain(solana_config)
        self.add_blockchain(polkadot_config)
    
    def export_blockchain_configs_to_dataframe(self) -> pd.DataFrame:
        """
        Export blockchain configurations to a pandas DataFrame.
        
        Returns:
            DataFrame with blockchain configurations
        """
        configs_list = []
        for b_id, config in self.blockchains.items():
            config_dict = {
                'id': b_id,
                'name': config.display_name,
                'description': config.description,
                'root_repository': config.root_repository,
                'num_seed_repositories': len(config.seed_repositories),
                'primary_language': config.primary_language,
                'github_org': config.github_org,
                'website': config.website,
                'year_founded': config.year_founded
            }
            configs_list.append(config_dict)
        
        return pd.DataFrame(configs_list)


# Adapter interface for blockchain-specific operations
class BlockchainAdapter:
    """
    Base adapter interface for blockchain-specific operations.
    Concrete implementations should be created for each blockchain.
    """
    
    def __init__(self, config: BlockchainConfig):
        """
        Initialize the blockchain adapter.
        
        Args:
            config: BlockchainConfig for the specific blockchain
        """
        self.config = config
    
    def get_seed_repositories(self) -> List[str]:
        """
        Get seed repositories for the blockchain.
        
        Returns:
            List of seed repository identifiers
        """
        return self.config.seed_repositories
    
    def get_root_repository(self) -> str:
        """
        Get root repository for the blockchain.
        
        Returns:
            Root repository identifier
        """
        return self.config.root_repository
    
    def adjust_pagerank_parameters(self) -> Dict[str, float]:
        """
        Get blockchain-specific PageRank parameters.
        
        Returns:
            Dictionary of PageRank parameters
        """
        # Default parameters, override in specific adapters
        return {
            'alpha': 0.85,
            'max_iter': 100,
            'tol': 1e-6
        }
    
    def adjust_contribution_weight(self) -> float:
        """
        Get blockchain-specific contribution weight parameter.
        
        Returns:
            Contribution weight value
        """
        # Default weight, override in specific adapters
        return 0.7


# Factory for creating blockchain-specific adapters
class BlockchainAdapterFactory:
    """
    Factory for creating blockchain-specific adapters.
    """
    
    @staticmethod
    def create_adapter(config: BlockchainConfig) -> BlockchainAdapter:
        """
        Create an appropriate adapter for the given blockchain config.
        
        Args:
            config: BlockchainConfig for the specific blockchain
            
        Returns:
            Blockchain-specific adapter instance
        """
        # In the future, use a registry of adapter classes
        # For now, return the base adapter
        return BlockchainAdapter(config)


# Concrete adapter implementations could be added here
# or in separate modules for each blockchain

class EthereumAdapter(BlockchainAdapter):
    """Ethereum-specific adapter implementation."""
    
    def adjust_pagerank_parameters(self) -> Dict[str, float]:
        """
        Get Ethereum-specific PageRank parameters.
        
        Returns:
            Dictionary of PageRank parameters
        """
        return {
            'alpha': 0.85,  # standard damping parameter
            'max_iter': 100,
            'tol': 1e-6
        }
    
    def adjust_contribution_weight(self) -> float:
        """
        Get Ethereum-specific contribution weight parameter.
        
        Returns:
            Contribution weight value
        """
        return 0.7  # Balanced between PageRank and GitHub metrics


class SolanaAdapter(BlockchainAdapter):
    """Solana-specific adapter implementation."""
    
    def adjust_pagerank_parameters(self) -> Dict[str, float]:
        """
        Get Solana-specific PageRank parameters.
        
        Returns:
            Dictionary of PageRank parameters
        """
        return {
            'alpha': 0.80,  # slightly lower to reflect more distributed ecosystem
            'max_iter': 100,
            'tol': 1e-6
        }
    
    def adjust_contribution_weight(self) -> float:
        """
        Get Solana-specific contribution weight parameter.
        
        Returns:
            Contribution weight value
        """
        return 0.65  # Slightly more weight to GitHub metrics


class PolkadotAdapter(BlockchainAdapter):
    """Polkadot-specific adapter implementation."""
    
    def adjust_pagerank_parameters(self) -> Dict[str, float]:
        """
        Get Polkadot-specific PageRank parameters.
        
        Returns:
            Dictionary of PageRank parameters
        """
        return {
            'alpha': 0.90,  # higher to reflect more concentrated ecosystem
            'max_iter': 100,
            'tol': 1e-6
        }
    
    def adjust_contribution_weight(self) -> float:
        """
        Get Polkadot-specific contribution weight parameter.
        
        Returns:
            Contribution weight value
        """
        return 0.75  # More weight to PageRank given centralized development


# Update the factory to use concrete implementations
class EnhancedBlockchainAdapterFactory:
    """
    Enhanced factory for creating blockchain-specific adapters.
    """
    
    @staticmethod
    def create_adapter(config: BlockchainConfig) -> BlockchainAdapter:
        """
        Create an appropriate adapter for the given blockchain config.
        
        Args:
            config: BlockchainConfig for the specific blockchain
            
        Returns:
            Blockchain-specific adapter instance
        """
        # Map blockchain names to their adapter classes
        adapter_map = {
            "ethereum": EthereumAdapter,
            "solana": SolanaAdapter,
            "polkadot": PolkadotAdapter
        }
        
        # Get the adapter class or use the base class as fallback
        adapter_class = adapter_map.get(config.name, BlockchainAdapter)
        return adapter_class(config)