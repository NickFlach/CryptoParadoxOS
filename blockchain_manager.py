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
    
    def _create_default_cardano_config(self) -> BlockchainConfig:
        """
        Create a default Cardano blockchain configuration.
        
        Returns:
            BlockchainConfig for Cardano
        """
        return BlockchainConfig(
            name="cardano",
            display_name="Cardano",
            description="Blockchain platform for changemakers, innovators, and visionaries",
            root_repository="input-output-hk/cardano-node",
            seed_repositories=[
                "input-output-hk/cardano-wallet",
                "input-output-hk/cardano-ledger",
                "input-output-hk/plutus",
                "input-output-hk/ouroboros-network",
                "input-output-hk/cardano-db-sync",
                "input-output-hk/cardano-addresses",
                "IntersectMBO/cardano-cli",
                "cardano-foundation/CIPs",
                "cardano-foundation/cardano-token-registry",
                "Emurgo/cardano-serialization-lib"
            ],
            logo_url="https://cardano.org/static/cardano.a1942c9c.svg",
            primary_language="Haskell",
            github_org="input-output-hk",
            website="https://cardano.org",
            documentation="https://docs.cardano.org",
            year_founded=2015,
            custom_parameters={
                "consensus_mechanism": "Ouroboros Proof of Stake",
                "average_block_time": 20,  # in seconds
                "has_smart_contracts": True
            }
        )
    
    def _create_default_avalanche_config(self) -> BlockchainConfig:
        """
        Create a default Avalanche blockchain configuration.
        
        Returns:
            BlockchainConfig for Avalanche
        """
        return BlockchainConfig(
            name="avalanche",
            display_name="Avalanche",
            description="Blazingly fast, low cost, and eco-friendly platform for launching decentralized applications",
            root_repository="ava-labs/avalanchego",
            seed_repositories=[
                "ava-labs/coreth",
                "ava-labs/subnet-evm",
                "ava-labs/ava-docs",
                "ava-labs/avalanche-wallet",
                "ava-labs/avalanche-network-runner",
                "ava-labs/avalanche-faucet",
                "ava-labs/avalanchejs",
                "ava-labs/avalanche-bridge",
                "ava-labs/avalanche-plugins",
                "ava-labs/avalanche-cli"
            ],
            logo_url="https://assets-global.website-files.com/6059b554e81c705f9dd2dd32/60ec6a370ae6d248fcc5e943_AVA-Red.svg",
            primary_language="Go",
            github_org="ava-labs",
            website="https://avax.network",
            documentation="https://docs.avax.network",
            year_founded=2020,
            custom_parameters={
                "consensus_mechanism": "Avalanche Consensus",
                "average_block_time": 2,  # in seconds
                "has_smart_contracts": True
            }
        )
    
    def _create_default_near_config(self) -> BlockchainConfig:
        """
        Create a default NEAR Protocol blockchain configuration.
        
        Returns:
            BlockchainConfig for NEAR
        """
        return BlockchainConfig(
            name="near",
            display_name="NEAR Protocol",
            description="A developer-friendly, sharded, proof-of-stake public blockchain",
            root_repository="near/nearcore",
            seed_repositories=[
                "near/near-sdk-rs",
                "near/near-sdk-js",
                "near/near-api-js",
                "near/near-cli",
                "near/wallet-selector",
                "near/near-wallet",
                "near/docs",
                "near/borsh-js",
                "near/near-indexer-for-explorer",
                "near/bos-loader"
            ],
            logo_url="https://near.org/wp-content/uploads/2021/09/logo-white.svg",
            primary_language="Rust",
            github_org="near",
            website="https://near.org",
            documentation="https://docs.near.org",
            year_founded=2018,
            custom_parameters={
                "consensus_mechanism": "Nightshade Proof of Stake",
                "average_block_time": 1,  # in seconds
                "has_smart_contracts": True
            }
        )
    
    def _create_default_cosmos_config(self) -> BlockchainConfig:
        """
        Create a default Cosmos blockchain configuration.
        
        Returns:
            BlockchainConfig for Cosmos
        """
        return BlockchainConfig(
            name="cosmos",
            display_name="Cosmos",
            description="An ecosystem of interoperable blockchains that scale and interoperate with each other",
            root_repository="cosmos/cosmos-sdk",
            seed_repositories=[
                "cosmos/gaia",
                "cosmos/ibc-go",
                "cosmos/relayer",
                "cosmos/cosmjs",
                "cosmos/ibc",
                "cosmos/cosmos-proto",
                "cosmos/chain-registry",
                "cosmos/interchain-security",
                "cosmos/governance",
                "cosmos/cosmos.github.io"
            ],
            logo_url="https://v1.cosmos.network/img/logo.svg",
            primary_language="Go",
            github_org="cosmos",
            website="https://cosmos.network",
            documentation="https://docs.cosmos.network",
            year_founded=2016,
            custom_parameters={
                "consensus_mechanism": "Tendermint Proof of Stake",
                "average_block_time": 6.5,  # in seconds
                "has_smart_contracts": True
            }
        )
    
    def _create_default_algorand_config(self) -> BlockchainConfig:
        """
        Create a default Algorand blockchain configuration.
        
        Returns:
            BlockchainConfig for Algorand
        """
        return BlockchainConfig(
            name="algorand",
            display_name="Algorand",
            description="A public blockchain that achieves decentralization, scalability, and security without compromises",
            root_repository="algorand/go-algorand",
            seed_repositories=[
                "algorand/js-algorand-sdk",
                "algorand/py-algorand-sdk",
                "algorand/algorand-sdk",
                "algorand/indexer",
                "algorand/docs",
                "algorand/algorand-walletconnect-example",
                "algorand/go-algorand-sdk",
                "algorand/mese-wallet",
                "algorand/smart-contracts",
                "algorand/proposals"
            ],
            logo_url="https://algorand.com/static/img/og.png",
            primary_language="Go",
            github_org="algorand",
            website="https://algorand.com",
            documentation="https://developer.algorand.org/docs",
            year_founded=2017,
            custom_parameters={
                "consensus_mechanism": "Pure Proof of Stake",
                "average_block_time": 4.5,  # in seconds
                "has_smart_contracts": True
            }
        )
    
    def _create_default_tezos_config(self) -> BlockchainConfig:
        """
        Create a default Tezos blockchain configuration.
        
        Returns:
            BlockchainConfig for Tezos
        """
        return BlockchainConfig(
            name="tezos",
            display_name="Tezos",
            description="Self-amending blockchain with formal verification for smart contract security",
            root_repository="tezos/tezos",
            seed_repositories=[
                "oxheadalpha/smart-contracts",
                "Marigold-Dev/tezos-nix",
                "trilitech/jstz",
                "tezos-commons/tzkt-api",
                "madfish-solutions/templewallet-extension",
                "ecadlabs/taquito",
                "marigold-dev/tzstamp",
                "baking-bad/netezos",
                "tezos-commons/tezos-core-tools",
                "SimplestakingTezosWallet/tezos-wallet-client"
            ],
            logo_url="https://tezos.com/img/logo-tezos-blue.svg",
            primary_language="OCaml",
            github_org="tezos",
            website="https://tezos.com",
            documentation="https://tezos.gitlab.io/active/index.html",
            year_founded=2014,
            custom_parameters={
                "consensus_mechanism": "Liquid Proof of Stake",
                "average_block_time": 30,  # in seconds
                "has_smart_contracts": True
            }
        )
        
    def add_default_blockchains(self) -> None:
        """
        Add default blockchain configurations.
        """
        ethereum_config = self._create_default_ethereum_config()
        solana_config = self._create_default_solana_config()
        polkadot_config = self._create_default_polkadot_config()
        cardano_config = self._create_default_cardano_config()
        avalanche_config = self._create_default_avalanche_config()
        near_config = self._create_default_near_config()
        cosmos_config = self._create_default_cosmos_config()
        algorand_config = self._create_default_algorand_config()
        tezos_config = self._create_default_tezos_config()
        
        self.add_blockchain(ethereum_config)
        self.add_blockchain(solana_config)
        self.add_blockchain(polkadot_config)
        self.add_blockchain(cardano_config)
        self.add_blockchain(avalanche_config)
        self.add_blockchain(near_config)
        self.add_blockchain(cosmos_config)
        self.add_blockchain(algorand_config)
        self.add_blockchain(tezos_config)
    
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
            "polkadot": PolkadotAdapter,
            # Default to base adapter for other blockchains for now
        }
        
        # Get the adapter class or use the base class as fallback
        adapter_class = adapter_map.get(config.name, BlockchainAdapter)
        return adapter_class(config)