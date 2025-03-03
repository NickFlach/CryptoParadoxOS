# Blockchain Configuration Files

This directory contains configuration files for blockchain ecosystems supported by the Cryptopond Funding Allocation Tool.

## File Format

Each blockchain is represented by a JSON file with the following format:

```json
{
  "name": "ethereum",
  "display_name": "Ethereum",
  "description": "The world's programmable blockchain",
  "root_repository": "ethereum/go-ethereum",
  "seed_repositories": [
    "ethereum/solidity",
    "ethereum/web3.py",
    "ethereum/EIPs"
  ],
  "logo_url": "https://ethereum.org/static/a110735dade3f354a46fc2446cd52476/f3a29/eth-home-icon.webp",
  "primary_language": "Solidity",
  "github_org": "ethereum",
  "website": "https://ethereum.org",
  "documentation": "https://ethereum.org/developers/docs/",
  "chain_id": 1,
  "year_founded": 2015,
  "custom_parameters": {
    "consensus_mechanism": "Proof of Stake",
    "average_block_time": 12,
    "has_smart_contracts": true
  }
}
```

## Adding a New Blockchain

To add support for a new blockchain:

1. Create a new JSON file with the blockchain's name (e.g., `avalanche.json`)
2. Fill in the required fields as shown in the format above
3. Add any blockchain-specific parameters in the `custom_parameters` object
4. Consider implementing a custom adapter in `blockchain_manager.py` for blockchain-specific logic

## Required Fields

- `name`: Internal identifier for the blockchain (lowercase, no spaces)
- `display_name`: User-facing name of the blockchain
- `description`: Brief description of the blockchain
- `root_repository`: GitHub repository serving as the root node (format: `owner/repo`)
- `seed_repositories`: List of direct dependency repositories (format: `owner/repo`)
- `logo_url`: URL to the blockchain's logo
- `primary_language`: Primary programming language used in the blockchain

## Optional Fields

- `github_org`: Main GitHub organization for the blockchain
- `website`: Official website URL
- `documentation`: Developer documentation URL
- `chain_id`: Chain ID for EVM-compatible chains
- `year_founded`: Year the blockchain was launched
- `custom_parameters`: Any blockchain-specific parameters

## Supported Blockchains

Currently, the tool has built-in support for:

- Ethereum
- Solana
- Polkadot

Additional blockchains can be added through the UI or by creating configuration files in this directory.