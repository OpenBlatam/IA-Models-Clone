# Code Exports: Swarm_Fusion_Report

solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title SecureToken
 * @notice A minimal, auditable ERC-20 token with common DeFi safety patterns.
 * @dev Designed for gas efficiency while preventing reentrancy and unauthorized minting.
 */
contract SecureToken is ERC20, Ownable, ReentrancyGuard {
    // ──────────────────────────────────────────────────────────────
    //  STATE VARIABLES
    // ──────────────────────────────────────────────────────────────
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10 ** 18; // 1B tokens
    bool public paused;

    // ──────────────────────────────────────────────────────────────
    //  EVENTS (for off-chain monitoring / auditing)
    // ──────────────────────────────────────────────────────────────
    event Mint(address indexed to, uint256 amount);
    event Burn(address indexed from, uint256 amount);
    event Paused(bool state);

    // ──────────────────────────────────────────────────────────────
    //  MODIFIERS
    // ──────────────────────────────────────────────────────────────
    modifier whenNotPaused() {
        require(!paused, "Token: paused");
        _;
    }

    modifier notExceedMaxSupply(uint256 amount) {
        require(totalSupply() + amount <= MAX_SUPPLY, "Token: max supply");
        _;
    }

    // ──────────────────────────────────────────────────────────────
    //  CONSTRUCTOR
    // ──────────────────────────────────────────────────────────────
    constructor(
        string memory name,
        string memory symbol,
        uint256 initialSupply
    ) ERC20(name, symbol) Ownable(msg.sender) {
        require(initialSupply <= MAX_SUPPLY, "Token: exceeds max");
        _mint(msg.sender, initialSupply);
    }

    // ──────────────────────────────────────────────────────────────
    //  MINT (onlyOwner, with pausable & max supply check)
    // ──────────────────────────────────────────────────────────────
    function mint(address to, uint256 amount)
        external
        onlyOwner
        whenNotPaused
        notExceedMaxSupply(amount)
    {
        _mint(to, amount);
        emit Mint(to, amount);
    }

    // ──────────────────────────────────────────────────────────────
    //  BURN (anyone can burn their own tokens)
    // ──────────────────────────────────────────────────────────────
    function burn(uint256 amount) external whenNotPaused {
        _burn(msg.sender, amount);
        emit Burn(msg.sender, amount);
    }

    // ──────────────────────────────────────────────────────────────
    //  PAUSE / UNPAUSE (emergency stop)
    // ──────────────────────────────────────────────────────────────
    function pause() external onlyOwner {
        paused = true;
        emit Paused(true);
    }

    function unpause() external onlyOwner {
        paused = false;
        emit Paused(false);
    }

    // ──────────────────────────────────────────────────────────────
    //  OVERRIDE _update to enforce pause on *all* transfers
    // ──────────────────────────────────────────────────────────────
    function _update(
        address from,
        address to,
        uint256 value
    ) internal override whenNotPaused {
        super._update(from, to, value);
    }

    // ──────────────────────────────────────────────────────────────
    //  SECURITY / AUDIT NOTES
    // ──────────────────────────────────────────────────────────────
    // 1. ReentrancyGuard protects external calls (none here, but inherited for future).
    // 2. Ownable restricts mint/pause to deployer.
    // 3. MAX_SUPPLY is immutable in effect (constant) — no supply inflation.
    // 4. _update override blocks all transfers when paused (including mint/burn).
    // 5. No selfdestruct, no delegatecall, no unchecked loops.
    // 6. Use OpenZeppelin v5 — audited library.
}

---

import json
from web3 import Web3
from eth_account import Account
from solcx import compile_standard, install_solc

# 1. Compile Solidity (requires solcx)
install_solc("0.8.20")
with open("SecureToken.sol", "r") as f:
    source = f.read()

compiled = compile_standard(
    {
        "language": "Solidity",
        "sources": {"SecureToken.sol": {"content": source}},
        "settings": {
            "outputSelection": {"*": {"*": ["abi", "metadata", "evm.bytecode"]}}
        },
    },
    solc_version="0.8.20",
)
bytecode = compiled["contracts"]["SecureToken.sol"]["SecureToken"]["evm"]["bytecode"]["object"]
abi = json.loads(compiled["contracts"]["SecureToken.sol"]["SecureToken"]["metadata"])["output"]["abi"]

# 2. Connect to chain (example: Sepolia)
w3 = Web3(Web3.HTTPProvider("https://sepolia.infura.io/v3/YOUR_INFURA_KEY"))
account = Account.from_key("YOUR_PRIVATE_KEY")

# 3. Deploy
SecureToken = w3.eth.contract(abi=abi, bytecode=bytecode)
tx = SecureToken.constructor("MyToken", "MTK", 1000 * 10**18).build_transaction({
    "from": account.address,
    "nonce": w3.eth.get_transaction_count(account.address),
    "gas": 2000000,
    "gasPrice": w3.eth.gas_price
})
signed = account.sign_transaction(tx)
tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
print(f"Contract deployed at {receipt.contractAddress}")

# 4. Verify on Etherscan (requires Etherscan API)
# (omitted for brevity, but critical for transparency)