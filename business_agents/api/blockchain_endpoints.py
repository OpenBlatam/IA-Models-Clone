"""
Blockchain API Endpoints
========================

REST API endpoints for blockchain integration, smart contracts,
and decentralized workflow execution.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.blockchain_service import (
    BlockchainService, BlockchainType, ContractType, TransactionStatus,
    SmartContract, BlockchainTransaction, WorkflowBlock
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/blockchain", tags=["Blockchain"])

# Pydantic models
class WorkflowExecutionRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID to execute")
    execution_data: Dict[str, Any] = Field(..., description="Workflow execution data")
    network: str = Field("ethereum", description="Blockchain network")

class PaymentRequest(BaseModel):
    to_address: str = Field(..., description="Recipient address")
    amount: int = Field(..., description="Payment amount")
    currency: str = Field("ETH", description="Currency")
    network: str = Field("ethereum", description="Blockchain network")

class IdentityVerificationRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    identity_hash: str = Field(..., description="Identity hash")

class SmartContractDeployRequest(BaseModel):
    contract_name: str = Field(..., description="Contract name")
    contract_type: str = Field(..., description="Contract type")
    abi: Dict[str, Any] = Field(..., description="Contract ABI")
    bytecode: str = Field(..., description="Contract bytecode")
    network: str = Field("ethereum", description="Blockchain network")

class WorkflowVerificationRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID to verify")

# Global blockchain service instance
blockchain_service = None

def get_blockchain_service() -> BlockchainService:
    """Get global blockchain service instance."""
    global blockchain_service
    if blockchain_service is None:
        blockchain_service = BlockchainService({"blockchain_enabled": True})
    return blockchain_service

# API Endpoints

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_blockchain_service(
    current_user: User = Depends(require_permission("blockchain:manage"))
):
    """Initialize the blockchain service."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        await blockchain_service.initialize()
        return {"message": "Blockchain Service initialized successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize blockchain service: {str(e)}")

@router.get("/status", response_model=Dict[str, Any])
async def get_blockchain_status(
    current_user: User = Depends(require_permission("blockchain:view"))
):
    """Get blockchain service status."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        status = await blockchain_service.get_blockchain_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get blockchain status: {str(e)}")

@router.post("/workflow/execute", response_model=Dict[str, Any])
async def execute_workflow_on_blockchain(
    request: WorkflowExecutionRequest,
    current_user: User = Depends(require_permission("workflows:execute"))
):
    """Execute workflow on blockchain."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        # Convert string to enum
        network = BlockchainType(request.network)
        
        # Execute workflow on blockchain
        transaction = await blockchain_service.execute_workflow_on_blockchain(
            request.workflow_id,
            request.execution_data,
            network
        )
        
        return {
            "transaction_id": transaction.transaction_id,
            "transaction_hash": transaction.hash,
            "status": transaction.status.value,
            "workflow_id": request.workflow_id,
            "network": request.network,
            "timestamp": transaction.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute workflow on blockchain: {str(e)}")

@router.post("/workflow/verify", response_model=Dict[str, Any])
async def verify_workflow_execution(
    request: WorkflowVerificationRequest,
    current_user: User = Depends(require_permission("workflows:view"))
):
    """Verify workflow execution on blockchain."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        verification_result = await blockchain_service.verify_workflow_execution(request.workflow_id)
        return verification_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify workflow execution: {str(e)}")

@router.get("/workflow/blocks/{workflow_id}", response_model=List[Dict[str, Any]])
async def get_workflow_blocks(
    workflow_id: str,
    current_user: User = Depends(require_permission("blockchain:view"))
):
    """Get workflow blocks for a specific workflow."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        blocks = await blockchain_service.get_workflow_blocks(workflow_id)
        
        result = []
        for block in blocks:
            block_dict = {
                "block_id": block.block_id,
                "workflow_id": block.workflow_id,
                "execution_id": block.execution_id,
                "block_hash": block.block_hash,
                "previous_hash": block.previous_hash,
                "timestamp": block.timestamp.isoformat(),
                "data": block.data,
                "nonce": block.nonce,
                "merkle_root": block.merkle_root,
                "signature": block.signature
            }
            result.append(block_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow blocks: {str(e)}")

@router.get("/blocks", response_model=List[Dict[str, Any]])
async def get_all_blocks(
    limit: int = Query(100, description="Maximum number of blocks to return"),
    current_user: User = Depends(require_permission("blockchain:view"))
):
    """Get all workflow blocks."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        blocks = await blockchain_service.get_workflow_blocks()
        
        # Limit results
        limited_blocks = blocks[-limit:] if limit else blocks
        
        result = []
        for block in limited_blocks:
            block_dict = {
                "block_id": block.block_id,
                "workflow_id": block.workflow_id,
                "execution_id": block.execution_id,
                "block_hash": block.block_hash,
                "previous_hash": block.previous_hash,
                "timestamp": block.timestamp.isoformat(),
                "data": block.data,
                "nonce": block.nonce,
                "merkle_root": block.merkle_root,
                "signature": block.signature
            }
            result.append(block_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get blocks: {str(e)}")

@router.get("/transactions", response_model=List[Dict[str, Any]])
async def get_transaction_history(
    limit: int = Query(100, description="Maximum number of transactions to return"),
    current_user: User = Depends(require_permission("blockchain:view"))
):
    """Get blockchain transaction history."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        transactions = await blockchain_service.get_transaction_history(limit)
        
        result = []
        for transaction in transactions:
            transaction_dict = {
                "transaction_id": transaction.transaction_id,
                "hash": transaction.hash,
                "from_address": transaction.from_address,
                "to_address": transaction.to_address,
                "value": transaction.value,
                "gas_used": transaction.gas_used,
                "gas_price": transaction.gas_price,
                "status": transaction.status.value,
                "block_number": transaction.block_number,
                "timestamp": transaction.timestamp.isoformat(),
                "data": transaction.data,
                "receipt": transaction.receipt
            }
            result.append(transaction_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transaction history: {str(e)}")

@router.post("/payment/process", response_model=Dict[str, Any])
async def process_payment(
    request: PaymentRequest,
    current_user: User = Depends(require_permission("blockchain:execute"))
):
    """Process payment through blockchain."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        # Convert string to enum
        network = BlockchainType(request.network)
        
        # Process payment
        transaction = await blockchain_service.process_payment(
            request.to_address,
            request.amount,
            request.currency,
            network
        )
        
        return {
            "transaction_id": transaction.transaction_id,
            "transaction_hash": transaction.hash,
            "status": transaction.status.value,
            "amount": request.amount,
            "currency": request.currency,
            "to_address": request.to_address,
            "network": request.network,
            "timestamp": transaction.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process payment: {str(e)}")

@router.post("/identity/verify", response_model=Dict[str, Any])
async def verify_identity(
    request: IdentityVerificationRequest,
    current_user: User = Depends(require_permission("blockchain:execute"))
):
    """Verify user identity on blockchain."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        verification_result = await blockchain_service.verify_identity(
            request.user_id,
            request.identity_hash
        )
        
        return verification_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify identity: {str(e)}")

@router.post("/contracts/deploy", response_model=Dict[str, Any])
async def deploy_smart_contract(
    request: SmartContractDeployRequest,
    current_user: User = Depends(require_permission("blockchain:manage"))
):
    """Deploy smart contract to blockchain."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        # Convert string to enum
        contract_type = ContractType(request.contract_type)
        network = BlockchainType(request.network)
        
        # Deploy smart contract
        contract = await blockchain_service.deploy_smart_contract(
            request.contract_name,
            contract_type,
            request.abi,
            request.bytecode,
            network
        )
        
        return {
            "contract_id": contract.contract_id,
            "name": contract.name,
            "contract_type": contract.contract_type.value,
            "address": contract.address,
            "network": contract.network.value,
            "deployed_at": contract.deployed_at.isoformat(),
            "owner": contract.owner
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deploy smart contract: {str(e)}")

@router.get("/contracts", response_model=List[Dict[str, Any]])
async def get_smart_contracts(
    current_user: User = Depends(require_permission("blockchain:view"))
):
    """Get deployed smart contracts."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        contracts = list(blockchain_service.smart_contracts.values())
        
        result = []
        for contract in contracts:
            contract_dict = {
                "contract_id": contract.contract_id,
                "name": contract.name,
                "contract_type": contract.contract_type.value,
                "address": contract.address,
                "network": contract.network.value,
                "deployed_at": contract.deployed_at.isoformat(),
                "owner": contract.owner,
                "metadata": contract.metadata
            }
            result.append(contract_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get smart contracts: {str(e)}")

@router.get("/contracts/{contract_id}", response_model=Dict[str, Any])
async def get_smart_contract(
    contract_id: str,
    current_user: User = Depends(require_permission("blockchain:view"))
):
    """Get specific smart contract details."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        contract = None
        for c in blockchain_service.smart_contracts.values():
            if c.contract_id == contract_id:
                contract = c
                break
                
        if not contract:
            raise HTTPException(status_code=404, detail="Smart contract not found")
        
        return {
            "contract_id": contract.contract_id,
            "name": contract.name,
            "contract_type": contract.contract_type.value,
            "address": contract.address,
            "abi": contract.abi,
            "bytecode": contract.bytecode,
            "network": contract.network.value,
            "deployed_at": contract.deployed_at.isoformat(),
            "owner": contract.owner,
            "metadata": contract.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get smart contract: {str(e)}")

@router.get("/networks", response_model=List[Dict[str, Any]])
async def get_supported_networks(
    current_user: User = Depends(require_permission("blockchain:view"))
):
    """Get supported blockchain networks."""
    
    try:
        networks = []
        for network_type in BlockchainType:
            networks.append({
                "name": network_type.value,
                "display_name": network_type.value.title(),
                "chain_id": getattr(blockchain_service.blockchain_configs.get(network_type), 'chain_id', 0) if blockchain_service else 0,
                "supported": True
            })
        
        return networks
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get supported networks: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def blockchain_health_check():
    """Blockchain service health check."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        # Check if service is initialized
        initialized = hasattr(blockchain_service, 'smart_contracts') and len(blockchain_service.smart_contracts) > 0
        
        # Get service status
        status = await blockchain_service.get_blockchain_status()
        
        return {
            "status": "healthy" if initialized else "initializing",
            "initialized": initialized,
            "networks_configured": status.get("networks_configured", 0),
            "smart_contracts_deployed": status.get("smart_contracts_deployed", 0),
            "total_transactions": status.get("total_transactions", 0),
            "total_blocks": status.get("total_blocks", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/blockchain/verify", response_model=Dict[str, Any])
async def verify_blockchain_integrity(
    current_user: User = Depends(require_permission("blockchain:view"))
):
    """Verify blockchain integrity."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        # Get all blocks
        blocks = await blockchain_service.get_workflow_blocks()
        
        # Verify integrity
        verification_result = await blockchain_service._verify_blockchain_integrity(blocks)
        
        return {
            "integrity_check": verification_result,
            "total_blocks": len(blocks),
            "verification_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify blockchain integrity: {str(e)}")

@router.get("/analytics", response_model=Dict[str, Any])
async def get_blockchain_analytics(
    current_user: User = Depends(require_permission("blockchain:view"))
):
    """Get blockchain analytics."""
    
    blockchain_service = get_blockchain_service()
    
    try:
        # Get transaction history
        transactions = await blockchain_service.get_transaction_history()
        
        # Get blocks
        blocks = await blockchain_service.get_workflow_blocks()
        
        # Calculate analytics
        analytics = {
            "total_transactions": len(transactions),
            "total_blocks": len(blocks),
            "successful_transactions": len([t for t in transactions if t.status == TransactionStatus.CONFIRMED]),
            "failed_transactions": len([t for t in transactions if t.status == TransactionStatus.FAILED]),
            "pending_transactions": len([t for t in transactions if t.status == TransactionStatus.PENDING]),
            "average_gas_used": sum(t.gas_used for t in transactions) / len(transactions) if transactions else 0,
            "total_gas_used": sum(t.gas_used for t in transactions),
            "blockchain_height": len(blocks) - 1,  # Exclude genesis block
            "unique_workflows": len(set(block.workflow_id for block in blocks)),
            "last_block_timestamp": blocks[-1].timestamp.isoformat() if blocks else None,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get blockchain analytics: {str(e)}")




























