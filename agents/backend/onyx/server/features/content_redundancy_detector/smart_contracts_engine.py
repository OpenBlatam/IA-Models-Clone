"""
Smart Contracts Engine for Advanced Smart Contract Processing
Motor de Contratos Inteligentes para procesamiento avanzado de contratos inteligentes ultra-optimizado
"""

import asyncio
import logging
import time
import json
import threading
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from datetime import datetime, timedelta
import statistics
import random
import math

logger = logging.getLogger(__name__)


class ContractType(Enum):
    """Tipos de contratos inteligentes"""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    ERC777 = "erc777"
    ERC1400 = "erc1400"
    ERC3525 = "erc3525"
    CUSTOM = "custom"
    DAO = "dao"
    DEFI = "defi"
    NFT = "nft"
    GAMING = "gaming"
    SUPPLY_CHAIN = "supply_chain"
    IDENTITY = "identity"
    VOTING = "voting"
    ESCROW = "escrow"


class ContractStatus(Enum):
    """Estados de contratos inteligentes"""
    DRAFT = "draft"
    DEPLOYED = "deployed"
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"
    UPGRADED = "upgraded"
    FAILED = "failed"


class ContractLanguage(Enum):
    """Lenguajes de contratos inteligentes"""
    SOLIDITY = "solidity"
    VYPER = "vyper"
    RUST = "rust"
    GO = "go"
    JAVASCRIPT = "javascript"
    PYTHON = "python"
    C_SHARP = "c_sharp"
    JAVA = "java"


@dataclass
class SmartContract:
    """Contrato inteligente"""
    id: str
    name: str
    description: str
    contract_type: ContractType
    language: ContractLanguage
    status: ContractStatus
    source_code: str
    bytecode: str
    abi: Dict[str, Any]
    address: Optional[str]
    deployer: Optional[str]
    gas_used: int
    gas_price: int
    transaction_hash: Optional[str]
    block_number: Optional[int]
    created_at: float
    deployed_at: Optional[float]
    last_modified: float
    metadata: Dict[str, Any]


@dataclass
class ContractExecution:
    """Ejecución de contrato inteligente"""
    id: str
    contract_id: str
    function_name: str
    parameters: Dict[str, Any]
    gas_limit: int
    gas_used: int
    status: str
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    transaction_hash: Optional[str]
    block_number: Optional[int]
    created_at: float
    executed_at: Optional[float]
    execution_time: float
    metadata: Dict[str, Any]


class SmartContractCompiler:
    """Compilador de contratos inteligentes"""
    
    def __init__(self):
        self.compilers: Dict[ContractLanguage, Callable] = {
            ContractLanguage.SOLIDITY: self._compile_solidity,
            ContractLanguage.VYPER: self._compile_vyper,
            ContractLanguage.RUST: self._compile_rust,
            ContractLanguage.GO: self._compile_go,
            ContractLanguage.JAVASCRIPT: self._compile_javascript,
            ContractLanguage.PYTHON: self._compile_python,
            ContractLanguage.C_SHARP: self._compile_csharp,
            ContractLanguage.JAVA: self._compile_java
        }
    
    async def compile_contract(self, source_code: str, language: ContractLanguage) -> Dict[str, Any]:
        """Compilar contrato inteligente"""
        try:
            compiler = self.compilers.get(language)
            if not compiler:
                raise ValueError(f"Unsupported contract language: {language}")
            
            return await compiler(source_code)
            
        except Exception as e:
            logger.error(f"Error compiling contract: {e}")
            raise
    
    async def _compile_solidity(self, source_code: str) -> Dict[str, Any]:
        """Compilar contrato Solidity"""
        # Simular compilación de Solidity
        bytecode = f"0x{hashlib.sha256(source_code.encode()).hexdigest()[:64]}"
        
        abi = [
            {
                "inputs": [],
                "name": "constructor",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "inputs": [],
                "name": "getValue",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "_value", "type": "uint256"}],
                "name": "setValue",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        return {
            "bytecode": bytecode,
            "abi": abi,
            "compiler_version": "0.8.19",
            "optimization": True,
            "gas_estimate": 200000
        }
    
    async def _compile_vyper(self, source_code: str) -> Dict[str, Any]:
        """Compilar contrato Vyper"""
        bytecode = f"0x{hashlib.sha256(source_code.encode()).hexdigest()[:64]}"
        
        abi = [
            {
                "name": "get_value",
                "outputs": [{"type": "uint256", "name": ""}],
                "inputs": [],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "name": "set_value",
                "outputs": [],
                "inputs": [{"type": "uint256", "name": "_value"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        return {
            "bytecode": bytecode,
            "abi": abi,
            "compiler_version": "0.3.7",
            "optimization": True,
            "gas_estimate": 180000
        }
    
    async def _compile_rust(self, source_code: str) -> Dict[str, Any]:
        """Compilar contrato Rust"""
        bytecode = f"0x{hashlib.sha256(source_code.encode()).hexdigest()[:64]}"
        
        abi = [
            {
                "name": "get_value",
                "outputs": [{"type": "u64", "name": ""}],
                "inputs": [],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "name": "set_value",
                "outputs": [],
                "inputs": [{"type": "u64", "name": "_value"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        return {
            "bytecode": bytecode,
            "abi": abi,
            "compiler_version": "1.70.0",
            "optimization": True,
            "gas_estimate": 150000
        }
    
    async def _compile_go(self, source_code: str) -> Dict[str, Any]:
        """Compilar contrato Go"""
        bytecode = f"0x{hashlib.sha256(source_code.encode()).hexdigest()[:64]}"
        
        abi = [
            {
                "name": "GetValue",
                "outputs": [{"type": "uint64", "name": ""}],
                "inputs": [],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "name": "SetValue",
                "outputs": [],
                "inputs": [{"type": "uint64", "name": "_value"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        return {
            "bytecode": bytecode,
            "abi": abi,
            "compiler_version": "1.21.0",
            "optimization": True,
            "gas_estimate": 160000
        }
    
    async def _compile_javascript(self, source_code: str) -> Dict[str, Any]:
        """Compilar contrato JavaScript"""
        bytecode = f"0x{hashlib.sha256(source_code.encode()).hexdigest()[:64]}"
        
        abi = [
            {
                "name": "getValue",
                "outputs": [{"type": "number", "name": ""}],
                "inputs": [],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "name": "setValue",
                "outputs": [],
                "inputs": [{"type": "number", "name": "_value"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        return {
            "bytecode": bytecode,
            "abi": abi,
            "compiler_version": "18.17.0",
            "optimization": True,
            "gas_estimate": 170000
        }
    
    async def _compile_python(self, source_code: str) -> Dict[str, Any]:
        """Compilar contrato Python"""
        bytecode = f"0x{hashlib.sha256(source_code.encode()).hexdigest()[:64]}"
        
        abi = [
            {
                "name": "get_value",
                "outputs": [{"type": "int", "name": ""}],
                "inputs": [],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "name": "set_value",
                "outputs": [],
                "inputs": [{"type": "int", "name": "_value"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        return {
            "bytecode": bytecode,
            "abi": abi,
            "compiler_version": "3.11.0",
            "optimization": True,
            "gas_estimate": 190000
        }
    
    async def _compile_csharp(self, source_code: str) -> Dict[str, Any]:
        """Compilar contrato C#"""
        bytecode = f"0x{hashlib.sha256(source_code.encode()).hexdigest()[:64]}"
        
        abi = [
            {
                "name": "GetValue",
                "outputs": [{"type": "ulong", "name": ""}],
                "inputs": [],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "name": "SetValue",
                "outputs": [],
                "inputs": [{"type": "ulong", "name": "_value"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        return {
            "bytecode": bytecode,
            "abi": "abi",
            "compiler_version": "8.0.0",
            "optimization": True,
            "gas_estimate": 175000
        }
    
    async def _compile_java(self, source_code: str) -> Dict[str, Any]:
        """Compilar contrato Java"""
        bytecode = f"0x{hashlib.sha256(source_code.encode()).hexdigest()[:64]}"
        
        abi = [
            {
                "name": "getValue",
                "outputs": [{"type": "long", "name": ""}],
                "inputs": [],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "name": "setValue",
                "outputs": [],
                "inputs": [{"type": "long", "name": "_value"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        return {
            "bytecode": bytecode,
            "abi": abi,
            "compiler_version": "17.0.0",
            "optimization": True,
            "gas_estimate": 185000
        }


class SmartContractDeployer:
    """Desplegador de contratos inteligentes"""
    
    def __init__(self):
        self.networks = {
            "ethereum": {"chain_id": 1, "rpc_url": "https://mainnet.infura.io/v3/"},
            "polygon": {"chain_id": 137, "rpc_url": "https://polygon-rpc.com/"},
            "bsc": {"chain_id": 56, "rpc_url": "https://bsc-dataseed.binance.org/"},
            "avalanche": {"chain_id": 43114, "rpc_url": "https://api.avax.network/ext/bc/C/rpc"},
            "fantom": {"chain_id": 250, "rpc_url": "https://rpc.ftm.tools/"},
            "arbitrum": {"chain_id": 42161, "rpc_url": "https://arb1.arbitrum.io/rpc"},
            "optimism": {"chain_id": 10, "rpc_url": "https://mainnet.optimism.io"},
            "base": {"chain_id": 8453, "rpc_url": "https://mainnet.base.org"}
        }
    
    async def deploy_contract(self, contract: SmartContract, network: str, 
                            private_key: str, constructor_args: List[Any] = None) -> Dict[str, Any]:
        """Desplegar contrato inteligente"""
        try:
            if network not in self.networks:
                raise ValueError(f"Unsupported network: {network}")
            
            network_config = self.networks[network]
            
            # Simular despliegue
            address = f"0x{hashlib.sha256(f'{contract.id}{network}{time.time()}'.encode()).hexdigest()[:40]}"
            transaction_hash = f"0x{hashlib.sha256(f'{contract.id}{address}{time.time()}'.encode()).hexdigest()[:64]}"
            block_number = random.randint(18000000, 19000000)
            gas_used = random.randint(100000, 500000)
            
            return {
                "address": address,
                "transaction_hash": transaction_hash,
                "block_number": block_number,
                "gas_used": gas_used,
                "gas_price": 20000000000,  # 20 gwei
                "network": network,
                "chain_id": network_config["chain_id"],
                "deployed_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error deploying contract: {e}")
            raise


class SmartContractExecutor:
    """Ejecutor de contratos inteligentes"""
    
    def __init__(self):
        self.execution_queue = queue.Queue()
        self.is_running = False
        self._execution_thread = None
    
    async def start(self):
        """Iniciar ejecutor de contratos"""
        self.is_running = True
        self._execution_thread = threading.Thread(target=self._execution_worker)
        self._execution_thread.start()
        logger.info("Smart contract executor started")
    
    async def stop(self):
        """Detener ejecutor de contratos"""
        self.is_running = False
        if self._execution_thread:
            self._execution_thread.join(timeout=5)
        logger.info("Smart contract executor stopped")
    
    def _execution_worker(self):
        """Worker para ejecución de contratos"""
        while self.is_running:
            try:
                execution_id = self.execution_queue.get(timeout=1)
                if execution_id:
                    asyncio.run(self._execute_contract(execution_id))
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in smart contract execution worker: {e}")
    
    async def execute_contract_function(self, execution: ContractExecution) -> str:
        """Ejecutar función de contrato inteligente"""
        execution_id = execution.id
        self.execution_queue.put(execution_id)
        return execution_id
    
    async def _execute_contract(self, execution_id: str):
        """Ejecutar contrato internamente"""
        try:
            # Simular ejecución de contrato
            await asyncio.sleep(0.1)
            
            # Simular resultado
            result = {
                "return_value": random.randint(1, 1000),
                "gas_used": random.randint(21000, 100000),
                "status": "success"
            }
            
            logger.info(f"Contract execution {execution_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error executing contract {execution_id}: {e}")


class SmartContractsEngine:
    """Motor principal de contratos inteligentes"""
    
    def __init__(self):
        self.contracts: Dict[str, SmartContract] = {}
        self.executions: Dict[str, ContractExecution] = {}
        self.compiler = SmartContractCompiler()
        self.deployer = SmartContractDeployer()
        self.executor = SmartContractExecutor()
        self.is_running = False
        self._lock = threading.Lock()
    
    async def start(self):
        """Iniciar motor de contratos inteligentes"""
        try:
            self.is_running = True
            await self.executor.start()
            logger.info("Smart contracts engine started")
        except Exception as e:
            logger.error(f"Error starting smart contracts engine: {e}")
            raise
    
    async def stop(self):
        """Detener motor de contratos inteligentes"""
        try:
            self.is_running = False
            await self.executor.stop()
            logger.info("Smart contracts engine stopped")
        except Exception as e:
            logger.error(f"Error stopping smart contracts engine: {e}")
    
    async def create_smart_contract(self, contract_info: Dict[str, Any]) -> str:
        """Crear contrato inteligente"""
        contract_id = f"contract_{uuid.uuid4().hex[:8]}"
        
        # Compilar contrato
        compilation_result = await self.compiler.compile_contract(
            contract_info["source_code"],
            ContractLanguage(contract_info["language"])
        )
        
        contract = SmartContract(
            id=contract_id,
            name=contract_info["name"],
            description=contract_info.get("description", ""),
            contract_type=ContractType(contract_info["contract_type"]),
            language=ContractLanguage(contract_info["language"]),
            status=ContractStatus.DRAFT,
            source_code=contract_info["source_code"],
            bytecode=compilation_result["bytecode"],
            abi=compilation_result["abi"],
            address=None,
            deployer=None,
            gas_used=0,
            gas_price=0,
            transaction_hash=None,
            block_number=None,
            created_at=time.time(),
            deployed_at=None,
            last_modified=time.time(),
            metadata=contract_info.get("metadata", {})
        )
        
        async with self._lock:
            self.contracts[contract_id] = contract
        
        logger.info(f"Smart contract created: {contract_id} ({contract.name})")
        return contract_id
    
    async def deploy_smart_contract(self, contract_id: str, network: str, 
                                  private_key: str, constructor_args: List[Any] = None) -> Dict[str, Any]:
        """Desplegar contrato inteligente"""
        if contract_id not in self.contracts:
            raise ValueError(f"Smart contract {contract_id} not found")
        
        contract = self.contracts[contract_id]
        
        # Desplegar contrato
        deployment_result = await self.deployer.deploy_contract(
            contract, network, private_key, constructor_args
        )
        
        # Actualizar contrato
        contract.status = ContractStatus.DEPLOYED
        contract.address = deployment_result["address"]
        contract.deployer = "0x" + hashlib.sha256(private_key.encode()).hexdigest()[:40]
        contract.gas_used = deployment_result["gas_used"]
        contract.gas_price = deployment_result["gas_price"]
        contract.transaction_hash = deployment_result["transaction_hash"]
        contract.block_number = deployment_result["block_number"]
        contract.deployed_at = deployment_result["deployed_at"]
        contract.last_modified = time.time()
        
        return deployment_result
    
    async def execute_contract_function(self, contract_id: str, function_name: str, 
                                      parameters: Dict[str, Any], gas_limit: int = 100000) -> str:
        """Ejecutar función de contrato inteligente"""
        if contract_id not in self.contracts:
            raise ValueError(f"Smart contract {contract_id} not found")
        
        contract = self.contracts[contract_id]
        if contract.status != ContractStatus.DEPLOYED:
            raise ValueError(f"Smart contract {contract_id} is not deployed")
        
        execution_id = f"execution_{uuid.uuid4().hex[:8]}"
        
        execution = ContractExecution(
            id=execution_id,
            contract_id=contract_id,
            function_name=function_name,
            parameters=parameters,
            gas_limit=gas_limit,
            gas_used=0,
            status="pending",
            result=None,
            error_message=None,
            transaction_hash=None,
            block_number=None,
            created_at=time.time(),
            executed_at=None,
            execution_time=0.0,
            metadata={}
        )
        
        async with self._lock:
            self.executions[execution_id] = execution
        
        # Ejecutar función
        await self.executor.execute_contract_function(execution)
        
        return execution_id
    
    async def get_contract_info(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información del contrato inteligente"""
        if contract_id not in self.contracts:
            return None
        
        contract = self.contracts[contract_id]
        return {
            "id": contract.id,
            "name": contract.name,
            "description": contract.description,
            "contract_type": contract.contract_type.value,
            "language": contract.language.value,
            "status": contract.status.value,
            "address": contract.address,
            "deployer": contract.deployer,
            "gas_used": contract.gas_used,
            "transaction_hash": contract.transaction_hash,
            "block_number": contract.block_number,
            "created_at": contract.created_at,
            "deployed_at": contract.deployed_at,
            "last_modified": contract.last_modified
        }
    
    async def get_execution_result(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtener resultado de ejecución"""
        if execution_id not in self.executions:
            return None
        
        execution = self.executions[execution_id]
        return {
            "id": execution.id,
            "contract_id": execution.contract_id,
            "function_name": execution.function_name,
            "parameters": execution.parameters,
            "gas_limit": execution.gas_limit,
            "gas_used": execution.gas_used,
            "status": execution.status,
            "result": execution.result,
            "error_message": execution.error_message,
            "transaction_hash": execution.transaction_hash,
            "block_number": execution.block_number,
            "created_at": execution.created_at,
            "executed_at": execution.executed_at,
            "execution_time": execution.execution_time
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "is_running": self.is_running,
            "contracts": {
                "total": len(self.contracts),
                "by_type": {
                    contract_type.value: sum(1 for c in self.contracts.values() if c.contract_type == contract_type)
                    for contract_type in ContractType
                },
                "by_status": {
                    status.value: sum(1 for c in self.contracts.values() if c.status == status)
                    for status in ContractStatus
                },
                "by_language": {
                    language.value: sum(1 for c in self.contracts.values() if c.language == language)
                    for language in ContractLanguage
                }
            },
            "executions": {
                "total": len(self.executions),
                "by_status": {
                    "pending": sum(1 for e in self.executions.values() if e.status == "pending"),
                    "completed": sum(1 for e in self.executions.values() if e.status == "completed"),
                    "failed": sum(1 for e in self.executions.values() if e.status == "failed")
                }
            }
        }


# Instancia global del motor de contratos inteligentes
smart_contracts_engine = SmartContractsEngine()


# Router para endpoints del motor de contratos inteligentes
smart_contracts_router = APIRouter()


@smart_contracts_router.post("/smart-contracts")
async def create_smart_contract_endpoint(contract_data: dict):
    """Crear contrato inteligente"""
    try:
        contract_id = await smart_contracts_engine.create_smart_contract(contract_data)
        
        return {
            "message": "Smart contract created successfully",
            "contract_id": contract_id,
            "name": contract_data["name"],
            "contract_type": contract_data["contract_type"],
            "language": contract_data["language"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid contract type or language: {e}")
    except Exception as e:
        logger.error(f"Error creating smart contract: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create smart contract: {str(e)}")


@smart_contracts_router.get("/smart-contracts")
async def get_smart_contracts_endpoint():
    """Obtener contratos inteligentes"""
    try:
        contracts = smart_contracts_engine.contracts
        return {
            "contracts": [
                {
                    "id": contract.id,
                    "name": contract.name,
                    "description": contract.description,
                    "contract_type": contract.contract_type.value,
                    "language": contract.language.value,
                    "status": contract.status.value,
                    "address": contract.address,
                    "created_at": contract.created_at,
                    "deployed_at": contract.deployed_at
                }
                for contract in contracts.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting smart contracts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get smart contracts: {str(e)}")


@smart_contracts_router.get("/smart-contracts/{contract_id}")
async def get_smart_contract_endpoint(contract_id: str):
    """Obtener contrato inteligente específico"""
    try:
        info = await smart_contracts_engine.get_contract_info(contract_id)
        
        if info:
            return info
        else:
            raise HTTPException(status_code=404, detail="Smart contract not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting smart contract: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get smart contract: {str(e)}")


@smart_contracts_router.post("/smart-contracts/{contract_id}/deploy")
async def deploy_smart_contract_endpoint(contract_id: str, deployment_data: dict):
    """Desplegar contrato inteligente"""
    try:
        network = deployment_data["network"]
        private_key = deployment_data["private_key"]
        constructor_args = deployment_data.get("constructor_args", [])
        
        result = await smart_contracts_engine.deploy_smart_contract(
            contract_id, network, private_key, constructor_args
        )
        
        return {
            "message": "Smart contract deployed successfully",
            "contract_id": contract_id,
            "deployment_result": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deploying smart contract: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to deploy smart contract: {str(e)}")


@smart_contracts_router.post("/smart-contracts/{contract_id}/execute")
async def execute_contract_function_endpoint(contract_id: str, execution_data: dict):
    """Ejecutar función de contrato inteligente"""
    try:
        function_name = execution_data["function_name"]
        parameters = execution_data.get("parameters", {})
        gas_limit = execution_data.get("gas_limit", 100000)
        
        execution_id = await smart_contracts_engine.execute_contract_function(
            contract_id, function_name, parameters, gas_limit
        )
        
        return {
            "message": "Contract function execution started successfully",
            "execution_id": execution_id,
            "contract_id": contract_id,
            "function_name": function_name
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing contract function: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute contract function: {str(e)}")


@smart_contracts_router.get("/smart-contracts/executions/{execution_id}")
async def get_execution_result_endpoint(execution_id: str):
    """Obtener resultado de ejecución"""
    try:
        result = await smart_contracts_engine.get_execution_result(execution_id)
        
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="Contract execution not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get execution result: {str(e)}")


@smart_contracts_router.get("/smart-contracts/stats")
async def get_smart_contracts_stats_endpoint():
    """Obtener estadísticas del motor de contratos inteligentes"""
    try:
        stats = await smart_contracts_engine.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting smart contracts stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get smart contracts stats: {str(e)}")


# Funciones de utilidad para integración
async def start_smart_contracts_engine():
    """Iniciar motor de contratos inteligentes"""
    await smart_contracts_engine.start()


async def stop_smart_contracts_engine():
    """Detener motor de contratos inteligentes"""
    await smart_contracts_engine.stop()


async def create_smart_contract(contract_info: Dict[str, Any]) -> str:
    """Crear contrato inteligente"""
    return await smart_contracts_engine.create_smart_contract(contract_info)


async def deploy_smart_contract(contract_id: str, network: str, private_key: str, 
                              constructor_args: List[Any] = None) -> Dict[str, Any]:
    """Desplegar contrato inteligente"""
    return await smart_contracts_engine.deploy_smart_contract(
        contract_id, network, private_key, constructor_args
    )


async def get_smart_contracts_engine_stats() -> Dict[str, Any]:
    """Obtener estadísticas del motor de contratos inteligentes"""
    return await smart_contracts_engine.get_system_stats()


logger.info("Smart contracts engine module loaded successfully")

