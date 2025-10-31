"""
Biocomputing API Endpoints
==========================

API endpoints for biocomputing service.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.biocomputing_service import (
    BiocomputingService,
    DNAStrand,
    Protein,
    MolecularSimulation,
    GeneticAlgorithm,
    CellularAutomaton,
    BiocomputingType,
    MoleculeType,
    ProteinStructure,
    DNAOperation
)

logger = logging.getLogger(__name__)

# Create router
biocomputing_router = APIRouter(prefix="/biocomputing", tags=["Biocomputing"])

# Pydantic models for request/response
class DNAStrandRequest(BaseModel):
    sequence: str
    metadata: Dict[str, Any] = {}

class DNAOperationRequest(BaseModel):
    operation: DNAOperation
    parameters: Dict[str, Any] = {}

class ProteinRequest(BaseModel):
    name: str
    sequence: str
    metadata: Dict[str, Any] = {}

class MolecularSimulationRequest(BaseModel):
    name: str
    simulation_type: str
    molecules: List[str]
    parameters: Dict[str, Any] = {}
    timesteps: int = 1000
    temperature: float = 300.0
    pressure: float = 1.0
    metadata: Dict[str, Any] = {}

class GeneticAlgorithmRequest(BaseModel):
    name: str
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    selection_method: str = "tournament"
    fitness_function: str = "sphere"
    metadata: Dict[str, Any] = {}

class CellularAutomatonRequest(BaseModel):
    name: str
    grid_size: Tuple[int, int]
    rules: Dict[str, Any] = {}
    initial_state: List[List[int]] = []
    metadata: Dict[str, Any] = {}

class DNAStrandResponse(BaseModel):
    strand_id: str
    sequence: str
    length: int
    gc_content: float
    melting_temperature: float
    secondary_structure: str
    created_at: datetime
    metadata: Dict[str, Any]

class ProteinResponse(BaseModel):
    protein_id: str
    name: str
    sequence: str
    length: int
    molecular_weight: float
    isoelectric_point: float
    structure_level: str
    secondary_structure: str
    tertiary_structure: Dict[str, Any]
    created_at: datetime
    metadata: Dict[str, Any]

class MolecularSimulationResponse(BaseModel):
    simulation_id: str
    name: str
    simulation_type: str
    molecules: List[str]
    parameters: Dict[str, Any]
    timesteps: int
    temperature: float
    pressure: float
    result: Optional[Any]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class GeneticAlgorithmResponse(BaseModel):
    ga_id: str
    name: str
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    selection_method: str
    fitness_function: str
    individuals: List[Dict[str, Any]]
    best_individual: Optional[Dict[str, Any]]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class CellularAutomatonResponse(BaseModel):
    ca_id: str
    name: str
    grid_size: Tuple[int, int]
    rules: Dict[str, Any]
    initial_state: List[List[int]]
    current_state: List[List[int]]
    generations: int
    status: str
    created_at: datetime
    last_update: datetime
    metadata: Dict[str, Any]

class ServiceStatusResponse(BaseModel):
    service_status: str
    total_dna_strands: int
    total_proteins: int
    total_simulations: int
    total_genetic_algorithms: int
    total_cellular_automata: int
    active_simulations: int
    active_genetic_algorithms: int
    active_cellular_automata: int
    biological_databases: int
    dna_computing_enabled: bool
    protein_folding_enabled: bool
    molecular_simulation_enabled: bool
    genetic_algorithm_enabled: bool
    cellular_automata_enabled: bool
    timestamp: str

# Dependency to get biocomputing service
async def get_biocomputing_service() -> BiocomputingService:
    """Get biocomputing service instance."""
    # This would be injected from your dependency injection system
    # For now, we'll create a mock instance
    from ..main import get_biocomputing_service
    return await get_biocomputing_service()

@biocomputing_router.post("/dna/strands", response_model=Dict[str, str])
async def create_dna_strand(
    request: DNAStrandRequest,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Create a DNA strand."""
    try:
        strand_id = await biocomputing_service.create_dna_strand(
            sequence=request.sequence,
            metadata=request.metadata
        )
        
        return {"strand_id": strand_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create DNA strand: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.post("/dna/strands/{strand_id}/operations", response_model=Dict[str, Any])
async def perform_dna_operation(
    strand_id: str,
    request: DNAOperationRequest,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Perform DNA operation."""
    try:
        result = await biocomputing_service.perform_dna_operation(
            strand_id=strand_id,
            operation=request.operation,
            parameters=request.parameters
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to perform DNA operation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/dna/strands/{strand_id}", response_model=DNAStrandResponse)
async def get_dna_strand(
    strand_id: str,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Get a DNA strand."""
    try:
        if strand_id not in biocomputing_service.dna_strands:
            raise HTTPException(status_code=404, detail="DNA strand not found")
            
        strand = biocomputing_service.dna_strands[strand_id]
        
        return DNAStrandResponse(
            strand_id=strand.strand_id,
            sequence=strand.sequence,
            length=strand.length,
            gc_content=strand.gc_content,
            melting_temperature=strand.melting_temperature,
            secondary_structure=strand.secondary_structure,
            created_at=strand.created_at,
            metadata=strand.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get DNA strand: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/dna/strands", response_model=List[DNAStrandResponse])
async def list_dna_strands(
    limit: int = 100,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """List DNA strands."""
    try:
        strands = list(biocomputing_service.dna_strands.values())
        
        return [
            DNAStrandResponse(
                strand_id=strand.strand_id,
                sequence=strand.sequence,
                length=strand.length,
                gc_content=strand.gc_content,
                melting_temperature=strand.melting_temperature,
                secondary_structure=strand.secondary_structure,
                created_at=strand.created_at,
                metadata=strand.metadata
            )
            for strand in strands[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list DNA strands: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.post("/proteins", response_model=Dict[str, str])
async def create_protein(
    request: ProteinRequest,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Create a protein."""
    try:
        protein_id = await biocomputing_service.create_protein(
            name=request.name,
            sequence=request.sequence,
            metadata=request.metadata
        )
        
        return {"protein_id": protein_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create protein: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.post("/proteins/{protein_id}/predict-structure", response_model=Dict[str, Any])
async def predict_protein_structure(
    protein_id: str,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Predict protein structure."""
    try:
        result = await biocomputing_service.predict_protein_structure(protein_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to predict protein structure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/proteins/{protein_id}", response_model=ProteinResponse)
async def get_protein(
    protein_id: str,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Get a protein."""
    try:
        if protein_id not in biocomputing_service.proteins:
            raise HTTPException(status_code=404, detail="Protein not found")
            
        protein = biocomputing_service.proteins[protein_id]
        
        return ProteinResponse(
            protein_id=protein.protein_id,
            name=protein.name,
            sequence=protein.sequence,
            length=protein.length,
            molecular_weight=protein.molecular_weight,
            isoelectric_point=protein.isoelectric_point,
            structure_level=protein.structure_level.value,
            secondary_structure=protein.secondary_structure,
            tertiary_structure=protein.tertiary_structure,
            created_at=protein.created_at,
            metadata=protein.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get protein: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/proteins", response_model=List[ProteinResponse])
async def list_proteins(
    limit: int = 100,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """List proteins."""
    try:
        proteins = list(biocomputing_service.proteins.values())
        
        return [
            ProteinResponse(
                protein_id=protein.protein_id,
                name=protein.name,
                sequence=protein.sequence,
                length=protein.length,
                molecular_weight=protein.molecular_weight,
                isoelectric_point=protein.isoelectric_point,
                structure_level=protein.structure_level.value,
                secondary_structure=protein.secondary_structure,
                tertiary_structure=protein.tertiary_structure,
                created_at=protein.created_at,
                metadata=protein.metadata
            )
            for protein in proteins[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list proteins: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.post("/simulations", response_model=Dict[str, str])
async def run_molecular_simulation(
    request: MolecularSimulationRequest,
    background_tasks: BackgroundTasks,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Run molecular simulation."""
    try:
        simulation = MolecularSimulation(
            simulation_id="",
            name=request.name,
            simulation_type=request.simulation_type,
            molecules=request.molecules,
            parameters=request.parameters,
            timesteps=0,
            temperature=request.temperature,
            pressure=request.pressure,
            result=None,
            status="pending",
            created_at=datetime.utcnow(),
            completed_at=None,
            metadata=request.metadata
        )
        
        simulation_id = await biocomputing_service.run_molecular_simulation(simulation)
        
        return {"simulation_id": simulation_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to run molecular simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/simulations/{simulation_id}", response_model=MolecularSimulationResponse)
async def get_molecular_simulation(
    simulation_id: str,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Get molecular simulation result."""
    try:
        if simulation_id not in biocomputing_service.molecular_simulations:
            raise HTTPException(status_code=404, detail="Molecular simulation not found")
            
        simulation = biocomputing_service.molecular_simulations[simulation_id]
        
        return MolecularSimulationResponse(
            simulation_id=simulation.simulation_id,
            name=simulation.name,
            simulation_type=simulation.simulation_type,
            molecules=simulation.molecules,
            parameters=simulation.parameters,
            timesteps=simulation.timesteps,
            temperature=simulation.temperature,
            pressure=simulation.pressure,
            result=simulation.result,
            status=simulation.status,
            created_at=simulation.created_at,
            completed_at=simulation.completed_at,
            metadata=simulation.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get molecular simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/simulations", response_model=List[MolecularSimulationResponse])
async def list_molecular_simulations(
    status: Optional[str] = None,
    limit: int = 100,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """List molecular simulations."""
    try:
        simulations = list(biocomputing_service.molecular_simulations.values())
        
        if status:
            simulations = [sim for sim in simulations if sim.status == status]
            
        return [
            MolecularSimulationResponse(
                simulation_id=simulation.simulation_id,
                name=simulation.name,
                simulation_type=simulation.simulation_type,
                molecules=simulation.molecules,
                parameters=simulation.parameters,
                timesteps=simulation.timesteps,
                temperature=simulation.temperature,
                pressure=simulation.pressure,
                result=simulation.result,
                status=simulation.status,
                created_at=simulation.created_at,
                completed_at=simulation.completed_at,
                metadata=simulation.metadata
            )
            for simulation in simulations[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list molecular simulations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.post("/genetic-algorithms", response_model=Dict[str, str])
async def create_genetic_algorithm(
    request: GeneticAlgorithmRequest,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Create genetic algorithm."""
    try:
        ga = GeneticAlgorithm(
            ga_id="",
            name=request.name,
            population_size=request.population_size,
            generations=request.generations,
            mutation_rate=request.mutation_rate,
            crossover_rate=request.crossover_rate,
            selection_method=request.selection_method,
            fitness_function=request.fitness_function,
            individuals=[],
            best_individual=None,
            status="created",
            created_at=datetime.utcnow(),
            completed_at=None,
            metadata=request.metadata
        )
        
        ga_id = await biocomputing_service.create_genetic_algorithm(ga)
        
        return {"ga_id": ga_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create genetic algorithm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.post("/genetic-algorithms/{ga_id}/run", response_model=Dict[str, str])
async def run_genetic_algorithm(
    ga_id: str,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Run genetic algorithm."""
    try:
        result_ga_id = await biocomputing_service.run_genetic_algorithm(ga_id)
        
        return {"ga_id": result_ga_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to run genetic algorithm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/genetic-algorithms/{ga_id}", response_model=GeneticAlgorithmResponse)
async def get_genetic_algorithm(
    ga_id: str,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Get genetic algorithm."""
    try:
        if ga_id not in biocomputing_service.genetic_algorithms:
            raise HTTPException(status_code=404, detail="Genetic algorithm not found")
            
        ga = biocomputing_service.genetic_algorithms[ga_id]
        
        return GeneticAlgorithmResponse(
            ga_id=ga.ga_id,
            name=ga.name,
            population_size=ga.population_size,
            generations=ga.generations,
            mutation_rate=ga.mutation_rate,
            crossover_rate=ga.crossover_rate,
            selection_method=ga.selection_method,
            fitness_function=ga.fitness_function,
            individuals=ga.individuals,
            best_individual=ga.best_individual,
            status=ga.status,
            created_at=ga.created_at,
            completed_at=ga.completed_at,
            metadata=ga.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get genetic algorithm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/genetic-algorithms", response_model=List[GeneticAlgorithmResponse])
async def list_genetic_algorithms(
    status: Optional[str] = None,
    limit: int = 100,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """List genetic algorithms."""
    try:
        genetic_algorithms = list(biocomputing_service.genetic_algorithms.values())
        
        if status:
            genetic_algorithms = [ga for ga in genetic_algorithms if ga.status == status]
            
        return [
            GeneticAlgorithmResponse(
                ga_id=ga.ga_id,
                name=ga.name,
                population_size=ga.population_size,
                generations=ga.generations,
                mutation_rate=ga.mutation_rate,
                crossover_rate=ga.crossover_rate,
                selection_method=ga.selection_method,
                fitness_function=ga.fitness_function,
                individuals=ga.individuals,
                best_individual=ga.best_individual,
                status=ga.status,
                created_at=ga.created_at,
                completed_at=ga.completed_at,
                metadata=ga.metadata
            )
            for ga in genetic_algorithms[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list genetic algorithms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.post("/cellular-automata", response_model=Dict[str, str])
async def create_cellular_automaton(
    request: CellularAutomatonRequest,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Create cellular automaton."""
    try:
        ca = CellularAutomaton(
            ca_id="",
            name=request.name,
            grid_size=request.grid_size,
            rules=request.rules,
            initial_state=request.initial_state or [[0 for _ in range(request.grid_size[1])] for _ in range(request.grid_size[0])],
            current_state=request.initial_state or [[0 for _ in range(request.grid_size[1])] for _ in range(request.grid_size[0])],
            generations=0,
            status="created",
            created_at=datetime.utcnow(),
            last_update=datetime.utcnow(),
            metadata=request.metadata
        )
        
        ca_id = await biocomputing_service.create_cellular_automaton(ca)
        
        return {"ca_id": ca_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create cellular automaton: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.post("/cellular-automata/{ca_id}/run", response_model=Dict[str, str])
async def run_cellular_automaton(
    ca_id: str,
    generations: int = 100,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Run cellular automaton."""
    try:
        result_ca_id = await biocomputing_service.run_cellular_automaton(ca_id, generations)
        
        return {"ca_id": result_ca_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to run cellular automaton: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/cellular-automata/{ca_id}", response_model=CellularAutomatonResponse)
async def get_cellular_automaton(
    ca_id: str,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Get cellular automaton."""
    try:
        if ca_id not in biocomputing_service.cellular_automata:
            raise HTTPException(status_code=404, detail="Cellular automaton not found")
            
        ca = biocomputing_service.cellular_automata[ca_id]
        
        return CellularAutomatonResponse(
            ca_id=ca.ca_id,
            name=ca.name,
            grid_size=ca.grid_size,
            rules=ca.rules,
            initial_state=ca.initial_state,
            current_state=ca.current_state,
            generations=ca.generations,
            status=ca.status,
            created_at=ca.created_at,
            last_update=ca.last_update,
            metadata=ca.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cellular automaton: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/cellular-automata", response_model=List[CellularAutomatonResponse])
async def list_cellular_automata(
    status: Optional[str] = None,
    limit: int = 100,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """List cellular automata."""
    try:
        cellular_automata = list(biocomputing_service.cellular_automata.values())
        
        if status:
            cellular_automata = [ca for ca in cellular_automata if ca.status == status]
            
        return [
            CellularAutomatonResponse(
                ca_id=ca.ca_id,
                name=ca.name,
                grid_size=ca.grid_size,
                rules=ca.rules,
                initial_state=ca.initial_state,
                current_state=ca.current_state,
                generations=ca.generations,
                status=ca.status,
                created_at=ca.created_at,
                last_update=ca.last_update,
                metadata=ca.metadata
            )
            for ca in cellular_automata[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list cellular automata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/status", response_model=ServiceStatusResponse)
async def get_service_status(
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Get biocomputing service status."""
    try:
        status = await biocomputing_service.get_service_status()
        
        return ServiceStatusResponse(
            service_status=status["service_status"],
            total_dna_strands=status["total_dna_strands"],
            total_proteins=status["total_proteins"],
            total_simulations=status["total_simulations"],
            total_genetic_algorithms=status["total_genetic_algorithms"],
            total_cellular_automata=status["total_cellular_automata"],
            active_simulations=status["active_simulations"],
            active_genetic_algorithms=status["active_genetic_algorithms"],
            active_cellular_automata=status["active_cellular_automata"],
            biological_databases=status["biological_databases"],
            dna_computing_enabled=status["dna_computing_enabled"],
            protein_folding_enabled=status["protein_folding_enabled"],
            molecular_simulation_enabled=status["molecular_simulation_enabled"],
            genetic_algorithm_enabled=status["genetic_algorithm_enabled"],
            cellular_automata_enabled=status["cellular_automata_enabled"],
            timestamp=status["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get service status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/databases", response_model=Dict[str, Any])
async def get_biological_databases(
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Get biological databases."""
    try:
        return biocomputing_service.biological_databases
        
    except Exception as e:
        logger.error(f"Failed to get biological databases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.get("/dna-operations", response_model=List[str])
async def get_dna_operations():
    """Get available DNA operations."""
    return [operation.value for operation in DNAOperation]

@biocomputing_router.get("/molecule-types", response_model=List[str])
async def get_molecule_types():
    """Get available molecule types."""
    return [molecule_type.value for molecule_type in MoleculeType]

@biocomputing_router.get("/protein-structures", response_model=List[str])
async def get_protein_structures():
    """Get available protein structure levels."""
    return [structure.value for structure in ProteinStructure]

@biocomputing_router.get("/biocomputing-types", response_model=List[str])
async def get_biocomputing_types():
    """Get available biocomputing types."""
    return [biocomputing_type.value for biocomputing_type in BiocomputingType]

@biocomputing_router.delete("/dna/strands/{strand_id}")
async def delete_dna_strand(
    strand_id: str,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Delete a DNA strand."""
    try:
        if strand_id not in biocomputing_service.dna_strands:
            raise HTTPException(status_code=404, detail="DNA strand not found")
            
        del biocomputing_service.dna_strands[strand_id]
        
        return {"status": "deleted", "strand_id": strand_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete DNA strand: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@biocomputing_router.delete("/proteins/{protein_id}")
async def delete_protein(
    protein_id: str,
    biocomputing_service: BiocomputingService = Depends(get_biocomputing_service)
):
    """Delete a protein."""
    try:
        if protein_id not in biocomputing_service.proteins:
            raise HTTPException(status_code=404, detail="Protein not found")
            
        del biocomputing_service.proteins[protein_id]
        
        return {"status": "deleted", "protein_id": protein_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete protein: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

























