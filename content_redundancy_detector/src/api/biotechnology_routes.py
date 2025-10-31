"""
Biotechnology API Routes - Advanced biotechnology and synthetic biology endpoints
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.biotechnology_engine import (
    get_biotechnology_engine, 
    BiotechnologyConfig,
    Organism,
    Protein,
    Gene,
    Cell,
    Experiment
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/biotechnology", tags=["Biotechnology"])


# Pydantic models
class OrganismCreate(BaseModel):
    name: str = Field(..., description="Organism name")
    species: str = Field(..., description="Species name")
    strain: str = Field(..., description="Strain name")
    organism_type: str = Field(..., description="Organism type")
    genome_size: int = Field(..., description="Genome size in base pairs")
    chromosome_count: int = Field(..., description="Number of chromosomes")
    gene_count: int = Field(..., description="Number of genes")
    protein_count: int = Field(..., description="Number of proteins")
    metabolic_pathways: List[str] = Field(default=[], description="Metabolic pathways")
    genetic_modifications: List[Dict[str, Any]] = Field(default=[], description="Genetic modifications")
    phenotype: Dict[str, Any] = Field(default={}, description="Phenotype")
    genotype: Dict[str, Any] = Field(default={}, description="Genotype")
    growth_rate: float = Field(..., description="Growth rate")
    optimal_temperature: float = Field(..., description="Optimal temperature")
    optimal_ph: float = Field(..., description="Optimal pH")
    optimal_oxygen: float = Field(..., description="Optimal oxygen percentage")
    nutritional_requirements: List[str] = Field(default=[], description="Nutritional requirements")
    stress_resistance: Dict[str, float] = Field(default={}, description="Stress resistance")
    bioproduction_capabilities: List[str] = Field(default=[], description="Bioproduction capabilities")
    safety_level: str = Field(..., description="Safety level")
    containment_requirements: List[str] = Field(default=[], description="Containment requirements")
    regulatory_status: str = Field(..., description="Regulatory status")
    intellectual_property: List[str] = Field(default=[], description="Intellectual property")
    commercial_applications: List[str] = Field(default=[], description="Commercial applications")
    research_applications: List[str] = Field(default=[], description="Research applications")


class ExperimentCreate(BaseModel):
    name: str = Field(..., description="Experiment name")
    experiment_type: str = Field(..., description="Experiment type")
    objective: str = Field(..., description="Experiment objective")
    hypothesis: str = Field(..., description="Experiment hypothesis")
    experimental_design: Dict[str, Any] = Field(default={}, description="Experimental design")
    materials: List[str] = Field(default=[], description="Materials")
    methods: List[str] = Field(default=[], description="Methods")
    protocols: List[str] = Field(default=[], description="Protocols")
    controls: List[str] = Field(default=[], description="Controls")
    variables: List[str] = Field(default=[], description="Variables")
    measurements: List[str] = Field(default=[], description="Measurements")
    data_collection: Dict[str, Any] = Field(default={}, description="Data collection")
    analysis_methods: List[str] = Field(default=[], description="Analysis methods")
    statistical_tests: List[str] = Field(default=[], description="Statistical tests")
    results: Dict[str, Any] = Field(default={}, description="Results")
    conclusions: str = Field(default="", description="Conclusions")
    limitations: List[str] = Field(default=[], description="Limitations")
    future_work: List[str] = Field(default=[], description="Future work")
    regulatory_implications: List[str] = Field(default=[], description="Regulatory implications")
    ethical_considerations: List[str] = Field(default=[], description="Ethical considerations")
    safety_considerations: List[str] = Field(default=[], description="Safety considerations")
    intellectual_property: List[str] = Field(default=[], description="Intellectual property")
    collaborators: List[str] = Field(default=[], description="Collaborators")
    funding_sources: List[str] = Field(default=[], description="Funding sources")


class GeneticCircuitDesign(BaseModel):
    name: str = Field(..., description="Circuit name")
    function: str = Field(..., description="Circuit function")
    components: List[str] = Field(default=[], description="Circuit components")
    connections: List[Dict[str, Any]] = Field(default=[], description="Circuit connections")
    input_signals: List[str] = Field(default=[], description="Input signals")
    output_signals: List[str] = Field(default=[], description="Output signals")
    regulatory_elements: List[str] = Field(default=[], description="Regulatory elements")
    promoters: List[str] = Field(default=[], description="Promoters")
    ribosome_binding_sites: List[str] = Field(default=[], description="Ribosome binding sites")
    coding_sequences: List[str] = Field(default=[], description="Coding sequences")
    terminators: List[str] = Field(default=[], description="Terminators")
    circuit_topology: str = Field(default="linear", description="Circuit topology")
    applications: List[str] = Field(default=[], description="Applications")
    safety_considerations: List[str] = Field(default=[], description="Safety considerations")
    regulatory_requirements: List[str] = Field(default=[], description="Regulatory requirements")
    intellectual_property: List[str] = Field(default=[], description="Intellectual property")


class ProteinDesign(BaseModel):
    name: str = Field(..., description="Protein name")
    function: str = Field(..., description="Protein function")
    target_properties: Dict[str, Any] = Field(default={}, description="Target properties")
    sequence: str = Field(default="", description="Protein sequence")
    applications: List[str] = Field(default=[], description="Applications")
    safety_profile: str = Field(default="safe", description="Safety profile")
    regulatory_status: str = Field(default="experimental", description="Regulatory status")
    intellectual_property: List[str] = Field(default=[], description="Intellectual property")


class CellEngineering(BaseModel):
    name: str = Field(..., description="Cell name")
    cell_type: str = Field(..., description="Cell type")
    species: str = Field(..., description="Species")
    tissue_origin: str = Field(..., description="Tissue origin")
    genetic_modifications: List[Dict[str, Any]] = Field(default=[], description="Genetic modifications")
    phenotype: Dict[str, Any] = Field(default={}, description="Phenotype")
    applications: List[str] = Field(default=[], description="Applications")
    regulatory_status: str = Field(default="experimental", description="Regulatory status")
    intellectual_property: List[str] = Field(default=[], description="Intellectual property")


class CircuitOptimization(BaseModel):
    optimization_goals: List[str] = Field(..., description="Optimization goals")


class CellDifferentiation(BaseModel):
    target_cell_type: str = Field(..., description="Target cell type")


class StructurePrediction(BaseModel):
    sequence: str = Field(..., description="Protein sequence")


# Dependency
async def get_biotech_engine():
    """Get biotechnology engine dependency"""
    engine = await get_biotechnology_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Biotechnology engine not available")
    return engine


# Organism management endpoints
@router.post("/organisms", response_model=Dict[str, Any])
async def create_organism(
    organism_data: OrganismCreate,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Create a new organism"""
    try:
        organism_dict = organism_data.dict()
        organism = await engine.create_organism(organism_dict)
        
        return {
            "organism_id": organism.organism_id,
            "timestamp": organism.timestamp.isoformat(),
            "name": organism.name,
            "species": organism.species,
            "strain": organism.strain,
            "organism_type": organism.organism_type,
            "genome_size": organism.genome_size,
            "chromosome_count": organism.chromosome_count,
            "gene_count": organism.gene_count,
            "protein_count": organism.protein_count,
            "growth_rate": organism.growth_rate,
            "optimal_temperature": organism.optimal_temperature,
            "optimal_ph": organism.optimal_ph,
            "optimal_oxygen": organism.optimal_oxygen,
            "safety_level": organism.safety_level,
            "regulatory_status": organism.regulatory_status,
            "status": organism.status,
            "message": "Organism created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating organism: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organisms", response_model=Dict[str, Any])
async def get_organisms(
    skip: int = 0,
    limit: int = 100,
    organism_type: Optional[str] = None,
    species: Optional[str] = None,
    status: Optional[str] = None,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Get organisms with filtering"""
    try:
        organisms = list(engine.organisms.values())
        
        # Apply filters
        if organism_type:
            organisms = [o for o in organisms if o.organism_type == organism_type]
        if species:
            organisms = [o for o in organisms if o.species == species]
        if status:
            organisms = [o for o in organisms if o.status == status]
        
        # Apply pagination
        total = len(organisms)
        organisms = organisms[skip:skip + limit]
        
        organism_list = []
        for organism in organisms:
            organism_list.append({
                "organism_id": organism.organism_id,
                "timestamp": organism.timestamp.isoformat(),
                "name": organism.name,
                "species": organism.species,
                "strain": organism.strain,
                "organism_type": organism.organism_type,
                "genome_size": organism.genome_size,
                "chromosome_count": organism.chromosome_count,
                "gene_count": organism.gene_count,
                "protein_count": organism.protein_count,
                "growth_rate": organism.growth_rate,
                "optimal_temperature": organism.optimal_temperature,
                "optimal_ph": organism.optimal_ph,
                "optimal_oxygen": organism.optimal_oxygen,
                "safety_level": organism.safety_level,
                "regulatory_status": organism.regulatory_status,
                "status": organism.status
            })
        
        return {
            "organisms": organism_list,
            "total": total,
            "skip": skip,
            "limit": limit,
            "message": f"Retrieved {len(organism_list)} organisms"
        }
        
    except Exception as e:
        logger.error(f"Error getting organisms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organisms/{organism_id}", response_model=Dict[str, Any])
async def get_organism(
    organism_id: str,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Get specific organism"""
    try:
        if organism_id not in engine.organisms:
            raise HTTPException(status_code=404, detail="Organism not found")
        
        organism = engine.organisms[organism_id]
        
        return {
            "organism_id": organism.organism_id,
            "timestamp": organism.timestamp.isoformat(),
            "name": organism.name,
            "species": organism.species,
            "strain": organism.strain,
            "organism_type": organism.organism_type,
            "genome_size": organism.genome_size,
            "chromosome_count": organism.chromosome_count,
            "gene_count": organism.gene_count,
            "protein_count": organism.protein_count,
            "metabolic_pathways": organism.metabolic_pathways,
            "genetic_modifications": organism.genetic_modifications,
            "phenotype": organism.phenotype,
            "genotype": organism.genotype,
            "growth_rate": organism.growth_rate,
            "optimal_temperature": organism.optimal_temperature,
            "optimal_ph": organism.optimal_ph,
            "optimal_oxygen": organism.optimal_oxygen,
            "nutritional_requirements": organism.nutritional_requirements,
            "stress_resistance": organism.stress_resistance,
            "bioproduction_capabilities": organism.bioproduction_capabilities,
            "safety_level": organism.safety_level,
            "containment_requirements": organism.containment_requirements,
            "regulatory_status": organism.regulatory_status,
            "intellectual_property": organism.intellectual_property,
            "commercial_applications": organism.commercial_applications,
            "research_applications": organism.research_applications,
            "status": organism.status,
            "message": "Organism retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting organism: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Experiment management endpoints
@router.post("/experiments", response_model=Dict[str, Any])
async def create_experiment(
    experiment_data: ExperimentCreate,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Create a new biotechnology experiment"""
    try:
        experiment_dict = experiment_data.dict()
        experiment = await engine.create_experiment(experiment_dict)
        
        return {
            "experiment_id": experiment.experiment_id,
            "timestamp": experiment.timestamp.isoformat(),
            "name": experiment.name,
            "experiment_type": experiment.experiment_type,
            "objective": experiment.objective,
            "hypothesis": experiment.hypothesis,
            "experimental_design": experiment.experimental_design,
            "materials": experiment.materials,
            "methods": experiment.methods,
            "protocols": experiment.protocols,
            "controls": experiment.controls,
            "variables": experiment.variables,
            "measurements": experiment.measurements,
            "data_collection": experiment.data_collection,
            "analysis_methods": experiment.analysis_methods,
            "statistical_tests": experiment.statistical_tests,
            "results": experiment.results,
            "conclusions": experiment.conclusions,
            "limitations": experiment.limitations,
            "future_work": experiment.future_work,
            "reproducibility": experiment.reproducibility,
            "significance": experiment.significance,
            "effect_size": experiment.effect_size,
            "confidence_interval": experiment.confidence_interval,
            "p_value": experiment.p_value,
            "publication_status": experiment.publication_status,
            "citations": experiment.citations,
            "impact_factor": experiment.impact_factor,
            "commercial_potential": experiment.commercial_potential,
            "regulatory_implications": experiment.regulatory_implications,
            "ethical_considerations": experiment.ethical_considerations,
            "safety_considerations": experiment.safety_considerations,
            "intellectual_property": experiment.intellectual_property,
            "collaborators": experiment.collaborators,
            "funding_sources": experiment.funding_sources,
            "status": experiment.status,
            "message": "Experiment created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments", response_model=Dict[str, Any])
async def get_experiments(
    skip: int = 0,
    limit: int = 100,
    experiment_type: Optional[str] = None,
    status: Optional[str] = None,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Get experiments with filtering"""
    try:
        experiments = list(engine.experiments.values())
        
        # Apply filters
        if experiment_type:
            experiments = [e for e in experiments if e.experiment_type == experiment_type]
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        # Apply pagination
        total = len(experiments)
        experiments = experiments[skip:skip + limit]
        
        experiment_list = []
        for experiment in experiments:
            experiment_list.append({
                "experiment_id": experiment.experiment_id,
                "timestamp": experiment.timestamp.isoformat(),
                "name": experiment.name,
                "experiment_type": experiment.experiment_type,
                "objective": experiment.objective,
                "hypothesis": experiment.hypothesis,
                "reproducibility": experiment.reproducibility,
                "significance": experiment.significance,
                "effect_size": experiment.effect_size,
                "p_value": experiment.p_value,
                "publication_status": experiment.publication_status,
                "citations": experiment.citations,
                "impact_factor": experiment.impact_factor,
                "commercial_potential": experiment.commercial_potential,
                "status": experiment.status
            })
        
        return {
            "experiments": experiment_list,
            "total": total,
            "skip": skip,
            "limit": limit,
            "message": f"Retrieved {len(experiment_list)} experiments"
        }
        
    except Exception as e:
        logger.error(f"Error getting experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Synthetic biology endpoints
@router.post("/synthetic-biology/design-circuit", response_model=Dict[str, Any])
async def design_genetic_circuit(
    circuit_data: GeneticCircuitDesign,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Design genetic circuit"""
    try:
        circuit_dict = circuit_data.dict()
        circuit = await engine.synthetic_biology.design_genetic_circuit(circuit_dict)
        
        return {
            "circuit": circuit,
            "message": "Genetic circuit designed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error designing genetic circuit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthetic-biology/optimize-circuit/{circuit_id}", response_model=Dict[str, Any])
async def optimize_genetic_circuit(
    circuit_id: str,
    optimization_data: CircuitOptimization,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Optimize genetic circuit"""
    try:
        optimization_result = await engine.synthetic_biology.optimize_circuit(
            circuit_id, 
            optimization_data.optimization_goals
        )
        
        return {
            "optimization_result": optimization_result,
            "message": "Genetic circuit optimized successfully"
        }
        
    except Exception as e:
        logger.error(f"Error optimizing genetic circuit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Protein engineering endpoints
@router.post("/protein-engineering/design-protein", response_model=Dict[str, Any])
async def design_protein(
    protein_data: ProteinDesign,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Design protein with specific properties"""
    try:
        protein_dict = protein_data.dict()
        protein_design = await engine.protein_engineering.design_protein(protein_dict)
        
        return {
            "protein_design": protein_design,
            "message": "Protein designed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error designing protein: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/protein-engineering/predict-structure", response_model=Dict[str, Any])
async def predict_protein_structure(
    structure_data: StructurePrediction,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Predict protein 3D structure"""
    try:
        structure_prediction = await engine.protein_engineering.predict_protein_structure(
            structure_data.sequence
        )
        
        return {
            "structure_prediction": structure_prediction,
            "message": "Protein structure predicted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error predicting protein structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cell engineering endpoints
@router.post("/cell-engineering/engineer-cell", response_model=Dict[str, Any])
async def engineer_cell(
    cell_data: CellEngineering,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Engineer cell with specific properties"""
    try:
        cell_dict = cell_data.dict()
        engineered_cell = await engine.cell_engineering.engineer_cell(cell_dict)
        
        return {
            "engineered_cell": engineered_cell,
            "message": "Cell engineered successfully"
        }
        
    except Exception as e:
        logger.error(f"Error engineering cell: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cell-engineering/differentiate-cell/{cell_id}", response_model=Dict[str, Any])
async def differentiate_cell(
    cell_id: str,
    differentiation_data: CellDifferentiation,
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Differentiate cell to target cell type"""
    try:
        differentiation_result = await engine.cell_engineering.differentiate_cell(
            cell_id, 
            differentiation_data.target_cell_type
        )
        
        return {
            "differentiation_result": differentiation_result,
            "message": "Cell differentiated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error differentiating cell: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System information endpoints
@router.get("/capabilities", response_model=Dict[str, Any])
async def get_biotechnology_capabilities(
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Get biotechnology capabilities"""
    try:
        capabilities = await engine.get_biotechnology_capabilities()
        
        return {
            "capabilities": capabilities,
            "message": "Biotechnology capabilities retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting biotechnology capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-metrics", response_model=Dict[str, Any])
async def get_biotechnology_performance_metrics(
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Get biotechnology performance metrics"""
    try:
        metrics = await engine.get_biotechnology_performance_metrics()
        
        return {
            "performance_metrics": metrics,
            "message": "Biotechnology performance metrics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting biotechnology performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def biotechnology_health_check(
    engine: BiotechnologyEngine = Depends(get_biotech_engine)
):
    """Biotechnology engine health check"""
    try:
        capabilities = await engine.get_biotechnology_capabilities()
        metrics = await engine.get_biotechnology_performance_metrics()
        
        return {
            "status": "healthy",
            "service": "Biotechnology Engine",
            "timestamp": datetime.now().isoformat(),
            "capabilities": capabilities,
            "performance_metrics": metrics,
            "message": "Biotechnology engine is healthy"
        }
        
    except Exception as e:
        logger.error(f"Error in biotechnology health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

















