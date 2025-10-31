"""
Advanced Biomedical and Genomic Data Analysis System
Sistema avanzado de análisis de datos biomédicos y genómicos
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import warnings
warnings.filterwarnings('ignore')

# Biomedical and genomic imports
try:
    import Bio
    from Bio import SeqIO, AlignIO, Phylo
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Align import MultipleSeqAlignment
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    from Bio.Phylo.Consensus import majority_consensus
    from Bio.SeqUtils import molecular_weight, GC
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.Blast import NCBIWWW, NCBIXML
    from Bio.Align.Applications import ClustalwCommandline
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

try:
    import pysam
    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False

try:
    import vcf
    PYVCF_AVAILABLE = True
except ImportError:
    PYVCF_AVAILABLE = False

try:
    import scanpy as sc
    import anndata
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

try:
    import pyensembl
    PYENSEMBL_AVAILABLE = True
except ImportError:
    PYENSEMBL_AVAILABLE = False

# Machine learning for biomedical data
try:
    from sklearn.decomposition import PCA, tSNE, UMAP
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score, train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Statistical analysis
try:
    import scipy.stats as stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Tipos de datos biomédicos"""
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"
    GENOMIC = "genomic"
    TRANSCRIPTOMIC = "transcriptomic"
    PROTEOMIC = "proteomic"
    METABOLOMIC = "metabolomic"
    CLINICAL = "clinical"
    PHENOTYPIC = "phenotypic"

class AnalysisType(Enum):
    """Tipos de análisis biomédicos"""
    SEQUENCE_ANALYSIS = "sequence_analysis"
    ALIGNMENT = "alignment"
    PHYLOGENETIC = "phylogenetic"
    VARIANT_CALLING = "variant_calling"
    EXPRESSION_ANALYSIS = "expression_analysis"
    PATHWAY_ANALYSIS = "pathway_analysis"
    DRUG_DISCOVERY = "drug_discovery"
    CLINICAL_PREDICTION = "clinical_prediction"
    BIOMARKER_DISCOVERY = "biomarker_discovery"

class Organism(Enum):
    """Organismos modelo"""
    HUMAN = "homo_sapiens"
    MOUSE = "mus_musculus"
    RAT = "rattus_norvegicus"
    DROSOPHILA = "drosophila_melanogaster"
    C_ELEGANS = "caenorhabditis_elegans"
    YEAST = "saccharomyces_cerevisiae"
    E_COLI = "escherichia_coli"
    ARABIDOPSIS = "arabidopsis_thaliana"

@dataclass
class BiologicalSequence:
    """Secuencia biológica"""
    id: str
    sequence: str
    sequence_type: DataType
    organism: Optional[Organism] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class GenomicVariant:
    """Variante genómica"""
    id: str
    chromosome: str
    position: int
    reference: str
    alternate: str
    variant_type: str
    quality_score: float
    frequency: Optional[float] = None
    clinical_significance: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ExpressionData:
    """Datos de expresión génica"""
    id: str
    gene_id: str
    sample_id: str
    expression_value: float
    expression_type: str  # TPM, FPKM, counts, etc.
    condition: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ClinicalData:
    """Datos clínicos"""
    id: str
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    biomarkers: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class BiomedicalAnalysis:
    """Análisis biomédico"""
    id: str
    analysis_type: AnalysisType
    input_data: List[str]
    results: Dict[str, Any]
    statistics: Dict[str, float]
    visualizations: List[str]
    insights: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedBiomedicalAnalyzer:
    """
    Analizador avanzado de datos biomédicos y genómicos
    """
    
    def __init__(
        self,
        enable_biopython: bool = True,
        enable_pysam: bool = True,
        enable_pyvcf: bool = True,
        enable_scanpy: bool = True,
        enable_pyensembl: bool = True,
        data_directory: str = "data/biomedical/"
    ):
        self.enable_biopython = enable_biopython and BIOPYTHON_AVAILABLE
        self.enable_pysam = enable_pysam and PYSAM_AVAILABLE
        self.enable_pyvcf = enable_pyvcf and PYVCF_AVAILABLE
        self.enable_scanpy = enable_scanpy and SCANPY_AVAILABLE
        self.enable_pyensembl = enable_pyensembl and PYENSEMBL_AVAILABLE
        self.data_directory = data_directory
        
        # Almacenamiento
        self.biological_sequences: Dict[str, BiologicalSequence] = {}
        self.genomic_variants: Dict[str, GenomicVariant] = {}
        self.expression_data: Dict[str, ExpressionData] = {}
        self.clinical_data: Dict[str, ClinicalData] = {}
        self.biomedical_analyses: Dict[str, BiomedicalAnalysis] = {}
        
        # Configuración
        self.config = {
            "default_organism": Organism.HUMAN,
            "min_sequence_length": 10,
            "max_sequence_length": 1000000,
            "quality_threshold": 20.0,
            "expression_threshold": 1.0,
            "p_value_threshold": 0.05,
            "fold_change_threshold": 2.0
        }
        
        # Crear directorio de datos
        import os
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Inicializar bases de datos
        self._initialize_databases()
        
        logger.info("Advanced Biomedical Analyzer inicializado")
    
    def _initialize_databases(self):
        """Inicializar bases de datos biomédicas"""
        try:
            if self.enable_pyensembl:
                # Inicializar PyEnsembl para anotaciones genéticas
                try:
                    self.ensembl = pyensembl.EnsemblRelease(100)  # Release 100
                    logger.info("PyEnsembl inicializado")
                except Exception as e:
                    logger.warning(f"Error inicializando PyEnsembl: {e}")
                    self.enable_pyensembl = False
            
            # Inicializar Scanpy para análisis de scRNA-seq
            if self.enable_scanpy:
                sc.settings.verbosity = 3
                sc.settings.set_figure_params(dpi=80, facecolor='white')
                logger.info("Scanpy inicializado")
            
            logger.info("Bases de datos biomédicas inicializadas")
            
        except Exception as e:
            logger.error(f"Error inicializando bases de datos: {e}")
    
    async def add_biological_sequence(
        self,
        sequence_id: str,
        sequence: str,
        sequence_type: DataType,
        organism: Optional[Organism] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BiologicalSequence:
        """
        Agregar secuencia biológica
        
        Args:
            sequence_id: ID de la secuencia
            sequence: Secuencia de nucleótidos o aminoácidos
            sequence_type: Tipo de secuencia
            organism: Organismo
            description: Descripción
            metadata: Metadatos adicionales
            
        Returns:
            Secuencia biológica
        """
        try:
            # Validar secuencia
            if len(sequence) < self.config["min_sequence_length"]:
                raise ValueError(f"Secuencia muy corta: {len(sequence)} < {self.config['min_sequence_length']}")
            
            if len(sequence) > self.config["max_sequence_length"]:
                raise ValueError(f"Secuencia muy larga: {len(sequence)} > {self.config['max_sequence_length']}")
            
            # Validar tipo de secuencia
            if sequence_type == DataType.DNA:
                valid_chars = set("ATCGN")
            elif sequence_type == DataType.RNA:
                valid_chars = set("AUCGN")
            elif sequence_type == DataType.PROTEIN:
                valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
            else:
                valid_chars = set("ATCGN")
            
            if not set(sequence.upper()).issubset(valid_chars):
                raise ValueError(f"Secuencia contiene caracteres inválidos para tipo {sequence_type.value}")
            
            # Crear secuencia biológica
            biological_sequence = BiologicalSequence(
                id=sequence_id,
                sequence=sequence.upper(),
                sequence_type=sequence_type,
                organism=organism or self.config["default_organism"],
                description=description,
                metadata=metadata or {}
            )
            
            # Almacenar secuencia
            self.biological_sequences[sequence_id] = biological_sequence
            
            logger.info(f"Secuencia biológica agregada: {sequence_id} ({sequence_type.value}, {len(sequence)} bp)")
            return biological_sequence
            
        except Exception as e:
            logger.error(f"Error agregando secuencia biológica: {e}")
            raise
    
    async def analyze_sequence(
        self,
        sequence_id: str,
        analysis_type: AnalysisType = AnalysisType.SEQUENCE_ANALYSIS
    ) -> BiomedicalAnalysis:
        """
        Analizar secuencia biológica
        
        Args:
            sequence_id: ID de la secuencia
            analysis_type: Tipo de análisis
            
        Returns:
            Análisis biomédico
        """
        try:
            if sequence_id not in self.biological_sequences:
                raise ValueError(f"Secuencia {sequence_id} no encontrada")
            
            sequence = self.biological_sequences[sequence_id]
            
            logger.info(f"Analizando secuencia {sequence_id} ({analysis_type.value})")
            
            # Realizar análisis según el tipo
            if analysis_type == AnalysisType.SEQUENCE_ANALYSIS:
                results = await self._analyze_sequence_properties(sequence)
            elif analysis_type == AnalysisType.ALIGNMENT:
                results = await self._perform_sequence_alignment(sequence)
            elif analysis_type == AnalysisType.PHYLOGENETIC:
                results = await self._perform_phylogenetic_analysis(sequence)
            else:
                results = await self._analyze_sequence_properties(sequence)
            
            # Calcular estadísticas
            statistics = await self._calculate_sequence_statistics(sequence, results)
            
            # Generar visualizaciones
            visualizations = await self._create_sequence_visualizations(sequence, results)
            
            # Generar insights
            insights = await self._generate_sequence_insights(sequence, results, statistics)
            
            # Crear análisis
            analysis = BiomedicalAnalysis(
                id=f"analysis_{sequence_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                analysis_type=analysis_type,
                input_data=[sequence_id],
                results=results,
                statistics=statistics,
                visualizations=visualizations,
                insights=insights
            )
            
            # Almacenar análisis
            self.biomedical_analyses[analysis.id] = analysis
            
            logger.info(f"Análisis de secuencia completado: {analysis.id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando secuencia: {e}")
            raise
    
    async def _analyze_sequence_properties(self, sequence: BiologicalSequence) -> Dict[str, Any]:
        """Analizar propiedades de la secuencia"""
        try:
            results = {}
            
            if self.enable_biopython:
                # Crear objeto Seq de BioPython
                bio_seq = Seq(sequence.sequence)
                
                # Análisis básico
                results["length"] = len(bio_seq)
                results["composition"] = {
                    "A": bio_seq.count("A"),
                    "T": bio_seq.count("T"),
                    "C": bio_seq.count("C"),
                    "G": bio_seq.count("G"),
                    "N": bio_seq.count("N")
                }
                
                # Contenido GC
                if sequence.sequence_type in [DataType.DNA, DataType.RNA]:
                    results["gc_content"] = GC(bio_seq)
                
                # Peso molecular
                if sequence.sequence_type == DataType.PROTEIN:
                    results["molecular_weight"] = molecular_weight(bio_seq, "protein")
                else:
                    results["molecular_weight"] = molecular_weight(bio_seq, "DNA")
                
                # Análisis de proteína
                if sequence.sequence_type == DataType.PROTEIN:
                    protein_analysis = ProteinAnalysis(str(bio_seq))
                    results["protein_analysis"] = {
                        "amino_acid_count": protein_analysis.count_amino_acids(),
                        "molecular_weight": protein_analysis.molecular_weight(),
                        "aromaticity": protein_analysis.aromaticity(),
                        "instability_index": protein_analysis.instability_index(),
                        "isoelectric_point": protein_analysis.isoelectric_point(),
                        "secondary_structure_fraction": protein_analysis.secondary_structure_fraction()
                    }
                
                # Motivos y patrones
                results["motifs"] = await self._find_sequence_motifs(bio_seq, sequence.sequence_type)
                
                # Análisis de codones (para DNA)
                if sequence.sequence_type == DataType.DNA and len(bio_seq) % 3 == 0:
                    results["codon_analysis"] = await self._analyze_codons(bio_seq)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analizando propiedades de secuencia: {e}")
            return {}
    
    async def _find_sequence_motifs(self, sequence: Seq, sequence_type: DataType) -> Dict[str, List]:
        """Encontrar motivos en la secuencia"""
        try:
            motifs = {
                "promoter_motifs": [],
                "transcription_factor_binding_sites": [],
                "restriction_sites": [],
                "repeats": []
            }
            
            if sequence_type == DataType.DNA:
                # Motivos de promotor comunes
                promoter_motifs = [
                    "TATAAT",  # TATA box
                    "TTGACA",  # -35 box
                    "TATAWAW",  # TATA box variante
                    "CAAT",    # CAAT box
                    "GCGCGC"   # GC box
                ]
                
                for motif in promoter_motifs:
                    positions = []
                    start = 0
                    while True:
                        pos = sequence.find(motif, start)
                        if pos == -1:
                            break
                        positions.append(pos)
                        start = pos + 1
                    
                    if positions:
                        motifs["promoter_motifs"].append({
                            "motif": motif,
                            "positions": positions
                        })
                
                # Sitios de restricción comunes
                restriction_sites = [
                    "GAATTC",  # EcoRI
                    "AAGCTT",  # HindIII
                    "GGATCC",  # BamHI
                    "CTCGAG",  # XhoI
                    "GCGGCCGC"  # NotI
                ]
                
                for site in restriction_sites:
                    positions = []
                    start = 0
                    while True:
                        pos = sequence.find(site, start)
                        if pos == -1:
                            break
                        positions.append(pos)
                        start = pos + 1
                    
                    if positions:
                        motifs["restriction_sites"].append({
                            "enzyme": site,
                            "positions": positions
                        })
            
            elif sequence_type == DataType.PROTEIN:
                # Motivos de proteína
                protein_motifs = [
                    "N[^P][ST][^P]",  # N-glycosylation
                    "[ST]P",          # Proline-directed phosphorylation
                    "C[^C][^C]C",     # Zinc finger
                    "G[^G][^G]G"      # G-quadruplex
                ]
                
                for motif in protein_motifs:
                    # Implementación simplificada de búsqueda de motivos
                    # En un sistema real, usarías herramientas como MEME
                    pass
            
            return motifs
            
        except Exception as e:
            logger.error(f"Error encontrando motivos: {e}")
            return {}
    
    async def _analyze_codons(self, sequence: Seq) -> Dict[str, Any]:
        """Analizar codones en secuencia de DNA"""
        try:
            codon_analysis = {
                "codon_usage": {},
                "start_codons": [],
                "stop_codons": [],
                "codon_bias": {}
            }
            
            # Codones de inicio y parada
            start_codons = ["ATG"]
            stop_codons = ["TAA", "TAG", "TGA"]
            
            # Analizar cada codón
            for i in range(0, len(sequence) - 2, 3):
                codon = str(sequence[i:i+3])
                
                if len(codon) == 3:
                    # Contar uso de codones
                    codon_analysis["codon_usage"][codon] = codon_analysis["codon_usage"].get(codon, 0) + 1
                    
                    # Identificar codones de inicio y parada
                    if codon in start_codons:
                        codon_analysis["start_codons"].append(i)
                    elif codon in stop_codons:
                        codon_analysis["stop_codons"].append(i)
            
            # Calcular sesgo de codones (simplificado)
            total_codons = sum(codon_analysis["codon_usage"].values())
            for codon, count in codon_analysis["codon_usage"].items():
                codon_analysis["codon_bias"][codon] = count / total_codons if total_codons > 0 else 0
            
            return codon_analysis
            
        except Exception as e:
            logger.error(f"Error analizando codones: {e}")
            return {}
    
    async def _perform_sequence_alignment(self, sequence: BiologicalSequence) -> Dict[str, Any]:
        """Realizar alineamiento de secuencias"""
        try:
            alignment_results = {
                "self_alignment": {},
                "database_hits": [],
                "conservation_scores": []
            }
            
            if self.enable_biopython:
                # Alineamiento consigo mismo (para encontrar repeticiones)
                bio_seq = Seq(sequence.sequence)
                
                # Búsqueda de repeticiones simples
                repeats = []
                for i in range(len(bio_seq) - 10):
                    for j in range(i + 10, len(bio_seq) - 10):
                        if bio_seq[i:i+10] == bio_seq[j:j+10]:
                            repeats.append({
                                "position1": i,
                                "position2": j,
                                "sequence": str(bio_seq[i:i+10])
                            })
                
                alignment_results["self_alignment"]["repeats"] = repeats
                
                # Simulación de búsqueda en base de datos
                # En un sistema real, usarías BLAST
                alignment_results["database_hits"] = [
                    {
                        "accession": "simulated_hit_1",
                        "description": "Simulated protein sequence",
                        "identity": 85.5,
                        "e_value": 1e-50,
                        "alignment_length": 150
                    }
                ]
            
            return alignment_results
            
        except Exception as e:
            logger.error(f"Error en alineamiento de secuencias: {e}")
            return {}
    
    async def _perform_phylogenetic_analysis(self, sequence: BiologicalSequence) -> Dict[str, Any]:
        """Realizar análisis filogenético"""
        try:
            phylogenetic_results = {
                "distance_matrix": {},
                "phylogenetic_tree": {},
                "evolutionary_relationships": []
            }
            
            if self.enable_biopython:
                # Crear secuencias de referencia simuladas
                reference_sequences = [
                    ("species_1", sequence.sequence),
                    ("species_2", self._mutate_sequence(sequence.sequence, 0.1)),
                    ("species_3", self._mutate_sequence(sequence.sequence, 0.15)),
                    ("species_4", self._mutate_sequence(sequence.sequence, 0.2))
                ]
                
                # Calcular matriz de distancias
                sequences = [seq for _, seq in reference_sequences]
                distance_matrix = self._calculate_distance_matrix(sequences)
                
                phylogenetic_results["distance_matrix"] = {
                    "species": [name for name, _ in reference_sequences],
                    "distances": distance_matrix.tolist()
                }
                
                # Construir árbol filogenético (simplificado)
                phylogenetic_results["phylogenetic_tree"] = {
                    "newick": "((species_1,species_2),(species_3,species_4));",
                    "bootstrap_values": [95, 87, 92]
                }
            
            return phylogenetic_results
            
        except Exception as e:
            logger.error(f"Error en análisis filogenético: {e}")
            return {}
    
    def _mutate_sequence(self, sequence: str, mutation_rate: float) -> str:
        """Mutate sequence with given rate"""
        try:
            mutated = list(sequence)
            for i in range(len(mutated)):
                if np.random.random() < mutation_rate:
                    if mutated[i] in "ATCG":
                        # Mutate to different nucleotide
                        options = [n for n in "ATCG" if n != mutated[i]]
                        mutated[i] = np.random.choice(options)
            
            return "".join(mutated)
            
        except Exception as e:
            logger.error(f"Error mutando secuencia: {e}")
            return sequence
    
    def _calculate_distance_matrix(self, sequences: List[str]) -> np.ndarray:
        """Calcular matriz de distancias entre secuencias"""
        try:
            n = len(sequences)
            distance_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    # Calcular distancia de Hamming
                    seq1, seq2 = sequences[i], sequences[j]
                    min_len = min(len(seq1), len(seq2))
                    
                    if min_len == 0:
                        distance = 1.0
                    else:
                        matches = sum(1 for k in range(min_len) if seq1[k] == seq2[k])
                        distance = 1.0 - (matches / min_len)
                    
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
            
            return distance_matrix
            
        except Exception as e:
            logger.error(f"Error calculando matriz de distancias: {e}")
            return np.zeros((len(sequences), len(sequences)))
    
    async def _calculate_sequence_statistics(self, sequence: BiologicalSequence, results: Dict[str, Any]) -> Dict[str, float]:
        """Calcular estadísticas de la secuencia"""
        try:
            statistics = {}
            
            # Estadísticas básicas
            statistics["length"] = len(sequence.sequence)
            statistics["gc_content"] = results.get("gc_content", 0.0)
            statistics["molecular_weight"] = results.get("molecular_weight", 0.0)
            
            # Estadísticas de composición
            composition = results.get("composition", {})
            total_bases = sum(composition.values())
            if total_bases > 0:
                for base, count in composition.items():
                    statistics[f"{base}_frequency"] = count / total_bases
            
            # Estadísticas de motivos
            motifs = results.get("motifs", {})
            statistics["promoter_motifs_count"] = len(motifs.get("promoter_motifs", []))
            statistics["restriction_sites_count"] = len(motifs.get("restriction_sites", []))
            
            # Estadísticas de codones
            codon_analysis = results.get("codon_analysis", {})
            statistics["start_codons_count"] = len(codon_analysis.get("start_codons", []))
            statistics["stop_codons_count"] = len(codon_analysis.get("stop_codons", []))
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculando estadísticas: {e}")
            return {}
    
    async def _create_sequence_visualizations(self, sequence: BiologicalSequence, results: Dict[str, Any]) -> List[str]:
        """Crear visualizaciones de la secuencia"""
        try:
            visualizations = []
            
            # Gráfico de composición de bases
            composition = results.get("composition", {})
            if composition:
                fig, ax = plt.subplots(figsize=(10, 6))
                bases = list(composition.keys())
                counts = list(composition.values())
                
                ax.bar(bases, counts)
                ax.set_title(f"Composición de bases - {sequence.id}")
                ax.set_xlabel("Bases")
                ax.set_ylabel("Frecuencia")
                
                viz_path = f"{self.data_directory}/composition_{sequence.id}.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(viz_path)
            
            # Gráfico de contenido GC por ventana
            if len(sequence.sequence) > 100:
                window_size = 50
                gc_content_windows = []
                positions = []
                
                for i in range(0, len(sequence.sequence) - window_size, window_size):
                    window = sequence.sequence[i:i + window_size]
                    gc_count = window.count('G') + window.count('C')
                    gc_content = gc_count / window_size
                    gc_content_windows.append(gc_content)
                    positions.append(i)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(positions, gc_content_windows)
                ax.set_title(f"Contenido GC por ventana - {sequence.id}")
                ax.set_xlabel("Posición")
                ax.set_ylabel("Contenido GC")
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='GC = 0.5')
                ax.legend()
                
                viz_path = f"{self.data_directory}/gc_content_{sequence.id}.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(viz_path)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creando visualizaciones: {e}")
            return []
    
    async def _generate_sequence_insights(self, sequence: BiologicalSequence, results: Dict[str, Any], statistics: Dict[str, float]) -> List[str]:
        """Generar insights de la secuencia"""
        try:
            insights = []
            
            # Insight sobre longitud
            length = statistics.get("length", 0)
            if length > 10000:
                insights.append("Secuencia larga, posible gen completo o región genómica extensa")
            elif length > 1000:
                insights.append("Secuencia de longitud media, posible gen o transcripto")
            else:
                insights.append("Secuencia corta, posible fragmento o secuencia reguladora")
            
            # Insight sobre contenido GC
            gc_content = statistics.get("gc_content", 0)
            if gc_content > 0.6:
                insights.append("Alto contenido GC, región rica en genes o promotores")
            elif gc_content < 0.4:
                insights.append("Bajo contenido GC, posible región intergénica")
            else:
                insights.append("Contenido GC equilibrado")
            
            # Insight sobre motivos
            promoter_count = statistics.get("promoter_motifs_count", 0)
            if promoter_count > 0:
                insights.append(f"Se encontraron {promoter_count} motivos de promotor, posible región reguladora")
            
            restriction_count = statistics.get("restriction_sites_count", 0)
            if restriction_count > 0:
                insights.append(f"Se encontraron {restriction_count} sitios de restricción, útil para clonación")
            
            # Insight sobre codones
            start_codons = statistics.get("start_codons_count", 0)
            stop_codons = statistics.get("stop_codons_count", 0)
            if start_codons > 0 and stop_codons > 0:
                insights.append("Secuencia contiene codones de inicio y parada, posible marco de lectura abierto")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generando insights: {e}")
            return []
    
    async def add_genomic_variant(
        self,
        variant_id: str,
        chromosome: str,
        position: int,
        reference: str,
        alternate: str,
        quality_score: float,
        frequency: Optional[float] = None,
        clinical_significance: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GenomicVariant:
        """
        Agregar variante genómica
        
        Args:
            variant_id: ID de la variante
            chromosome: Cromosoma
            position: Posición
            reference: Alelo de referencia
            alternate: Alelo alternativo
            quality_score: Score de calidad
            frequency: Frecuencia poblacional
            clinical_significance: Significado clínico
            metadata: Metadatos adicionales
            
        Returns:
            Variante genómica
        """
        try:
            # Determinar tipo de variante
            variant_type = self._determine_variant_type(reference, alternate)
            
            # Crear variante genómica
            variant = GenomicVariant(
                id=variant_id,
                chromosome=chromosome,
                position=position,
                reference=reference,
                alternate=alternate,
                variant_type=variant_type,
                quality_score=quality_score,
                frequency=frequency,
                clinical_significance=clinical_significance,
                metadata=metadata or {}
            )
            
            # Almacenar variante
            self.genomic_variants[variant_id] = variant
            
            logger.info(f"Variante genómica agregada: {variant_id} ({variant_type})")
            return variant
            
        except Exception as e:
            logger.error(f"Error agregando variante genómica: {e}")
            raise
    
    def _determine_variant_type(self, reference: str, alternate: str) -> str:
        """Determinar tipo de variante"""
        try:
            if len(reference) == len(alternate):
                if len(reference) == 1:
                    return "SNV"  # Single Nucleotide Variant
                else:
                    return "MNV"  # Multi Nucleotide Variant
            elif len(reference) > len(alternate):
                return "DEL"  # Deletion
            else:
                return "INS"  # Insertion
        except Exception as e:
            logger.error(f"Error determinando tipo de variante: {e}")
            return "UNKNOWN"
    
    async def analyze_variants(
        self,
        variant_ids: List[str],
        analysis_type: AnalysisType = AnalysisType.VARIANT_CALLING
    ) -> BiomedicalAnalysis:
        """
        Analizar variantes genómicas
        
        Args:
            variant_ids: IDs de las variantes
            analysis_type: Tipo de análisis
            
        Returns:
            Análisis biomédico
        """
        try:
            logger.info(f"Analizando {len(variant_ids)} variantes genómicas")
            
            # Filtrar variantes válidas
            valid_variants = [v for v in variant_ids if v in self.genomic_variants]
            
            if not valid_variants:
                raise ValueError("No se encontraron variantes válidas")
            
            # Realizar análisis
            results = await self._analyze_variant_properties(valid_variants)
            
            # Calcular estadísticas
            statistics = await self._calculate_variant_statistics(valid_variants, results)
            
            # Generar visualizaciones
            visualizations = await self._create_variant_visualizations(valid_variants, results)
            
            # Generar insights
            insights = await self._generate_variant_insights(valid_variants, results, statistics)
            
            # Crear análisis
            analysis = BiomedicalAnalysis(
                id=f"variant_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                analysis_type=analysis_type,
                input_data=valid_variants,
                results=results,
                statistics=statistics,
                visualizations=visualizations,
                insights=insights
            )
            
            # Almacenar análisis
            self.biomedical_analyses[analysis.id] = analysis
            
            logger.info(f"Análisis de variantes completado: {analysis.id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando variantes: {e}")
            raise
    
    async def _analyze_variant_properties(self, variant_ids: List[str]) -> Dict[str, Any]:
        """Analizar propiedades de las variantes"""
        try:
            results = {
                "variant_summary": {},
                "chromosome_distribution": {},
                "variant_type_distribution": {},
                "quality_distribution": {},
                "clinical_significance": {},
                "pathogenic_variants": [],
                "common_variants": [],
                "rare_variants": []
            }
            
            for variant_id in variant_ids:
                variant = self.genomic_variants[variant_id]
                
                # Resumen de variante
                results["variant_summary"][variant_id] = {
                    "chromosome": variant.chromosome,
                    "position": variant.position,
                    "type": variant.variant_type,
                    "quality": variant.quality_score,
                    "frequency": variant.frequency,
                    "clinical_significance": variant.clinical_significance
                }
                
                # Distribución por cromosoma
                results["chromosome_distribution"][variant.chromosome] = results["chromosome_distribution"].get(variant.chromosome, 0) + 1
                
                # Distribución por tipo
                results["variant_type_distribution"][variant.variant_type] = results["variant_type_distribution"].get(variant.variant_type, 0) + 1
                
                # Clasificar por frecuencia
                if variant.frequency is not None:
                    if variant.frequency > 0.05:
                        results["common_variants"].append(variant_id)
                    elif variant.frequency < 0.01:
                        results["rare_variants"].append(variant_id)
                
                # Clasificar por significado clínico
                if variant.clinical_significance:
                    if "pathogenic" in variant.clinical_significance.lower():
                        results["pathogenic_variants"].append(variant_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analizando propiedades de variantes: {e}")
            return {}
    
    async def _calculate_variant_statistics(self, variant_ids: List[str], results: Dict[str, Any]) -> Dict[str, float]:
        """Calcular estadísticas de las variantes"""
        try:
            statistics = {}
            
            variants = [self.genomic_variants[v_id] for v_id in variant_ids]
            
            # Estadísticas básicas
            statistics["total_variants"] = len(variants)
            statistics["average_quality"] = np.mean([v.quality_score for v in variants])
            statistics["median_quality"] = np.median([v.quality_score for v in variants])
            
            # Estadísticas de frecuencia
            frequencies = [v.frequency for v in variants if v.frequency is not None]
            if frequencies:
                statistics["average_frequency"] = np.mean(frequencies)
                statistics["median_frequency"] = np.median(frequencies)
                statistics["rare_variant_percentage"] = len([f for f in frequencies if f < 0.01]) / len(frequencies) * 100
            
            # Estadísticas de tipos
            type_counts = results.get("variant_type_distribution", {})
            total_variants = sum(type_counts.values())
            if total_variants > 0:
                for variant_type, count in type_counts.items():
                    statistics[f"{variant_type}_percentage"] = count / total_variants * 100
            
            # Estadísticas clínicas
            pathogenic_count = len(results.get("pathogenic_variants", []))
            statistics["pathogenic_percentage"] = pathogenic_count / len(variants) * 100 if variants else 0
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculando estadísticas de variantes: {e}")
            return {}
    
    async def _create_variant_visualizations(self, variant_ids: List[str], results: Dict[str, Any]) -> List[str]:
        """Crear visualizaciones de las variantes"""
        try:
            visualizations = []
            
            # Gráfico de distribución por cromosoma
            chromosome_dist = results.get("chromosome_distribution", {})
            if chromosome_dist:
                fig, ax = plt.subplots(figsize=(12, 6))
                chromosomes = list(chromosome_dist.keys())
                counts = list(chromosome_dist.values())
                
                ax.bar(chromosomes, counts)
                ax.set_title("Distribución de variantes por cromosoma")
                ax.set_xlabel("Cromosoma")
                ax.set_ylabel("Número de variantes")
                plt.xticks(rotation=45)
                
                viz_path = f"{self.data_directory}/variant_chromosome_distribution.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(viz_path)
            
            # Gráfico de distribución por tipo
            type_dist = results.get("variant_type_distribution", {})
            if type_dist:
                fig, ax = plt.subplots(figsize=(8, 8))
                types = list(type_dist.keys())
                counts = list(type_dist.values())
                
                ax.pie(counts, labels=types, autopct='%1.1f%%')
                ax.set_title("Distribución de variantes por tipo")
                
                viz_path = f"{self.data_directory}/variant_type_distribution.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(viz_path)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creando visualizaciones de variantes: {e}")
            return []
    
    async def _generate_variant_insights(self, variant_ids: List[str], results: Dict[str, Any], statistics: Dict[str, float]) -> List[str]:
        """Generar insights de las variantes"""
        try:
            insights = []
            
            # Insight sobre número total
            total_variants = statistics.get("total_variants", 0)
            insights.append(f"Se analizaron {total_variants} variantes genómicas")
            
            # Insight sobre calidad
            avg_quality = statistics.get("average_quality", 0)
            if avg_quality > 30:
                insights.append("Alta calidad promedio de las variantes")
            elif avg_quality > 20:
                insights.append("Calidad moderada de las variantes")
            else:
                insights.append("Baja calidad promedio de las variantes")
            
            # Insight sobre tipos
            snv_percentage = statistics.get("SNV_percentage", 0)
            if snv_percentage > 70:
                insights.append("Predominan las variantes de nucleótido único (SNV)")
            
            # Insight sobre variantes raras
            rare_percentage = statistics.get("rare_variant_percentage", 0)
            if rare_percentage > 50:
                insights.append("Alto porcentaje de variantes raras, posible carga genética significativa")
            
            # Insight sobre variantes patogénicas
            pathogenic_percentage = statistics.get("pathogenic_percentage", 0)
            if pathogenic_percentage > 10:
                insights.append("Alto porcentaje de variantes patogénicas, requiere atención clínica")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generando insights de variantes: {e}")
            return []
    
    async def get_biomedical_summary(self) -> Dict[str, Any]:
        """Obtener resumen del sistema biomédico"""
        try:
            return {
                "total_sequences": len(self.biological_sequences),
                "total_variants": len(self.genomic_variants),
                "total_expression_data": len(self.expression_data),
                "total_clinical_data": len(self.clinical_data),
                "total_analyses": len(self.biomedical_analyses),
                "sequence_types": {
                    seq_type.value: len([s for s in self.biological_sequences.values() if s.sequence_type == seq_type])
                    for seq_type in DataType
                },
                "analysis_types": {
                    analysis_type.value: len([a for a in self.biomedical_analyses.values() if a.analysis_type == analysis_type])
                    for analysis_type in AnalysisType
                },
                "capabilities": {
                    "biopython": self.enable_biopython,
                    "pysam": self.enable_pysam,
                    "pyvcf": self.enable_pyvcf,
                    "scanpy": self.enable_scanpy,
                    "pyensembl": self.enable_pyensembl
                },
                "last_activity": max([
                    max([s.created_at for s in self.biological_sequences.values()]) if self.biological_sequences else datetime.min,
                    max([v.created_at for v in self.genomic_variants.values()]) if self.genomic_variants else datetime.min,
                    max([a.created_at for a in self.biomedical_analyses.values()]) if self.biomedical_analyses else datetime.min
                ]).isoformat() if any([self.biological_sequences, self.genomic_variants, self.biomedical_analyses]) else None
            }
        except Exception as e:
            logger.error(f"Error obteniendo resumen biomédico: {e}")
            return {}
    
    async def export_biomedical_data(self, filepath: str = None) -> str:
        """Exportar datos biomédicos"""
        try:
            if filepath is None:
                filepath = f"exports/biomedical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            export_data = {
                "biological_sequences": {
                    seq_id: {
                        "sequence": seq.sequence,
                        "sequence_type": seq.sequence_type.value,
                        "organism": seq.organism.value if seq.organism else None,
                        "description": seq.description,
                        "metadata": seq.metadata,
                        "created_at": seq.created_at.isoformat()
                    }
                    for seq_id, seq in self.biological_sequences.items()
                },
                "genomic_variants": {
                    var_id: {
                        "chromosome": variant.chromosome,
                        "position": variant.position,
                        "reference": variant.reference,
                        "alternate": variant.alternate,
                        "variant_type": variant.variant_type,
                        "quality_score": variant.quality_score,
                        "frequency": variant.frequency,
                        "clinical_significance": variant.clinical_significance,
                        "metadata": variant.metadata,
                        "created_at": variant.created_at.isoformat()
                    }
                    for var_id, variant in self.genomic_variants.items()
                },
                "biomedical_analyses": {
                    analysis_id: {
                        "analysis_type": analysis.analysis_type.value,
                        "input_data": analysis.input_data,
                        "results": analysis.results,
                        "statistics": analysis.statistics,
                        "visualizations": analysis.visualizations,
                        "insights": analysis.insights,
                        "created_at": analysis.created_at.isoformat()
                    }
                    for analysis_id, analysis in self.biomedical_analyses.items()
                },
                "summary": await self.get_biomedical_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Datos biomédicos exportados a {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exportando datos biomédicos: {e}")
            raise
























