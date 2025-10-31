import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  Button, 
  StyleSheet, 
  ScrollView, 
  ActivityIndicator, 
  Alert, 
  FlatList,
  TouchableOpacity,
  Modal,
  TextInput
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import axios from 'axios';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { format } from 'date-fns';

// Types
interface QuantumBackend {
  name: string;
  type: string;
  qubits: number;
  status: string;
}

interface QuantumJob {
  job_id: string;
  circuit_id: string;
  algorithm: string;
  backend: string;
  shots: number;
  status: string;
  result?: any;
  execution_time?: number;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

interface QuantumMLModel {
  model_id: string;
  name: string;
  model_type: string;
  num_qubits: number;
  num_layers: number;
  parameters: Record<string, any>;
  accuracy: number;
  loss: number;
  is_trained: boolean;
  created_at: string;
}

interface QuantumOptimization {
  optimization_id: string;
  problem_type: string;
  objective_function: string;
  constraints: string[];
  variables: string[];
  num_qubits: number;
  algorithm: string;
  backend: string;
  solution?: any;
  optimal_value?: number;
  convergence_history: number[];
  created_at: string;
}

// API calls
const createQuantumCircuit = async (circuitInfo: Record<string, any>): Promise<{circuit_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/quantum-neural/quantum/circuits/create', circuitInfo);
  return response.data;
};

const executeQuantumJob = async (jobInfo: Record<string, any>): Promise<{job_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/quantum-neural/quantum/jobs/execute', jobInfo);
  return response.data;
};

const trainQuantumMLModel = async (modelInfo: Record<string, any>): Promise<{model_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/quantum-neural/quantum/ml/train', modelInfo);
  return response.data;
};

const solveQuantumOptimization = async (optimizationInfo: Record<string, any>): Promise<{optimization_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/quantum-neural/quantum/optimization/solve', optimizationInfo);
  return response.data;
};

const getQuantumJobStatus = async (jobId: string): Promise<QuantumJob> => {
  const response = await axios.get(`http://localhost:8000/api/v1/quantum-neural/quantum/jobs/${jobId}/status`);
  return response.data;
};

const getQuantumModelInfo = async (modelId: string): Promise<QuantumMLModel> => {
  const response = await axios.get(`http://localhost:8000/api/v1/quantum-neural/quantum/ml/${modelId}`);
  return response.data;
};

const getQuantumOptimizationResult = async (optimizationId: string): Promise<QuantumOptimization> => {
  const response = await axios.get(`http://localhost:8000/api/v1/quantum-neural/quantum/optimization/${optimizationId}`);
  return response.data;
};

const getAvailableBackends = async (): Promise<QuantumBackend[]> => {
  const response = await axios.get('http://localhost:8000/api/v1/quantum-neural/quantum/backends');
  return response.data.backends || [];
};

const getQuantumStatistics = async (): Promise<Record<string, any>> => {
  const response = await axios.get('http://localhost:8000/api/v1/quantum-neural/quantum/statistics');
  return response.data;
};

const QuantumAIScreen: React.FC = () => {
  const queryClient = useQueryClient();
  
  // State for Circuit Creation
  const [circuitName, setCircuitName] = useState<string>('');
  const [circuitDescription, setCircuitDescription] = useState<string>('');
  const [numQubits, setNumQubits] = useState<string>('3');
  const [selectedBackend, setSelectedBackend] = useState<string>('qiskit');
  
  // State for Job Execution
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('grover');
  const [shots, setShots] = useState<string>('1024');
  const [jobParameters, setJobParameters] = useState<string>('');
  
  // State for ML Model Training
  const [modelName, setModelName] = useState<string>('');
  const [modelType, setModelType] = useState<string>('classifier');
  const [modelQubits, setModelQubits] = useState<string>('4');
  const [numLayers, setNumLayers] = useState<string>('2');
  
  // State for Optimization
  const [problemType, setProblemType] = useState<string>('minimization');
  const [objectiveFunction, setObjectiveFunction] = useState<string>('');
  const [constraints, setConstraints] = useState<string>('');
  const [variables, setVariables] = useState<string>('');
  
  // State for tracking
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [activeOptimizationId, setActiveOptimizationId] = useState<string | null>(null);
  
  // State for modals
  const [showCircuitModal, setShowCircuitModal] = useState<boolean>(false);
  const [showJobModal, setShowJobModal] = useState<boolean>(false);
  const [showMLModal, setShowMLModal] = useState<boolean>(false);
  const [showOptimizationModal, setShowOptimizationModal] = useState<boolean>(false);
  
  // Options
  const algorithms = [
    { value: 'grover', label: "Grover's Algorithm" },
    { value: 'shor', label: "Shor's Algorithm" },
    { value: 'qaoa', label: 'QAOA (Quantum Approximate Optimization)' },
    { value: 'vqe', label: 'VQE (Variational Quantum Eigensolver)' },
    { value: 'qml_classifier', label: 'Quantum ML Classifier' },
    { value: 'qml_regressor', label: 'Quantum ML Regressor' },
    { value: 'quantum_neural_network', label: 'Quantum Neural Network' },
    { value: 'variational_quantum_eigensolver', label: 'Variational Quantum Eigensolver' }
  ];
  
  const modelTypes = [
    { value: 'classifier', label: 'Classifier' },
    { value: 'regressor', label: 'Regressor' },
    { value: 'generator', label: 'Generator' },
    { value: 'discriminator', label: 'Discriminator' },
    { value: 'autoencoder', label: 'Autoencoder' },
    { value: 'transformer', label: 'Transformer' }
  ];
  
  const problemTypes = [
    { value: 'minimization', label: 'Minimization' },
    { value: 'maximization', label: 'Maximization' },
    { value: 'constraint_satisfaction', label: 'Constraint Satisfaction' },
    { value: 'portfolio_optimization', label: 'Portfolio Optimization' },
    { value: 'traveling_salesman', label: 'Traveling Salesman' },
    { value: 'max_cut', label: 'Max Cut' }
  ];
  
  // Queries
  const { data: availableBackends, isLoading: isLoadingBackends } = useQuery<QuantumBackend[], Error>(
    'quantumBackends',
    getAvailableBackends,
    {
      refetchInterval: 60000,
    }
  );
  
  const { data: statistics, isLoading: isLoadingStats } = useQuery<Record<string, any>, Error>(
    'quantumStatistics',
    getQuantumStatistics,
    {
      refetchInterval: 30000,
    }
  );
  
  const { data: jobStatus, isLoading: isLoadingJob } = useQuery<QuantumJob, Error>(
    ['quantumJobStatus', activeJobId],
    () => activeJobId ? getQuantumJobStatus(activeJobId) : Promise.resolve(null),
    {
      enabled: !!activeJobId,
      refetchInterval: 2000,
    }
  );
  
  const { data: modelInfo, isLoading: isLoadingModel } = useQuery<QuantumMLModel, Error>(
    ['quantumModelInfo', activeModelId],
    () => activeModelId ? getQuantumModelInfo(activeModelId) : Promise.resolve(null),
    {
      enabled: !!activeModelId,
      refetchInterval: 5000,
    }
  );
  
  const { data: optimizationResult, isLoading: isLoadingOptimization } = useQuery<QuantumOptimization, Error>(
    ['quantumOptimizationResult', activeOptimizationId],
    () => activeOptimizationId ? getQuantumOptimizationResult(activeOptimizationId) : Promise.resolve(null),
    {
      enabled: !!activeOptimizationId,
      refetchInterval: 3000,
    }
  );
  
  // Mutations
  const createCircuitMutation = useMutation<{circuit_id: string}, Error, Record<string, any>>(createQuantumCircuit, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Circuito cuántico creado: ${data.circuit_id}`);
      setShowCircuitModal(false);
      queryClient.invalidateQueries('quantumStatistics');
    },
    onError: (error) => {
      Alert.alert('Error', `Error al crear circuito: ${error.message}`);
    },
  });
  
  const executeJobMutation = useMutation<{job_id: string}, Error, Record<string, any>>(executeQuantumJob, {
    onSuccess: (data) => {
      setActiveJobId(data.job_id);
      Alert.alert('Éxito', `Trabajo cuántico iniciado: ${data.job_id}`);
      setShowJobModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al ejecutar trabajo: ${error.message}`);
    },
  });
  
  const trainModelMutation = useMutation<{model_id: string}, Error, Record<string, any>>(trainQuantumMLModel, {
    onSuccess: (data) => {
      setActiveModelId(data.model_id);
      Alert.alert('Éxito', `Modelo ML cuántico iniciado: ${data.model_id}`);
      setShowMLModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al entrenar modelo: ${error.message}`);
    },
  });
  
  const solveOptimizationMutation = useMutation<{optimization_id: string}, Error, Record<string, any>>(solveQuantumOptimization, {
    onSuccess: (data) => {
      setActiveOptimizationId(data.optimization_id);
      Alert.alert('Éxito', `Optimización cuántica iniciada: ${data.optimization_id}`);
      setShowOptimizationModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al resolver optimización: ${error.message}`);
    },
  });
  
  // Handlers
  const handleCreateCircuit = () => {
    if (!circuitName.trim()) {
      Alert.alert('Error', 'Por favor ingresa un nombre para el circuito.');
      return;
    }
    
    createCircuitMutation.mutate({
      name: circuitName,
      description: circuitDescription,
      num_qubits: parseInt(numQubits, 10),
      backend: selectedBackend,
      gates: [
        { type: 'hadamard', qubits: [0] },
        { type: 'cnot', qubits: [0, 1] },
        { type: 'measure', qubits: [0, 1] }
      ],
      parameters: {},
      measurements: [{ qubits: [0, 1], classical_bits: [0, 1] }]
    });
  };
  
  const handleExecuteJob = () => {
    if (!selectedAlgorithm) {
      Alert.alert('Error', 'Por favor selecciona un algoritmo.');
      return;
    }
    
    executeJobMutation.mutate({
      algorithm: selectedAlgorithm,
      backend: selectedBackend,
      shots: parseInt(shots, 10),
      parameters: jobParameters ? JSON.parse(jobParameters) : {}
    });
  };
  
  const handleTrainModel = () => {
    if (!modelName.trim()) {
      Alert.alert('Error', 'Por favor ingresa un nombre para el modelo.');
      return;
    }
    
    trainModelMutation.mutate({
      name: modelName,
      model_type: modelType,
      num_qubits: parseInt(modelQubits, 10),
      num_layers: parseInt(numLayers, 10),
      parameters: {},
      training_data: {
        features: [[0, 1], [1, 0], [1, 1], [0, 0]],
        labels: [1, 1, 0, 0]
      }
    });
  };
  
  const handleSolveOptimization = () => {
    if (!objectiveFunction.trim()) {
      Alert.alert('Error', 'Por favor ingresa la función objetivo.');
      return;
    }
    
    solveOptimizationMutation.mutate({
      problem_type: problemType,
      objective_function: objectiveFunction,
      constraints: constraints.split('\n').filter(c => c.trim()),
      variables: variables.split(',').map(v => v.trim()).filter(v => v),
      num_qubits: parseInt(modelQubits, 10),
      algorithm: 'qaoa',
      backend: selectedBackend
    });
  };
  
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Quantum AI</Text>
      <Text style={styles.subtitle}>Computación Cuántica e Inteligencia Artificial</Text>
      
      {/* Statistics */}
      {statistics && (
        <View style={styles.statisticsContainer}>
          <Text style={styles.sectionTitle}>Estadísticas del Sistema</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_quantum_circuits || 0}</Text>
              <Text style={styles.statLabel}>Circuitos</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_quantum_jobs || 0}</Text>
              <Text style={styles.statLabel}>Trabajos</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_quantum_models || 0}</Text>
              <Text style={styles.statLabel}>Modelos ML</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_optimizations || 0}</Text>
              <Text style={styles.statLabel}>Optimizaciones</Text>
            </View>
          </View>
        </View>
      )}
      
      {/* Quick Actions */}
      <View style={styles.quickActions}>
        <Text style={styles.sectionTitle}>Acciones Rápidas</Text>
        <View style={styles.actionButtons}>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#4CAF50' }]}
            onPress={() => setShowCircuitModal(true)}
          >
            <Text style={styles.actionButtonText}>Crear Circuito</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#2196F3' }]}
            onPress={() => setShowJobModal(true)}
          >
            <Text style={styles.actionButtonText}>Ejecutar Trabajo</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#FF9800' }]}
            onPress={() => setShowMLModal(true)}
          >
            <Text style={styles.actionButtonText}>Entrenar ML</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#9C27B0' }]}
            onPress={() => setShowOptimizationModal(true)}
          >
            <Text style={styles.actionButtonText}>Optimizar</Text>
          </TouchableOpacity>
        </View>
      </View>
      
      {/* Available Backends */}
      <View style={styles.backendsSection}>
        <Text style={styles.sectionTitle}>Backends Cuánticos Disponibles</Text>
        {isLoadingBackends && <ActivityIndicator size="large" color="#0000ff" style={styles.activityIndicator} />}
        {availableBackends && availableBackends.length > 0 ? (
          <FlatList
            data={availableBackends}
            renderItem={({ item }) => (
              <View style={styles.backendCard}>
                <Text style={styles.backendName}>{item.name}</Text>
                <Text style={styles.backendInfo}>
                  {item.type} • {item.qubits} qubits • {item.status}
                </Text>
              </View>
            )}
            keyExtractor={(item) => item.name}
            style={styles.backendsList}
            scrollEnabled={false}
          />
        ) : (
          <Text style={styles.noDataText}>No hay backends disponibles.</Text>
        )}
      </View>
      
      {/* Job Status */}
      {jobStatus && (
        <View style={styles.statusContainer}>
          <Text style={styles.sectionTitle}>Estado del Trabajo Cuántico</Text>
          <View style={styles.statusCard}>
            <Text style={styles.statusTitle}>Trabajo: {jobStatus.job_id}</Text>
            <Text style={styles.statusInfo}>Algoritmo: {jobStatus.algorithm}</Text>
            <Text style={styles.statusInfo}>Backend: {jobStatus.backend}</Text>
            <Text style={styles.statusInfo}>Shots: {jobStatus.shots}</Text>
            <Text style={[styles.statusInfo, { color: jobStatus.status === 'completed' ? '#4CAF50' : '#FF9800' }]}>
              Estado: {jobStatus.status}
            </Text>
            {jobStatus.execution_time && (
              <Text style={styles.statusInfo}>Tiempo: {jobStatus.execution_time}s</Text>
            )}
            {jobStatus.result && (
              <Text style={styles.statusInfo}>
                Resultado: {JSON.stringify(jobStatus.result).substring(0, 100)}...
              </Text>
            )}
          </View>
        </View>
      )}
      
      {/* Model Info */}
      {modelInfo && (
        <View style={styles.statusContainer}>
          <Text style={styles.sectionTitle}>Información del Modelo ML</Text>
          <View style={styles.statusCard}>
            <Text style={styles.statusTitle}>Modelo: {modelInfo.name}</Text>
            <Text style={styles.statusInfo}>Tipo: {modelInfo.model_type}</Text>
            <Text style={styles.statusInfo}>Qubits: {modelInfo.num_qubits}</Text>
            <Text style={styles.statusInfo}>Capas: {modelInfo.num_layers}</Text>
            <Text style={styles.statusInfo}>Precisión: {(modelInfo.accuracy * 100).toFixed(1)}%</Text>
            <Text style={styles.statusInfo}>Pérdida: {modelInfo.loss.toFixed(4)}</Text>
            <Text style={[styles.statusInfo, { color: modelInfo.is_trained ? '#4CAF50' : '#FF9800' }]}>
              Estado: {modelInfo.is_trained ? 'Entrenado' : 'Entrenando'}
            </Text>
          </View>
        </View>
      )}
      
      {/* Optimization Result */}
      {optimizationResult && (
        <View style={styles.statusContainer}>
          <Text style={styles.sectionTitle}>Resultado de Optimización</Text>
          <View style={styles.statusCard}>
            <Text style={styles.statusTitle}>Optimización: {optimizationResult.optimization_id}</Text>
            <Text style={styles.statusInfo}>Tipo: {optimizationResult.problem_type}</Text>
            <Text style={styles.statusInfo}>Algoritmo: {optimizationResult.algorithm}</Text>
            <Text style={styles.statusInfo}>Qubits: {optimizationResult.num_qubits}</Text>
            {optimizationResult.optimal_value && (
              <Text style={styles.statusInfo}>Valor Óptimo: {optimizationResult.optimal_value.toFixed(4)}</Text>
            )}
            {optimizationResult.solution && (
              <Text style={styles.statusInfo}>
                Solución: {JSON.stringify(optimizationResult.solution).substring(0, 100)}...
              </Text>
            )}
          </View>
        </View>
      )}
      
      {/* Circuit Creation Modal */}
      <Modal
        visible={showCircuitModal}
        animationType="slide"
        onRequestClose={() => setShowCircuitModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Crear Circuito Cuántico</Text>
            <Button title="Cerrar" onPress={() => setShowCircuitModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Nombre del Circuito:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: Circuito de Grover"
              value={circuitName}
              onChangeText={setCircuitName}
            />
            
            <Text style={styles.label}>Descripción:</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Descripción del circuito cuántico..."
              value={circuitDescription}
              onChangeText={setCircuitDescription}
              multiline
              numberOfLines={3}
            />
            
            <Text style={styles.label}>Número de Qubits:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={numQubits}
                onValueChange={setNumQubits}
                style={styles.picker}
              >
                <Picker.Item label="2" value="2" />
                <Picker.Item label="3" value="3" />
                <Picker.Item label="4" value="4" />
                <Picker.Item label="5" value="5" />
                <Picker.Item label="6" value="6" />
                <Picker.Item label="7" value="7" />
                <Picker.Item label="8" value="8" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Backend:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={selectedBackend}
                onValueChange={setSelectedBackend}
                style={styles.picker}
              >
                <Picker.Item label="Qiskit" value="qiskit" />
                <Picker.Item label="Cirq" value="cirq" />
                <Picker.Item label="PennyLane" value="pennylane" />
                <Picker.Item label="Q# (QSharp)" value="qsharp" />
                <Picker.Item label="Amazon Braket" value="braket" />
              </Picker>
            </View>
            
            <Button
              title={createCircuitMutation.isLoading ? 'Creando...' : 'Crear Circuito'}
              onPress={handleCreateCircuit}
              disabled={createCircuitMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Job Execution Modal */}
      <Modal
        visible={showJobModal}
        animationType="slide"
        onRequestClose={() => setShowJobModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Ejecutar Trabajo Cuántico</Text>
            <Button title="Cerrar" onPress={() => setShowJobModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Algoritmo:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={selectedAlgorithm}
                onValueChange={setSelectedAlgorithm}
                style={styles.picker}
              >
                {algorithms.map((algo) => (
                  <Picker.Item key={algo.value} label={algo.label} value={algo.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Backend:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={selectedBackend}
                onValueChange={setSelectedBackend}
                style={styles.picker}
              >
                <Picker.Item label="Qiskit" value="qiskit" />
                <Picker.Item label="Cirq" value="cirq" />
                <Picker.Item label="PennyLane" value="pennylane" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Número de Shots:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={shots}
                onValueChange={setShots}
                style={styles.picker}
              >
                <Picker.Item label="256" value="256" />
                <Picker.Item label="512" value="512" />
                <Picker.Item label="1024" value="1024" />
                <Picker.Item label="2048" value="2048" />
                <Picker.Item label="4096" value="4096" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Parámetros (JSON):</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder='{"target": "111", "iterations": 2}'
              value={jobParameters}
              onChangeText={setJobParameters}
              multiline
              numberOfLines={3}
            />
            
            <Button
              title={executeJobMutation.isLoading ? 'Ejecutando...' : 'Ejecutar Trabajo'}
              onPress={handleExecuteJob}
              disabled={executeJobMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* ML Training Modal */}
      <Modal
        visible={showMLModal}
        animationType="slide"
        onRequestClose={() => setShowMLModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Entrenar Modelo ML Cuántico</Text>
            <Button title="Cerrar" onPress={() => setShowMLModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Nombre del Modelo:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: Clasificador Cuántico"
              value={modelName}
              onChangeText={setModelName}
            />
            
            <Text style={styles.label}>Tipo de Modelo:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={modelType}
                onValueChange={setModelType}
                style={styles.picker}
              >
                {modelTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Número de Qubits:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={modelQubits}
                onValueChange={setModelQubits}
                style={styles.picker}
              >
                <Picker.Item label="2" value="2" />
                <Picker.Item label="3" value="3" />
                <Picker.Item label="4" value="4" />
                <Picker.Item label="5" value="5" />
                <Picker.Item label="6" value="6" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Número de Capas:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={numLayers}
                onValueChange={setNumLayers}
                style={styles.picker}
              >
                <Picker.Item label="1" value="1" />
                <Picker.Item label="2" value="2" />
                <Picker.Item label="3" value="3" />
                <Picker.Item label="4" value="4" />
                <Picker.Item label="5" value="5" />
              </Picker>
            </View>
            
            <Button
              title={trainModelMutation.isLoading ? 'Entrenando...' : 'Entrenar Modelo'}
              onPress={handleTrainModel}
              disabled={trainModelMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Optimization Modal */}
      <Modal
        visible={showOptimizationModal}
        animationType="slide"
        onRequestClose={() => setShowOptimizationModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Resolver Optimización Cuántica</Text>
            <Button title="Cerrar" onPress={() => setShowOptimizationModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Tipo de Problema:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={problemType}
                onValueChange={setProblemType}
                style={styles.picker}
              >
                {problemTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Función Objetivo:</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Ej: x1^2 + x2^2 - 2*x1*x2"
              value={objectiveFunction}
              onChangeText={setObjectiveFunction}
              multiline
              numberOfLines={2}
            />
            
            <Text style={styles.label}>Restricciones (una por línea):</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="x1 + x2 <= 1&#10;x1 >= 0&#10;x2 >= 0"
              value={constraints}
              onChangeText={setConstraints}
              multiline
              numberOfLines={4}
            />
            
            <Text style={styles.label}>Variables (separadas por comas):</Text>
            <TextInput
              style={styles.input}
              placeholder="x1, x2, x3"
              value={variables}
              onChangeText={setVariables}
            />
            
            <Button
              title={solveOptimizationMutation.isLoading ? 'Resolviendo...' : 'Resolver Optimización'}
              onPress={handleSolveOptimization}
              disabled={solveOptimizationMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
    color: '#333',
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
    color: '#666',
  },
  statisticsContainer: {
    backgroundColor: 'white',
    padding: 16,
    marginBottom: 16,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#333',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  statItem: {
    width: '48%',
    alignItems: 'center',
    marginBottom: 16,
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2196F3',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  quickActions: {
    backgroundColor: 'white',
    padding: 16,
    marginBottom: 16,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  actionButtons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  actionButton: {
    width: '48%',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 12,
  },
  actionButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 14,
  },
  backendsSection: {
    backgroundColor: 'white',
    padding: 16,
    marginBottom: 16,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  backendsList: {
    maxHeight: 200,
  },
  backendCard: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#2196F3',
  },
  backendName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  backendInfo: {
    fontSize: 12,
    color: '#666',
  },
  statusContainer: {
    backgroundColor: 'white',
    padding: 16,
    marginBottom: 16,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statusCard: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#4CAF50',
  },
  statusTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  statusInfo: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  noDataText: {
    textAlign: 'center',
    fontSize: 16,
    color: '#666',
    fontStyle: 'italic',
    marginTop: 20,
  },
  activityIndicator: {
    marginTop: 16,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'white',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  modalContent: {
    flex: 1,
    padding: 16,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: '#333',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    fontSize: 16,
    backgroundColor: '#f9f9f9',
  },
  textArea: {
    height: 80,
    textAlignVertical: 'top',
  },
  pickerContainer: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    marginBottom: 16,
    backgroundColor: '#f9f9f9',
  },
  picker: {
    height: 50,
  },
});

export default QuantumAIScreen;


