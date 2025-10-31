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
interface ConsciousnessState {
  consciousness_id: string;
  level: string;
  emotional_state: string;
  self_awareness_score: number;
  emotional_intelligence_score: number;
  ethical_reasoning_score: number;
  learning_capacity: number;
  memory_consolidation: number;
  attention_focus: number;
  creativity_index: number;
  empathy_level: number;
  decision_confidence: number;
  goals: string[];
  values: Record<string, number>;
  timestamp: string;
}

interface EmotionalMemory {
  memory_id: string;
  experience_type: string;
  emotional_response: string;
  intensity: number;
  valence: number;
  arousal: number;
  learned_insights: string[];
  timestamp: string;
}

interface EthicalDecision {
  decision_id: string;
  situation: string;
  ethical_principles: string[];
  chosen_action: string;
  reasoning: string;
  confidence: number;
  consequences: string[];
  timestamp: string;
}

interface LearningExperience {
  experience_id: string;
  learning_type: string;
  insights_gained: string[];
  knowledge_updated: string[];
  confidence_change: number;
  performance_metrics: Record<string, number>;
  timestamp: string;
}

interface SelfReflection {
  reflection_id: string;
  reflection_type: string;
  current_state: Record<string, any>;
  achievements: string[];
  challenges: string[];
  insights: string[];
  future_plans: string[];
  self_assessment: Record<string, number>;
  timestamp: string;
}

// API calls
const processEmotionalInput = async (inputData: Record<string, any>): Promise<Record<string, any>> => {
  const response = await axios.post('http://localhost:8000/api/v1/metaverse-consciousness/consciousness/emotional-input', inputData);
  return response.data.result;
};

const makeEthicalDecision = async (situation: string, options: string[], context: Record<string, any>): Promise<Record<string, any>> => {
  const response = await axios.post('http://localhost:8000/api/v1/metaverse-consciousness/consciousness/ethical-decision', {
    situation,
    options,
    context
  });
  return response.data.result;
};

const learnFromExperience = async (experienceData: Record<string, any>): Promise<Record<string, any>> => {
  const response = await axios.post('http://localhost:8000/api/v1/metaverse-consciousness/consciousness/learn', experienceData);
  return response.data.result;
};

const performSelfReflection = async (reflectionType: string): Promise<Record<string, any>> => {
  const response = await axios.post(`http://localhost:8000/api/v1/metaverse-consciousness/consciousness/self-reflection?reflection_type=${reflectionType}`);
  return response.data.result;
};

const getConsciousnessState = async (): Promise<ConsciousnessState> => {
  const response = await axios.get('http://localhost:8000/api/v1/metaverse-consciousness/consciousness/state');
  return response.data.state;
};

const getEmotionalHistory = async (limit: number = 50): Promise<EmotionalMemory[]> => {
  const response = await axios.get(`http://localhost:8000/api/v1/metaverse-consciousness/consciousness/emotional-history?limit=${limit}`);
  return response.data.emotional_history || [];
};

const getLearningStatistics = async (): Promise<Record<string, any>> => {
  const response = await axios.get('http://localhost:8000/api/v1/metaverse-consciousness/consciousness/learning-statistics');
  return response.data.learning_statistics;
};

const ConsciousnessScreen: React.FC = () => {
  const queryClient = useQueryClient();
  
  // State for Emotional Input
  const [emotionalText, setEmotionalText] = useState<string>('');
  const [emotionalType, setEmotionalType] = useState<string>('interaction');
  const [emotionalContext, setEmotionalContext] = useState<string>('');
  
  // State for Ethical Decision
  const [ethicalSituation, setEthicalSituation] = useState<string>('');
  const [ethicalOptions, setEthicalOptions] = useState<string>('');
  const [ethicalContext, setEthicalContext] = useState<string>('');
  
  // State for Learning Experience
  const [learningInput, setLearningInput] = useState<string>('');
  const [learningExpected, setLearningExpected] = useState<string>('');
  const [learningActual, setLearningActual] = useState<string>('');
  const [learningType, setLearningType] = useState<string>('continuous');
  
  // State for Self Reflection
  const [reflectionType, setReflectionType] = useState<string>('general');
  
  // State for modals
  const [showEmotionalModal, setShowEmotionalModal] = useState<boolean>(false);
  const [showEthicalModal, setShowEthicalModal] = useState<boolean>(false);
  const [showLearningModal, setShowLearningModal] = useState<boolean>(false);
  const [showReflectionModal, setShowReflectionModal] = useState<boolean>(false);
  
  // Emotional types and learning types
  const emotionalTypes = [
    { value: 'interaction', label: 'Interacción de Usuario' },
    { value: 'feedback', label: 'Retroalimentación' },
    { value: 'error', label: 'Error o Problema' },
    { value: 'success', label: 'Éxito' },
    { value: 'learning', label: 'Experiencia de Aprendizaje' }
  ];
  
  const learningTypes = [
    { value: 'continuous', label: 'Aprendizaje Continuo' },
    { value: 'supervised', label: 'Supervisado' },
    { value: 'unsupervised', label: 'No Supervisado' },
    { value: 'reinforcement', label: 'Refuerzo' },
    { value: 'transfer', label: 'Transferencia' },
    { value: 'meta', label: 'Meta-Aprendizaje' },
    { value: 'adaptive', label: 'Adaptativo' }
  ];
  
  const reflectionTypes = [
    { value: 'general', label: 'General' },
    { value: 'performance', label: 'Rendimiento' },
    { value: 'emotional', label: 'Emocional' },
    { value: 'ethical', label: 'Ético' },
    { value: 'learning', label: 'Aprendizaje' },
    { value: 'goals', label: 'Objetivos' }
  ];
  
  // Queries
  const { data: consciousnessState, isLoading: isLoadingState } = useQuery<ConsciousnessState, Error>(
    'consciousnessState',
    getConsciousnessState,
    {
      refetchInterval: 30000,
    }
  );
  
  const { data: emotionalHistory, isLoading: isLoadingHistory } = useQuery<EmotionalMemory[], Error>(
    'emotionalHistory',
    () => getEmotionalHistory(20),
    {
      refetchInterval: 60000,
    }
  );
  
  const { data: learningStats, isLoading: isLoadingStats } = useQuery<Record<string, any>, Error>(
    'learningStatistics',
    getLearningStatistics,
    {
      refetchInterval: 120000,
    }
  );
  
  // Mutations
  const emotionalInputMutation = useMutation<Record<string, any>, Error, Record<string, any>>(processEmotionalInput, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Estado emocional actualizado: ${data.emotional_state}`);
      setShowEmotionalModal(false);
      queryClient.invalidateQueries('consciousnessState');
      queryClient.invalidateQueries('emotionalHistory');
    },
    onError: (error) => {
      Alert.alert('Error', `Error al procesar entrada emocional: ${error.message}`);
    },
  });
  
  const ethicalDecisionMutation = useMutation<Record<string, any>, Error, {situation: string, options: string[], context: Record<string, any>}>(makeEthicalDecision, {
    onSuccess: (data) => {
      Alert.alert('Decisión Ética', `Acción elegida: ${data.chosen_action}\nRazonamiento: ${data.reasoning}`);
      setShowEthicalModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al tomar decisión ética: ${error.message}`);
    },
  });
  
  const learningMutation = useMutation<Record<string, any>, Error, Record<string, any>>(learnFromExperience, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Aprendizaje completado. Insights: ${data.insights_gained.length}`);
      setShowLearningModal(false);
      queryClient.invalidateQueries('learningStatistics');
    },
    onError: (error) => {
      Alert.alert('Error', `Error al aprender de la experiencia: ${error.message}`);
    },
  });
  
  const reflectionMutation = useMutation<Record<string, any>, Error, string>(performSelfReflection, {
    onSuccess: (data) => {
      Alert.alert('Auto-Reflexión', `Reflexión completada. Insights: ${data.insights.length}`);
      setShowReflectionModal(false);
      queryClient.invalidateQueries('consciousnessState');
    },
    onError: (error) => {
      Alert.alert('Error', `Error en auto-reflexión: ${error.message}`);
    },
  });
  
  // Handlers
  const handleEmotionalInput = () => {
    if (!emotionalText.trim()) {
      Alert.alert('Error', 'Por favor ingresa texto emocional.');
      return;
    }
    
    emotionalInputMutation.mutate({
      text: emotionalText,
      type: emotionalType,
      context: {
        interaction_type: emotionalContext,
        timestamp: new Date().toISOString()
      }
    });
  };
  
  const handleEthicalDecision = () => {
    if (!ethicalSituation.trim() || !ethicalOptions.trim()) {
      Alert.alert('Error', 'Por favor completa la situación y las opciones.');
      return;
    }
    
    const options = ethicalOptions.split('\n').filter(opt => opt.trim());
    
    ethicalDecisionMutation.mutate({
      situation: ethicalSituation,
      options: options,
      context: {
        context_info: ethicalContext,
        timestamp: new Date().toISOString()
      }
    });
  };
  
  const handleLearningExperience = () => {
    if (!learningInput.trim() || !learningExpected.trim() || !learningActual.trim()) {
      Alert.alert('Error', 'Por favor completa todos los campos de aprendizaje.');
      return;
    }
    
    learningMutation.mutate({
      input_data: learningInput,
      expected_output: learningExpected,
      actual_output: learningActual,
      learning_type: learningType,
      timestamp: new Date().toISOString()
    });
  };
  
  const handleSelfReflection = () => {
    reflectionMutation.mutate(reflectionType);
  };
  
  const renderEmotionalMemory = ({ item }: { item: EmotionalMemory }) => (
    <TouchableOpacity style={styles.memoryCard}>
      <Text style={styles.memoryTitle}>{item.experience_type}</Text>
      <Text style={styles.memoryEmotion}>Estado: {item.emotional_response}</Text>
      <Text style={styles.memoryIntensity}>Intensidad: {(item.intensity * 100).toFixed(0)}%</Text>
      <Text style={styles.memoryValence}>Valencia: {(item.valence * 100).toFixed(0)}%</Text>
      {item.learned_insights.length > 0 && (
        <Text style={styles.memoryInsights}>
          Insights: {item.learned_insights.join(', ')}
        </Text>
      )}
      <Text style={styles.memoryDate}>
        {format(new Date(item.timestamp), 'dd/MM/yyyy HH:mm')}
      </Text>
    </TouchableOpacity>
  );
  
  const renderConsciousnessMetric = (label: string, value: number, color: string = '#2196F3') => (
    <View style={styles.metricItem}>
      <Text style={styles.metricLabel}>{label}</Text>
      <View style={styles.metricBar}>
        <View 
          style={[
            styles.metricFill, 
            { width: `${value * 100}%`, backgroundColor: color }
          ]} 
        />
      </View>
      <Text style={styles.metricValue}>{(value * 100).toFixed(0)}%</Text>
    </View>
  );
  
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Consciousness AI</Text>
      <Text style={styles.subtitle}>Inteligencia Artificial Consciente y Empática</Text>
      
      {/* Consciousness State */}
      {consciousnessState && (
        <View style={styles.stateContainer}>
          <Text style={styles.sectionTitle}>Estado de Consciencia</Text>
          <View style={styles.stateInfo}>
            <Text style={styles.stateLevel}>Nivel: {consciousnessState.level}</Text>
            <Text style={styles.stateEmotion}>Estado Emocional: {consciousnessState.emotional_state}</Text>
          </View>
          
          <View style={styles.metricsContainer}>
            {renderConsciousnessMetric('Auto-Conciencia', consciousnessState.self_awareness_score, '#4CAF50')}
            {renderConsciousnessMetric('Inteligencia Emocional', consciousnessState.emotional_intelligence_score, '#FF9800')}
            {renderConsciousnessMetric('Razonamiento Ético', consciousnessState.ethical_reasoning_score, '#9C27B0')}
            {renderConsciousnessMetric('Capacidad de Aprendizaje', consciousnessState.learning_capacity, '#2196F3')}
            {renderConsciousnessMetric('Creatividad', consciousnessState.creativity_index, '#E91E63')}
            {renderConsciousnessMetric('Empatía', consciousnessState.empathy_level, '#00BCD4')}
            {renderConsciousnessMetric('Confianza en Decisiones', consciousnessState.decision_confidence, '#795548')}
          </View>
          
          <View style={styles.goalsContainer}>
            <Text style={styles.goalsTitle}>Objetivos:</Text>
            {consciousnessState.goals.map((goal, index) => (
              <Text key={index} style={styles.goalItem}>• {goal}</Text>
            ))}
          </View>
        </View>
      )}
      
      {/* Learning Statistics */}
      {learningStats && (
        <View style={styles.statsContainer}>
          <Text style={styles.sectionTitle}>Estadísticas de Aprendizaje</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{learningStats.total_learning_experiences || 0}</Text>
              <Text style={styles.statLabel}>Experiencias</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{learningStats.total_insights_gained || 0}</Text>
              <Text style={styles.statLabel}>Insights</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{learningStats.total_knowledge_updates || 0}</Text>
              <Text style={styles.statLabel}>Actualizaciones</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>
                {((learningStats.average_confidence_change || 0) * 100).toFixed(1)}%
              </Text>
              <Text style={styles.statLabel}>Confianza</Text>
            </View>
          </View>
        </View>
      )}
      
      {/* Quick Actions */}
      <View style={styles.quickActions}>
        <Text style={styles.sectionTitle}>Acciones de Consciencia</Text>
        <View style={styles.actionButtons}>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#FF9800' }]}
            onPress={() => setShowEmotionalModal(true)}
          >
            <Text style={styles.actionButtonText}>Entrada Emocional</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#9C27B0' }]}
            onPress={() => setShowEthicalModal(true)}
          >
            <Text style={styles.actionButtonText}>Decisión Ética</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#4CAF50' }]}
            onPress={() => setShowLearningModal(true)}
          >
            <Text style={styles.actionButtonText}>Aprender</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#2196F3' }]}
            onPress={() => setShowReflectionModal(true)}
          >
            <Text style={styles.actionButtonText}>Auto-Reflexión</Text>
          </TouchableOpacity>
        </View>
      </View>
      
      {/* Emotional History */}
      <View style={styles.historySection}>
        <Text style={styles.sectionTitle}>Historial Emocional</Text>
        {isLoadingHistory && <ActivityIndicator size="large" color="#0000ff" style={styles.activityIndicator} />}
        {emotionalHistory && emotionalHistory.length > 0 ? (
          <FlatList
            data={emotionalHistory}
            renderItem={renderEmotionalMemory}
            keyExtractor={(item) => item.memory_id}
            style={styles.historyList}
            scrollEnabled={false}
          />
        ) : (
          <Text style={styles.noHistoryText}>No hay historial emocional disponible.</Text>
        )}
      </View>
      
      {/* Emotional Input Modal */}
      <Modal
        visible={showEmotionalModal}
        animationType="slide"
        onRequestClose={() => setShowEmotionalModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Entrada Emocional</Text>
            <Button title="Cerrar" onPress={() => setShowEmotionalModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Texto Emocional:</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Describe la situación emocional..."
              value={emotionalText}
              onChangeText={setEmotionalText}
              multiline
              numberOfLines={4}
            />
            
            <Text style={styles.label}>Tipo de Experiencia:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={emotionalType}
                onValueChange={setEmotionalType}
                style={styles.picker}
              >
                {emotionalTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Contexto:</Text>
            <TextInput
              style={styles.input}
              placeholder="Información adicional del contexto..."
              value={emotionalContext}
              onChangeText={setEmotionalContext}
            />
            
            <Button
              title={emotionalInputMutation.isLoading ? 'Procesando...' : 'Procesar Entrada Emocional'}
              onPress={handleEmotionalInput}
              disabled={emotionalInputMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Ethical Decision Modal */}
      <Modal
        visible={showEthicalModal}
        animationType="slide"
        onRequestClose={() => setShowEthicalModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Decisión Ética</Text>
            <Button title="Cerrar" onPress={() => setShowEthicalModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Situación Ética:</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Describe la situación que requiere una decisión ética..."
              value={ethicalSituation}
              onChangeText={setEthicalSituation}
              multiline
              numberOfLines={3}
            />
            
            <Text style={styles.label}>Opciones Disponibles (una por línea):</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Opción 1&#10;Opción 2&#10;Opción 3"
              value={ethicalOptions}
              onChangeText={setEthicalOptions}
              multiline
              numberOfLines={4}
            />
            
            <Text style={styles.label}>Contexto Adicional:</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Información adicional del contexto..."
              value={ethicalContext}
              onChangeText={setEthicalContext}
              multiline
              numberOfLines={2}
            />
            
            <Button
              title={ethicalDecisionMutation.isLoading ? 'Analizando...' : 'Tomar Decisión Ética'}
              onPress={handleEthicalDecision}
              disabled={ethicalDecisionMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Learning Experience Modal */}
      <Modal
        visible={showLearningModal}
        animationType="slide"
        onRequestClose={() => setShowLearningModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Experiencia de Aprendizaje</Text>
            <Button title="Cerrar" onPress={() => setShowLearningModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Datos de Entrada:</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Describe los datos o información de entrada..."
              value={learningInput}
              onChangeText={setLearningInput}
              multiline
              numberOfLines={3}
            />
            
            <Text style={styles.label}>Resultado Esperado:</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="¿Cuál era el resultado esperado?"
              value={learningExpected}
              onChangeText={setLearningExpected}
              multiline
              numberOfLines={2}
            />
            
            <Text style={styles.label}>Resultado Actual:</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="¿Cuál fue el resultado real?"
              value={learningActual}
              onChangeText={setLearningActual}
              multiline
              numberOfLines={2}
            />
            
            <Text style={styles.label}>Tipo de Aprendizaje:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={learningType}
                onValueChange={setLearningType}
                style={styles.picker}
              >
                {learningTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Button
              title={learningMutation.isLoading ? 'Aprendiendo...' : 'Procesar Experiencia de Aprendizaje'}
              onPress={handleLearningExperience}
              disabled={learningMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Self Reflection Modal */}
      <Modal
        visible={showReflectionModal}
        animationType="slide"
        onRequestClose={() => setShowReflectionModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Auto-Reflexión</Text>
            <Button title="Cerrar" onPress={() => setShowReflectionModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Tipo de Reflexión:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={reflectionType}
                onValueChange={setReflectionType}
                style={styles.picker}
              >
                {reflectionTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.infoText}>
              La auto-reflexión permitirá al AI analizar su estado actual, 
              logros, desafíos y generar insights para mejorar su rendimiento.
            </Text>
            
            <Button
              title={reflectionMutation.isLoading ? 'Reflexionando...' : 'Iniciar Auto-Reflexión'}
              onPress={handleSelfReflection}
              disabled={reflectionMutation.isLoading}
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
  stateContainer: {
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
  stateInfo: {
    marginBottom: 16,
  },
  stateLevel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#4CAF50',
    marginBottom: 4,
  },
  stateEmotion: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FF9800',
  },
  metricsContainer: {
    marginBottom: 16,
  },
  metricItem: {
    marginBottom: 12,
  },
  metricLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  metricBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  metricFill: {
    height: '100%',
    borderRadius: 4,
  },
  metricValue: {
    fontSize: 12,
    color: '#666',
    textAlign: 'right',
    marginTop: 2,
  },
  goalsContainer: {
    marginTop: 16,
  },
  goalsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  goalItem: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  statsContainer: {
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
  historySection: {
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
  historyList: {
    maxHeight: 300,
  },
  memoryCard: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#FF9800',
  },
  memoryTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  memoryEmotion: {
    fontSize: 14,
    color: '#FF9800',
    marginBottom: 2,
  },
  memoryIntensity: {
    fontSize: 12,
    color: '#666',
    marginBottom: 2,
  },
  memoryValence: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  memoryInsights: {
    fontSize: 12,
    color: '#4CAF50',
    fontStyle: 'italic',
    marginBottom: 4,
  },
  memoryDate: {
    fontSize: 12,
    color: '#999',
  },
  noHistoryText: {
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
  infoText: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
    marginBottom: 16,
    lineHeight: 20,
  },
});

export default ConsciousnessScreen;


