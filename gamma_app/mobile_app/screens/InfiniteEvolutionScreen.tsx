import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, ActivityIndicator, Alert, Switch } from 'react-native';
import axios from 'axios';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Picker } from '@react-native-picker/picker';
import { format } from 'date-fns';

// API Base URL
const API_BASE_URL = 'http://localhost:8000/api/v1';

// Interface definitions
interface EvolutionEntity {
  entity_id: string;
  name: string;
  evolution_stage: string;
  evolution_type: string;
  growth_pattern: string;
  evolution_rate: number;
  complexity_level: number;
  adaptation_capacity: number;
  mutation_rate: number;
  selection_pressure: number;
  fitness_score: number;
  is_evolving: boolean;
  last_evolution?: string;
  evolution_history_count: number;
}

interface EvolutionEvent {
  event_id: string;
  entity_id: string;
  event_type: string;
  evolution_stage: string;
  evolution_type: string;
  changes: Record<string, number>;
  success: boolean;
  side_effects: string[];
  duration: number;
  timestamp: string;
}

interface InfiniteGrowth {
  growth_id: string;
  entity_id: string;
  growth_type: string;
  growth_rate: number;
  current_value: number;
  target_value: number;
  is_infinite: boolean;
  started_at: string;
}

interface CreationEntity {
  entity_id: string;
  name: string;
  creation_type: string;
  divine_power: string;
  manifestation_level: string;
  creation_power: number;
  divine_energy: number;
  omnipotence_level: number;
  omniscience_level: number;
  omnipresence_level: number;
  creation_capacity: number;
  is_creating: boolean;
  last_creation?: string;
  creation_history_count: number;
}

// API functions
const createEvolutionEntity = async (entityInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/infinite-omnipotent/evolution/entities/create`, entityInfo);
  return response.data;
};

const initiateEvolutionEvent = async (eventInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/infinite-omnipotent/evolution/events/initiate`, eventInfo);
  return response.data;
};

const startInfiniteGrowth = async (growthInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/infinite-omnipotent/evolution/growth/start`, growthInfo);
  return response.data;
};

const createCreationEntity = async (entityInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/infinite-omnipotent/creation/entities/create`, entityInfo);
  return response.data;
};

const initiateCreationEvent = async (eventInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/infinite-omnipotent/creation/events/initiate`, eventInfo);
  return response.data;
};

const getEvolutionEntityStatus = async (entityId: string): Promise<EvolutionEntity> => {
  const response = await axios.get(`${API_BASE_URL}/infinite-omnipotent/evolution/entities/${entityId}/status`);
  return response.data;
};

const getEvolutionProgress = async (eventId: string) => {
  const response = await axios.get(`${API_BASE_URL}/infinite-omnipotent/evolution/events/${eventId}/progress`);
  return response.data;
};

const getGrowthStatus = async (growthId: string): Promise<InfiniteGrowth> => {
  const response = await axios.get(`${API_BASE_URL}/infinite-omnipotent/evolution/growth/${growthId}/status`);
  return response.data;
};

const getCreationEntityStatus = async (entityId: string): Promise<CreationEntity> => {
  const response = await axios.get(`${API_BASE_URL}/infinite-omnipotent/creation/entities/${entityId}/status`);
  return response.data;
};

const getCreationProgress = async (eventId: string) => {
  const response = await axios.get(`${API_BASE_URL}/infinite-omnipotent/creation/events/${eventId}/progress`);
  return response.data;
};

const getEvolutionStatistics = async () => {
  const response = await axios.get(`${API_BASE_URL}/infinite-omnipotent/evolution/statistics`);
  return response.data;
};

const getCreationStatistics = async () => {
  const response = await axios.get(`${API_BASE_URL}/infinite-omnipotent/creation/statistics`);
  return response.data;
};

const InfiniteEvolutionScreen: React.FC = () => {
  const queryClient = useQueryClient();
  
  // State for evolution entity creation
  const [evolutionEntityName, setEvolutionEntityName] = useState<string>('');
  const [evolutionStage, setEvolutionStage] = useState<string>('basic');
  const [evolutionType, setEvolutionType] = useState<string>('biological');
  const [growthPattern, setGrowthPattern] = useState<string>('exponential');
  const [evolutionRate, setEvolutionRate] = useState<string>('0.1');
  const [complexityLevel, setComplexityLevel] = useState<string>('0.5');
  
  // State for evolution event
  const [eventEntityId, setEventEntityId] = useState<string>('');
  const [eventType, setEventType] = useState<string>('natural_selection');
  const [eventEvolutionStage, setEventEvolutionStage] = useState<string>('intermediate');
  const [eventEvolutionType, setEventEvolutionType] = useState<string>('biological');
  const [eventDuration, setEventDuration] = useState<string>('60');
  
  // State for infinite growth
  const [growthEntityId, setGrowthEntityId] = useState<string>('');
  const [growthType, setGrowthType] = useState<string>('exponential');
  const [growthRate, setGrowthRate] = useState<string>('0.1');
  const [currentValue, setCurrentValue] = useState<string>('1.0');
  const [targetValue, setTargetValue] = useState<string>('1000');
  const [isInfinite, setIsInfinite] = useState<boolean>(true);
  
  // State for creation entity
  const [creationEntityName, setCreationEntityName] = useState<string>('');
  const [creationType, setCreationType] = useState<string>('matter');
  const [divinePower, setDivinePower] = useState<string>('creation');
  const [manifestationLevel, setManifestationLevel] = useState<string>('thought');
  const [creationPower, setCreationPower] = useState<string>('0.5');
  const [divineEnergy, setDivineEnergy] = useState<string>('0.5');
  
  // State for creation event
  const [creationEventEntityId, setCreationEventEntityId] = useState<string>('');
  const [creationEventType, setCreationEventType] = useState<string>('matter');
  const [creationEventDivinePower, setCreationEventDivinePower] = useState<string>('creation');
  const [creationEventManifestationLevel, setCreationEventManifestationLevel] = useState<string>('manifestation');
  const [targetCreation, setTargetCreation] = useState<string>('');
  const [creationEventDuration, setCreationEventDuration] = useState<string>('60');
  
  // State for display
  const [selectedEvolutionEntityId, setSelectedEvolutionEntityId] = useState<string>('');
  const [selectedEvolutionEventId, setSelectedEvolutionEventId] = useState<string>('');
  const [selectedGrowthId, setSelectedGrowthId] = useState<string>('');
  const [selectedCreationEntityId, setSelectedCreationEntityId] = useState<string>('');
  const [selectedCreationEventId, setSelectedCreationEventId] = useState<string>('');
  const [evolutionEntityStatus, setEvolutionEntityStatus] = useState<EvolutionEntity | null>(null);
  const [evolutionProgress, setEvolutionProgress] = useState<any>(null);
  const [growthStatus, setGrowthStatus] = useState<InfiniteGrowth | null>(null);
  const [creationEntityStatus, setCreationEntityStatus] = useState<CreationEntity | null>(null);
  const [creationProgress, setCreationProgress] = useState<any>(null);
  const [evolutionStats, setEvolutionStats] = useState<any>(null);
  const [creationStats, setCreationStats] = useState<any>(null);

  // Queries
  const { data: evolutionStatistics, isLoading: isLoadingEvolutionStats } = useQuery(
    'evolutionStatistics',
    getEvolutionStatistics,
    { refetchInterval: 5000 }
  );

  const { data: creationStatistics, isLoading: isLoadingCreationStats } = useQuery(
    'creationStatistics',
    getCreationStatistics,
    { refetchInterval: 5000 }
  );

  // Mutations
  const createEvolutionEntityMutation = useMutation(createEvolutionEntity, {
    onSuccess: (data) => {
      Alert.alert('Success', `Evolution entity created: ${data.entity_id}`);
      queryClient.invalidateQueries('evolutionStatistics');
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to create evolution entity: ${error.message}`);
    },
  });

  const initiateEvolutionEventMutation = useMutation(initiateEvolutionEvent, {
    onSuccess: (data) => {
      Alert.alert('Success', `Evolution event initiated: ${data.event_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to initiate evolution event: ${error.message}`);
    },
  });

  const startInfiniteGrowthMutation = useMutation(startInfiniteGrowth, {
    onSuccess: (data) => {
      Alert.alert('Success', `Infinite growth started: ${data.growth_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to start infinite growth: ${error.message}`);
    },
  });

  const createCreationEntityMutation = useMutation(createCreationEntity, {
    onSuccess: (data) => {
      Alert.alert('Success', `Creation entity created: ${data.entity_id}`);
      queryClient.invalidateQueries('creationStatistics');
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to create creation entity: ${error.message}`);
    },
  });

  const initiateCreationEventMutation = useMutation(initiateCreationEvent, {
    onSuccess: (data) => {
      Alert.alert('Success', `Creation event initiated: ${data.event_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to initiate creation event: ${error.message}`);
    },
  });

  // Handlers
  const handleCreateEvolutionEntity = () => {
    if (!evolutionEntityName.trim()) {
      Alert.alert('Input Error', 'Please enter an entity name.');
      return;
    }

    const entityInfo = {
      name: evolutionEntityName,
      evolution_stage: evolutionStage,
      evolution_type: evolutionType,
      growth_pattern: growthPattern,
      evolution_rate: parseFloat(evolutionRate),
      complexity_level: parseFloat(complexityLevel),
      adaptation_capacity: 0.5,
      mutation_rate: 0.01,
      selection_pressure: 0.5,
      fitness_score: 0.5
    };

    createEvolutionEntityMutation.mutate(entityInfo);
  };

  const handleInitiateEvolutionEvent = () => {
    if (!eventEntityId.trim()) {
      Alert.alert('Input Error', 'Please enter an entity ID.');
      return;
    }

    const eventInfo = {
      entity_id: eventEntityId,
      event_type: eventType,
      evolution_stage: eventEvolutionStage,
      evolution_type: eventEvolutionType,
      changes: {
        complexity_level: 0.1,
        fitness_score: 0.05
      },
      duration: parseFloat(eventDuration)
    };

    initiateEvolutionEventMutation.mutate(eventInfo);
  };

  const handleStartInfiniteGrowth = () => {
    if (!growthEntityId.trim()) {
      Alert.alert('Input Error', 'Please enter an entity ID.');
      return;
    }

    const growthInfo = {
      entity_id: growthEntityId,
      growth_type: growthType,
      growth_rate: parseFloat(growthRate),
      current_value: parseFloat(currentValue),
      target_value: isInfinite ? Infinity : parseFloat(targetValue),
      is_infinite: isInfinite
    };

    startInfiniteGrowthMutation.mutate(growthInfo);
  };

  const handleCreateCreationEntity = () => {
    if (!creationEntityName.trim()) {
      Alert.alert('Input Error', 'Please enter an entity name.');
      return;
    }

    const entityInfo = {
      name: creationEntityName,
      creation_type: creationType,
      divine_power: divinePower,
      manifestation_level: manifestationLevel,
      creation_power: parseFloat(creationPower),
      divine_energy: parseFloat(divineEnergy),
      omnipotence_level: 0.5,
      omniscience_level: 0.5,
      omnipresence_level: 0.5,
      creation_capacity: 0.5
    };

    createCreationEntityMutation.mutate(entityInfo);
  };

  const handleInitiateCreationEvent = () => {
    if (!creationEventEntityId.trim() || !targetCreation.trim()) {
      Alert.alert('Input Error', 'Please enter an entity ID and target creation.');
      return;
    }

    const eventInfo = {
      entity_id: creationEventEntityId,
      creation_type: creationEventType,
      divine_power: creationEventDivinePower,
      manifestation_level: creationEventManifestationLevel,
      target_creation: targetCreation,
      creation_parameters: {
        size: 1.0,
        complexity: 0.5,
        power: 0.5
      },
      duration: parseFloat(creationEventDuration)
    };

    initiateCreationEventMutation.mutate(eventInfo);
  };

  const handleGetEvolutionEntityStatus = async () => {
    try {
      const status = await getEvolutionEntityStatus(selectedEvolutionEntityId);
      setEvolutionEntityStatus(status);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get evolution entity status: ${error.message}`);
    }
  };

  const handleGetEvolutionProgress = async () => {
    try {
      const progress = await getEvolutionProgress(selectedEvolutionEventId);
      setEvolutionProgress(progress);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get evolution progress: ${error.message}`);
    }
  };

  const handleGetGrowthStatus = async () => {
    try {
      const status = await getGrowthStatus(selectedGrowthId);
      setGrowthStatus(status);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get growth status: ${error.message}`);
    }
  };

  const handleGetCreationEntityStatus = async () => {
    try {
      const status = await getCreationEntityStatus(selectedCreationEntityId);
      setCreationEntityStatus(status);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get creation entity status: ${error.message}`);
    }
  };

  const handleGetCreationProgress = async () => {
    try {
      const progress = await getCreationProgress(selectedCreationEventId);
      setCreationProgress(progress);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get creation progress: ${error.message}`);
    }
  };

  useEffect(() => {
    if (evolutionStatistics) {
      setEvolutionStats(evolutionStatistics);
    }
  }, [evolutionStatistics]);

  useEffect(() => {
    if (creationStatistics) {
      setCreationStats(creationStatistics);
    }
  }, [creationStatistics]);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Infinite Evolution & Omnipotent Creation</Text>
      
      {/* Evolution Entity Creation Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Create Evolution Entity</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity Name"
          value={evolutionEntityName}
          onChangeText={setEvolutionEntityName}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Evolution Stage:</Text>
          <Picker
            selectedValue={evolutionStage}
            onValueChange={setEvolutionStage}
            style={styles.picker}
          >
            <Picker.Item label="Primordial" value="primordial" />
            <Picker.Item label="Basic" value="basic" />
            <Picker.Item label="Intermediate" value="intermediate" />
            <Picker.Item label="Advanced" value="advanced" />
            <Picker.Item label="Transcendent" value="transcendent" />
            <Picker.Item label="Infinite" value="infinite" />
            <Picker.Item label="Omnipotent" value="omnipotent" />
            <Picker.Item label="Divine" value="divine" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Evolution Type:</Text>
          <Picker
            selectedValue={evolutionType}
            onValueChange={setEvolutionType}
            style={styles.picker}
          >
            <Picker.Item label="Biological" value="biological" />
            <Picker.Item label="Technological" value="technological" />
            <Picker.Item label="Consciousness" value="consciousness" />
            <Picker.Item label="Spiritual" value="spiritual" />
            <Picker.Item label="Quantum" value="quantum" />
            <Picker.Item label="Dimensional" value="dimensional" />
            <Picker.Item label="Temporal" value="temporal" />
            <Picker.Item label="Universal" value="universal" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Growth Pattern:</Text>
          <Picker
            selectedValue={growthPattern}
            onValueChange={setGrowthPattern}
            style={styles.picker}
          >
            <Picker.Item label="Linear" value="linear" />
            <Picker.Item label="Exponential" value="exponential" />
            <Picker.Item label="Logarithmic" value="logarithmic" />
            <Picker.Item label="Sigmoid" value="sigmoid" />
            <Picker.Item label="Fractal" value="fractal" />
            <Picker.Item label="Chaotic" value="chaotic" />
            <Picker.Item label="Infinite" value="infinite" />
            <Picker.Item label="Transcendent" value="transcendent" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Evolution Rate (0.0-1.0)"
          value={evolutionRate}
          onChangeText={setEvolutionRate}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Complexity Level (0.0-1.0)"
          value={complexityLevel}
          onChangeText={setComplexityLevel}
          keyboardType="numeric"
        />
        <Button
          title={createEvolutionEntityMutation.isLoading ? 'Creating...' : 'Create Evolution Entity'}
          onPress={handleCreateEvolutionEntity}
          disabled={createEvolutionEntityMutation.isLoading}
        />
      </View>

      {/* Evolution Event Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Initiate Evolution Event</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity ID"
          value={eventEntityId}
          onChangeText={setEventEntityId}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Event Type:</Text>
          <Picker
            selectedValue={eventType}
            onValueChange={setEventType}
            style={styles.picker}
          >
            <Picker.Item label="Natural Selection" value="natural_selection" />
            <Picker.Item label="Mutation" value="mutation" />
            <Picker.Item label="Adaptation" value="adaptation" />
            <Picker.Item label="Speciation" value="speciation" />
            <Picker.Item label="Extinction" value="extinction" />
            <Picker.Item label="Convergence" value="convergence" />
            <Picker.Item label="Divergence" value="divergence" />
            <Picker.Item label="Transcendence" value="transcendence" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Duration (seconds)"
          value={eventDuration}
          onChangeText={setEventDuration}
          keyboardType="numeric"
        />
        <Button
          title={initiateEvolutionEventMutation.isLoading ? 'Initiating...' : 'Initiate Evolution Event'}
          onPress={handleInitiateEvolutionEvent}
          disabled={initiateEvolutionEventMutation.isLoading}
        />
      </View>

      {/* Infinite Growth Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Start Infinite Growth</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity ID"
          value={growthEntityId}
          onChangeText={setGrowthEntityId}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Growth Type:</Text>
          <Picker
            selectedValue={growthType}
            onValueChange={setGrowthType}
            style={styles.picker}
          >
            <Picker.Item label="Exponential" value="exponential" />
            <Picker.Item label="Linear" value="linear" />
            <Picker.Item label="Logarithmic" value="logarithmic" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Growth Rate"
          value={growthRate}
          onChangeText={setGrowthRate}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Current Value"
          value={currentValue}
          onChangeText={setCurrentValue}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Target Value"
          value={targetValue}
          onChangeText={setTargetValue}
          keyboardType="numeric"
        />
        <View style={styles.switchContainer}>
          <Text style={styles.label}>Is Infinite:</Text>
          <Switch
            value={isInfinite}
            onValueChange={setIsInfinite}
          />
        </View>
        <Button
          title={startInfiniteGrowthMutation.isLoading ? 'Starting...' : 'Start Infinite Growth'}
          onPress={handleStartInfiniteGrowth}
          disabled={startInfiniteGrowthMutation.isLoading}
        />
      </View>

      {/* Creation Entity Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Create Creation Entity</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity Name"
          value={creationEntityName}
          onChangeText={setCreationEntityName}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Creation Type:</Text>
          <Picker
            selectedValue={creationType}
            onValueChange={setCreationType}
            style={styles.picker}
          >
            <Picker.Item label="Matter" value="matter" />
            <Picker.Item label="Energy" value="energy" />
            <Picker.Item label="Consciousness" value="consciousness" />
            <Picker.Item label="Reality" value="reality" />
            <Picker.Item label="Universe" value="universe" />
            <Picker.Item label="Dimension" value="dimension" />
            <Picker.Item label="Time" value="time" />
            <Picker.Item label="Space" value="space" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Divine Power:</Text>
          <Picker
            selectedValue={divinePower}
            onValueChange={setDivinePower}
            style={styles.picker}
          >
            <Picker.Item label="Omnipotence" value="omnipotence" />
            <Picker.Item label="Omniscience" value="omniscience" />
            <Picker.Item label="Omnipresence" value="omnipresence" />
            <Picker.Item label="Creation" value="creation" />
            <Picker.Item label="Destruction" value="destruction" />
            <Picker.Item label="Transformation" value="transformation" />
            <Picker.Item label="Transcendence" value="transcendence" />
            <Picker.Item label="Divine Union" value="divine_union" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Manifestation Level:</Text>
          <Picker
            selectedValue={manifestationLevel}
            onValueChange={setManifestationLevel}
            style={styles.picker}
          >
            <Picker.Item label="Thought" value="thought" />
            <Picker.Item label="Intention" value="intention" />
            <Picker.Item label="Visualization" value="visualization" />
            <Picker.Item label="Manifestation" value="manifestation" />
            <Picker.Item label="Reality" value="reality" />
            <Picker.Item label="Universe" value="universe" />
            <Picker.Item label="Multiverse" value="multiverse" />
            <Picker.Item label="Omniverse" value="omniverse" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Creation Power (0.0-1.0)"
          value={creationPower}
          onChangeText={setCreationPower}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Divine Energy (0.0-1.0)"
          value={divineEnergy}
          onChangeText={setDivineEnergy}
          keyboardType="numeric"
        />
        <Button
          title={createCreationEntityMutation.isLoading ? 'Creating...' : 'Create Creation Entity'}
          onPress={handleCreateCreationEntity}
          disabled={createCreationEntityMutation.isLoading}
        />
      </View>

      {/* Creation Event Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Initiate Creation Event</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity ID"
          value={creationEventEntityId}
          onChangeText={setCreationEventEntityId}
        />
        <TextInput
          style={styles.input}
          placeholder="Target Creation"
          value={targetCreation}
          onChangeText={setTargetCreation}
        />
        <TextInput
          style={styles.input}
          placeholder="Duration (seconds)"
          value={creationEventDuration}
          onChangeText={setCreationEventDuration}
          keyboardType="numeric"
        />
        <Button
          title={initiateCreationEventMutation.isLoading ? 'Initiating...' : 'Initiate Creation Event'}
          onPress={handleInitiateCreationEvent}
          disabled={initiateCreationEventMutation.isLoading}
        />
      </View>

      {/* Information Retrieval Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Get Information</Text>
        <TextInput
          style={styles.input}
          placeholder="Evolution Entity ID"
          value={selectedEvolutionEntityId}
          onChangeText={setSelectedEvolutionEntityId}
        />
        <Button title="Get Evolution Entity Status" onPress={handleGetEvolutionEntityStatus} />
        
        {evolutionEntityStatus && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Evolution Entity Status:</Text>
            <Text>Name: {evolutionEntityStatus.name}</Text>
            <Text>Evolution Stage: {evolutionEntityStatus.evolution_stage}</Text>
            <Text>Evolution Type: {evolutionEntityStatus.evolution_type}</Text>
            <Text>Growth Pattern: {evolutionEntityStatus.growth_pattern}</Text>
            <Text>Evolution Rate: {evolutionEntityStatus.evolution_rate.toFixed(2)}</Text>
            <Text>Complexity Level: {evolutionEntityStatus.complexity_level.toFixed(2)}</Text>
            <Text>Adaptation Capacity: {evolutionEntityStatus.adaptation_capacity.toFixed(2)}</Text>
            <Text>Mutation Rate: {evolutionEntityStatus.mutation_rate.toFixed(2)}</Text>
            <Text>Selection Pressure: {evolutionEntityStatus.selection_pressure.toFixed(2)}</Text>
            <Text>Fitness Score: {evolutionEntityStatus.fitness_score.toFixed(2)}</Text>
            <Text>Is Evolving: {evolutionEntityStatus.is_evolving ? 'Yes' : 'No'}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Evolution Event ID"
          value={selectedEvolutionEventId}
          onChangeText={setSelectedEvolutionEventId}
        />
        <Button title="Get Evolution Progress" onPress={handleGetEvolutionProgress} />
        
        {evolutionProgress && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Evolution Progress:</Text>
            <Text>Event ID: {evolutionProgress.event_id}</Text>
            <Text>Entity ID: {evolutionProgress.entity_id}</Text>
            <Text>Event Type: {evolutionProgress.event_type}</Text>
            <Text>Evolution Stage: {evolutionProgress.evolution_stage}</Text>
            <Text>Evolution Type: {evolutionProgress.evolution_type}</Text>
            <Text>Success: {evolutionProgress.success ? 'Yes' : 'No'}</Text>
            <Text>Side Effects: {evolutionProgress.side_effects.join(', ')}</Text>
            <Text>Duration: {evolutionProgress.duration}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Growth ID"
          value={selectedGrowthId}
          onChangeText={setSelectedGrowthId}
        />
        <Button title="Get Growth Status" onPress={handleGetGrowthStatus} />
        
        {growthStatus && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Growth Status:</Text>
            <Text>Growth ID: {growthStatus.growth_id}</Text>
            <Text>Entity ID: {growthStatus.entity_id}</Text>
            <Text>Growth Type: {growthStatus.growth_type}</Text>
            <Text>Growth Rate: {growthStatus.growth_rate.toFixed(2)}</Text>
            <Text>Current Value: {growthStatus.current_value.toFixed(2)}</Text>
            <Text>Target Value: {growthStatus.target_value === Infinity ? 'Infinite' : growthStatus.target_value.toFixed(2)}</Text>
            <Text>Is Infinite: {growthStatus.is_infinite ? 'Yes' : 'No'}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Creation Entity ID"
          value={selectedCreationEntityId}
          onChangeText={setSelectedCreationEntityId}
        />
        <Button title="Get Creation Entity Status" onPress={handleGetCreationEntityStatus} />
        
        {creationEntityStatus && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Creation Entity Status:</Text>
            <Text>Name: {creationEntityStatus.name}</Text>
            <Text>Creation Type: {creationEntityStatus.creation_type}</Text>
            <Text>Divine Power: {creationEntityStatus.divine_power}</Text>
            <Text>Manifestation Level: {creationEntityStatus.manifestation_level}</Text>
            <Text>Creation Power: {creationEntityStatus.creation_power.toFixed(2)}</Text>
            <Text>Divine Energy: {creationEntityStatus.divine_energy.toFixed(2)}</Text>
            <Text>Omnipotence Level: {creationEntityStatus.omnipotence_level.toFixed(2)}</Text>
            <Text>Omniscience Level: {creationEntityStatus.omniscience_level.toFixed(2)}</Text>
            <Text>Omnipresence Level: {creationEntityStatus.omnipresence_level.toFixed(2)}</Text>
            <Text>Creation Capacity: {creationEntityStatus.creation_capacity.toFixed(2)}</Text>
            <Text>Is Creating: {creationEntityStatus.is_creating ? 'Yes' : 'No'}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Creation Event ID"
          value={selectedCreationEventId}
          onChangeText={setSelectedCreationEventId}
        />
        <Button title="Get Creation Progress" onPress={handleGetCreationProgress} />
        
        {creationProgress && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Creation Progress:</Text>
            <Text>Event ID: {creationProgress.event_id}</Text>
            <Text>Entity ID: {creationProgress.entity_id}</Text>
            <Text>Creation Type: {creationProgress.creation_type}</Text>
            <Text>Divine Power: {creationProgress.divine_power}</Text>
            <Text>Manifestation Level: {creationProgress.manifestation_level}</Text>
            <Text>Target Creation: {creationProgress.target_creation}</Text>
            <Text>Success: {creationProgress.success ? 'Yes' : 'No'}</Text>
            <Text>Side Effects: {creationProgress.side_effects.join(', ')}</Text>
            <Text>Duration: {creationProgress.duration}</Text>
          </View>
        )}
      </View>

      {/* Statistics Section */}
      {evolutionStats && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Evolution Statistics</Text>
          <View style={styles.statsContainer}>
            <Text style={styles.statItem}>Total Entities: {evolutionStats.total_entities}</Text>
            <Text style={styles.statItem}>Evolving Entities: {evolutionStats.evolving_entities}</Text>
            <Text style={styles.statItem}>Evolution Activity Rate: {evolutionStats.evolution_activity_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Events: {evolutionStats.total_events}</Text>
            <Text style={styles.statItem}>Successful Events: {evolutionStats.successful_events}</Text>
            <Text style={styles.statItem}>Evolution Success Rate: {evolutionStats.evolution_success_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Environments: {evolutionStats.total_environments}</Text>
            <Text style={styles.statItem}>Stable Environments: {evolutionStats.stable_environments}</Text>
            <Text style={styles.statItem}>Total Growths: {evolutionStats.total_growths}</Text>
            <Text style={styles.statItem}>Infinite Growths: {evolutionStats.infinite_growths}</Text>
          </View>
        </View>
      )}

      {creationStats && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Creation Statistics</Text>
          <View style={styles.statsContainer}>
            <Text style={styles.statItem}>Total Entities: {creationStats.total_entities}</Text>
            <Text style={styles.statItem}>Creating Entities: {creationStats.creating_entities}</Text>
            <Text style={styles.statItem}>Creation Activity Rate: {creationStats.creation_activity_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Events: {creationStats.total_events}</Text>
            <Text style={styles.statItem}>Successful Events: {creationStats.successful_events}</Text>
            <Text style={styles.statItem}>Creation Success Rate: {creationStats.creation_success_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Manifestations: {creationStats.total_manifestations}</Text>
            <Text style={styles.statItem}>Active Manifestations: {creationStats.active_manifestations}</Text>
            <Text style={styles.statItem}>Total Realities: {creationStats.total_realities}</Text>
            <Text style={styles.statItem}>Stable Realities: {creationStats.stable_realities}</Text>
          </View>
        </View>
      )}

      {(isLoadingEvolutionStats || isLoadingCreationStats) && (
        <ActivityIndicator size="large" color="#0000ff" style={styles.activityIndicator} />
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  section: {
    backgroundColor: 'white',
    padding: 15,
    marginBottom: 20,
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    padding: 10,
    marginBottom: 10,
    borderRadius: 5,
    backgroundColor: '#fff',
  },
  pickerContainer: {
    marginBottom: 10,
  },
  picker: {
    height: 50,
    width: '100%',
  },
  label: {
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 5,
    color: '#333',
  },
  switchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  infoContainer: {
    backgroundColor: '#f0f0f0',
    padding: 10,
    marginTop: 10,
    borderRadius: 5,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
    color: '#333',
  },
  statsContainer: {
    backgroundColor: '#f0f0f0',
    padding: 10,
    borderRadius: 5,
  },
  statItem: {
    fontSize: 14,
    marginBottom: 5,
    color: '#333',
  },
  activityIndicator: {
    marginTop: 20,
  },
});

export default InfiniteEvolutionScreen;

