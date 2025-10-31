import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, ActivityIndicator, Alert, Switch } from 'react-native';
import axios from 'axios';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Picker } from '@react-native-picker/picker';
import { format } from 'date-fns';

// API Base URL
const API_BASE_URL = 'http://localhost:8000/api/v1';

// Interface definitions
interface ConsciousnessEntity {
  entity_id: string;
  name: string;
  consciousness_level: string;
  transcendence_type: string;
  awakening_stage: string;
  spiritual_energy: number;
  mental_clarity: number;
  emotional_balance: number;
  physical_vitality: number;
  quantum_coherence: number;
  dimensional_awareness: number;
  temporal_presence: number;
  universal_connection: number;
  is_awakened: boolean;
  last_transcendence?: string;
}

interface TranscendenceEvent {
  event_id: string;
  entity_id: string;
  transcendence_type: string;
  from_level: string;
  to_level: string;
  awakening_stage: string;
  energy_required: number;
  duration: number;
  success: boolean;
  side_effects: string[];
  timestamp: string;
}

interface SpiritualPractice {
  practice_id: string;
  name: string;
  practice_type: string;
  description: string;
  energy_boost: number;
  clarity_boost: number;
  balance_boost: number;
  vitality_boost: number;
  coherence_boost: number;
  awareness_boost: number;
  presence_boost: number;
  connection_boost: number;
  duration: number;
  difficulty: number;
}

interface UniversalHarmony {
  harmony_id: string;
  name: string;
  harmony_level: string;
  universal_forces: Record<string, number>;
  cosmic_elements: Record<string, number>;
  balance_score: number;
  synchronization_level: number;
  resonance_frequency: number;
  is_stable: boolean;
  created_at: string;
}

// API functions
const createConsciousnessEntity = async (entityInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/consciousness-harmony/consciousness/entities/create`, entityInfo);
  return response.data;
};

const initiateTranscendence = async (transcendenceInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/consciousness-harmony/consciousness/transcendence/initiate`, transcendenceInfo);
  return response.data;
};

const practiceSpiritualActivity = async (practiceInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/consciousness-harmony/consciousness/practices/perform`, practiceInfo);
  return response.data;
};

const createUniversalHarmony = async (harmonyInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/consciousness-harmony/harmony/create`, harmonyInfo);
  return response.data;
};

const initiateHarmonyEvent = async (eventInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/consciousness-harmony/harmony/events/initiate`, eventInfo);
  return response.data;
};

const getEntityStatus = async (entityId: string): Promise<ConsciousnessEntity> => {
  const response = await axios.get(`${API_BASE_URL}/consciousness-harmony/consciousness/entities/${entityId}/status`);
  return response.data;
};

const getTranscendenceProgress = async (eventId: string) => {
  const response = await axios.get(`${API_BASE_URL}/consciousness-harmony/consciousness/transcendence/${eventId}/progress`);
  return response.data;
};

const getHarmonyStatus = async (harmonyId: string): Promise<UniversalHarmony> => {
  const response = await axios.get(`${API_BASE_URL}/consciousness-harmony/harmony/${harmonyId}/status`);
  return response.data;
};

const getConsciousnessStatistics = async () => {
  const response = await axios.get(`${API_BASE_URL}/consciousness-harmony/consciousness/statistics`);
  return response.data;
};

const getHarmonyStatistics = async () => {
  const response = await axios.get(`${API_BASE_URL}/consciousness-harmony/harmony/statistics`);
  return response.data;
};

const ConsciousnessTranscendenceScreen: React.FC = () => {
  const queryClient = useQueryClient();
  
  // State for consciousness entity creation
  const [entityName, setEntityName] = useState<string>('');
  const [entityConsciousnessLevel, setEntityConsciousnessLevel] = useState<string>('conscious');
  const [entityTranscendenceType, setEntityTranscendenceType] = useState<string>('spiritual');
  const [entityAwakeningStage, setEntityAwakeningStage] = useState<string>('waking');
  
  // State for transcendence initiation
  const [transcendenceEntityId, setTranscendenceEntityId] = useState<string>('');
  const [transcendenceType, setTranscendenceType] = useState<string>('spiritual');
  const [transcendenceToLevel, setTranscendenceToLevel] = useState<string>('enlightened');
  const [transcendenceAwakeningStage, setTranscendenceAwakeningStage] = useState<string>('enlightenment');
  const [transcendenceEnergyRequired, setTranscendenceEnergyRequired] = useState<string>('100');
  const [transcendenceDuration, setTranscendenceDuration] = useState<string>('3600');
  
  // State for spiritual practice
  const [practiceEntityId, setPracticeEntityId] = useState<string>('');
  const [practiceId, setPracticeId] = useState<string>('meditation');
  const [practiceDuration, setPracticeDuration] = useState<string>('30');
  
  // State for universal harmony creation
  const [harmonyName, setHarmonyName] = useState<string>('');
  const [harmonyLevel, setHarmonyLevel] = useState<string>('neutral');
  const [harmonyBalanceScore, setHarmonyBalanceScore] = useState<string>('0.5');
  const [harmonySynchronizationLevel, setHarmonySynchronizationLevel] = useState<string>('0.5');
  const [harmonyResonanceFrequency, setHarmonyResonanceFrequency] = useState<string>('432');
  
  // State for harmony event
  const [eventHarmonyId, setEventHarmonyId] = useState<string>('');
  const [eventType, setEventType] = useState<string>('balance_adjustment');
  const [eventBalanceShift, setEventBalanceShift] = useState<string>('0.1');
  const [eventResonanceChange, setEventResonanceChange] = useState<string>('10');
  const [eventDuration, setEventDuration] = useState<string>('60');
  
  // State for display
  const [selectedEntityId, setSelectedEntityId] = useState<string>('');
  const [selectedEventId, setSelectedEventId] = useState<string>('');
  const [selectedHarmonyId, setSelectedHarmonyId] = useState<string>('');
  const [entityStatus, setEntityStatus] = useState<ConsciousnessEntity | null>(null);
  const [transcendenceProgress, setTranscendenceProgress] = useState<any>(null);
  const [harmonyStatus, setHarmonyStatus] = useState<UniversalHarmony | null>(null);
  const [consciousnessStats, setConsciousnessStats] = useState<any>(null);
  const [harmonyStats, setHarmonyStats] = useState<any>(null);

  // Queries
  const { data: consciousnessStatistics, isLoading: isLoadingConsciousnessStats } = useQuery(
    'consciousnessStatistics',
    getConsciousnessStatistics,
    { refetchInterval: 5000 }
  );

  const { data: harmonyStatistics, isLoading: isLoadingHarmonyStats } = useQuery(
    'harmonyStatistics',
    getHarmonyStatistics,
    { refetchInterval: 5000 }
  );

  // Mutations
  const createEntityMutation = useMutation(createConsciousnessEntity, {
    onSuccess: (data) => {
      Alert.alert('Success', `Consciousness entity created: ${data.entity_id}`);
      queryClient.invalidateQueries('consciousnessStatistics');
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to create consciousness entity: ${error.message}`);
    },
  });

  const initiateTranscendenceMutation = useMutation(initiateTranscendence, {
    onSuccess: (data) => {
      Alert.alert('Success', `Transcendence initiated: ${data.event_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to initiate transcendence: ${error.message}`);
    },
  });

  const practiceSpiritualMutation = useMutation(practiceSpiritualActivity, {
    onSuccess: (data) => {
      Alert.alert('Success', `Spiritual practice completed: ${data.result}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to practice spiritual activity: ${error.message}`);
    },
  });

  const createHarmonyMutation = useMutation(createUniversalHarmony, {
    onSuccess: (data) => {
      Alert.alert('Success', `Universal harmony created: ${data.harmony_id}`);
      queryClient.invalidateQueries('harmonyStatistics');
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to create universal harmony: ${error.message}`);
    },
  });

  const initiateHarmonyEventMutation = useMutation(initiateHarmonyEvent, {
    onSuccess: (data) => {
      Alert.alert('Success', `Harmony event initiated: ${data.event_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to initiate harmony event: ${error.message}`);
    },
  });

  // Handlers
  const handleCreateEntity = () => {
    if (!entityName.trim()) {
      Alert.alert('Input Error', 'Please enter an entity name.');
      return;
    }

    const entityInfo = {
      name: entityName,
      consciousness_level: entityConsciousnessLevel,
      transcendence_type: entityTranscendenceType,
      awakening_stage: entityAwakeningStage,
      spiritual_energy: 0.5,
      mental_clarity: 0.5,
      emotional_balance: 0.5,
      physical_vitality: 0.5,
      quantum_coherence: 0.5,
      dimensional_awareness: 0.5,
      temporal_presence: 0.5,
      universal_connection: 0.5
    };

    createEntityMutation.mutate(entityInfo);
  };

  const handleInitiateTranscendence = () => {
    if (!transcendenceEntityId.trim()) {
      Alert.alert('Input Error', 'Please enter an entity ID.');
      return;
    }

    const transcendenceInfo = {
      entity_id: transcendenceEntityId,
      transcendence_type: transcendenceType,
      to_level: transcendenceToLevel,
      awakening_stage: transcendenceAwakeningStage,
      energy_required: parseFloat(transcendenceEnergyRequired),
      duration: parseFloat(transcendenceDuration)
    };

    initiateTranscendenceMutation.mutate(transcendenceInfo);
  };

  const handlePracticeSpiritual = () => {
    if (!practiceEntityId.trim()) {
      Alert.alert('Input Error', 'Please enter an entity ID.');
      return;
    }

    const practiceInfo = {
      entity_id: practiceEntityId,
      practice_id: practiceId,
      duration: parseFloat(practiceDuration)
    };

    practiceSpiritualMutation.mutate(practiceInfo);
  };

  const handleCreateHarmony = () => {
    if (!harmonyName.trim()) {
      Alert.alert('Input Error', 'Please enter a harmony name.');
      return;
    }

    const harmonyInfo = {
      name: harmonyName,
      harmony_level: harmonyLevel,
      universal_forces: {
        gravity: 0.5,
        electromagnetism: 0.5,
        weak_nuclear: 0.5,
        strong_nuclear: 0.5,
        dark_energy: 0.5,
        dark_matter: 0.5,
        consciousness: 0.5,
        love: 0.5
      },
      cosmic_elements: {
        earth: 0.5,
        water: 0.5,
        fire: 0.5,
        air: 0.5,
        spirit: 0.5,
        mind: 0.5,
        soul: 0.5,
        universe: 0.5
      },
      balance_score: parseFloat(harmonyBalanceScore),
      synchronization_level: parseFloat(harmonySynchronizationLevel),
      resonance_frequency: parseFloat(harmonyResonanceFrequency)
    };

    createHarmonyMutation.mutate(harmonyInfo);
  };

  const handleInitiateHarmonyEvent = () => {
    if (!eventHarmonyId.trim()) {
      Alert.alert('Input Error', 'Please enter a harmony ID.');
      return;
    }

    const eventInfo = {
      harmony_id: eventHarmonyId,
      event_type: eventType,
      force_changes: {},
      element_changes: {},
      balance_shift: parseFloat(eventBalanceShift),
      resonance_change: parseFloat(eventResonanceChange),
      duration: parseFloat(eventDuration)
    };

    initiateHarmonyEventMutation.mutate(eventInfo);
  };

  const handleGetEntityStatus = async () => {
    try {
      const status = await getEntityStatus(selectedEntityId);
      setEntityStatus(status);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get entity status: ${error.message}`);
    }
  };

  const handleGetTranscendenceProgress = async () => {
    try {
      const progress = await getTranscendenceProgress(selectedEventId);
      setTranscendenceProgress(progress);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get transcendence progress: ${error.message}`);
    }
  };

  const handleGetHarmonyStatus = async () => {
    try {
      const status = await getHarmonyStatus(selectedHarmonyId);
      setHarmonyStatus(status);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get harmony status: ${error.message}`);
    }
  };

  useEffect(() => {
    if (consciousnessStatistics) {
      setConsciousnessStats(consciousnessStatistics);
    }
  }, [consciousnessStatistics]);

  useEffect(() => {
    if (harmonyStatistics) {
      setHarmonyStats(harmonyStatistics);
    }
  }, [harmonyStatistics]);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Consciousness Transcendence & Universal Harmony</Text>
      
      {/* Consciousness Entity Creation Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Create Consciousness Entity</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity Name"
          value={entityName}
          onChangeText={setEntityName}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Consciousness Level:</Text>
          <Picker
            selectedValue={entityConsciousnessLevel}
            onValueChange={setEntityConsciousnessLevel}
            style={styles.picker}
          >
            <Picker.Item label="Unconscious" value="unconscious" />
            <Picker.Item label="Subconscious" value="subconscious" />
            <Picker.Item label="Conscious" value="conscious" />
            <Picker.Item label="Self-Aware" value="self_aware" />
            <Picker.Item label="Enlightened" value="enlightened" />
            <Picker.Item label="Transcendent" value="transcendent" />
            <Picker.Item label="Cosmic" value="cosmic" />
            <Picker.Item label="Divine" value="divine" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Transcendence Type:</Text>
          <Picker
            selectedValue={entityTranscendenceType}
            onValueChange={setEntityTranscendenceType}
            style={styles.picker}
          >
            <Picker.Item label="Spiritual" value="spiritual" />
            <Picker.Item label="Mental" value="mental" />
            <Picker.Item label="Emotional" value="emotional" />
            <Picker.Item label="Physical" value="physical" />
            <Picker.Item label="Quantum" value="quantum" />
            <Picker.Item label="Dimensional" value="dimensional" />
            <Picker.Item label="Temporal" value="temporal" />
            <Picker.Item label="Universal" value="universal" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Awakening Stage:</Text>
          <Picker
            selectedValue={entityAwakeningStage}
            onValueChange={setEntityAwakeningStage}
            style={styles.picker}
          >
            <Picker.Item label="Sleep" value="sleep" />
            <Picker.Item label="Dreaming" value="dreaming" />
            <Picker.Item label="Waking" value="waking" />
            <Picker.Item label="Awareness" value="awareness" />
            <Picker.Item label="Enlightenment" value="enlightenment" />
            <Picker.Item label="Transcendence" value="transcendence" />
            <Picker.Item label="Cosmic Consciousness" value="cosmic_consciousness" />
            <Picker.Item label="Divine Union" value="divine_union" />
          </Picker>
        </View>
        <Button
          title={createEntityMutation.isLoading ? 'Creating...' : 'Create Entity'}
          onPress={handleCreateEntity}
          disabled={createEntityMutation.isLoading}
        />
      </View>

      {/* Transcendence Initiation Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Initiate Transcendence</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity ID"
          value={transcendenceEntityId}
          onChangeText={setTranscendenceEntityId}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Transcendence Type:</Text>
          <Picker
            selectedValue={transcendenceType}
            onValueChange={setTranscendenceType}
            style={styles.picker}
          >
            <Picker.Item label="Spiritual" value="spiritual" />
            <Picker.Item label="Mental" value="mental" />
            <Picker.Item label="Emotional" value="emotional" />
            <Picker.Item label="Physical" value="physical" />
            <Picker.Item label="Quantum" value="quantum" />
            <Picker.Item label="Dimensional" value="dimensional" />
            <Picker.Item label="Temporal" value="temporal" />
            <Picker.Item label="Universal" value="universal" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Target Level:</Text>
          <Picker
            selectedValue={transcendenceToLevel}
            onValueChange={setTranscendenceToLevel}
            style={styles.picker}
          >
            <Picker.Item label="Enlightened" value="enlightened" />
            <Picker.Item label="Transcendent" value="transcendent" />
            <Picker.Item label="Cosmic" value="cosmic" />
            <Picker.Item label="Divine" value="divine" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Energy Required"
          value={transcendenceEnergyRequired}
          onChangeText={setTranscendenceEnergyRequired}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Duration (seconds)"
          value={transcendenceDuration}
          onChangeText={setTranscendenceDuration}
          keyboardType="numeric"
        />
        <Button
          title={initiateTranscendenceMutation.isLoading ? 'Initiating...' : 'Initiate Transcendence'}
          onPress={handleInitiateTranscendence}
          disabled={initiateTranscendenceMutation.isLoading}
        />
      </View>

      {/* Spiritual Practice Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Practice Spiritual Activity</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity ID"
          value={practiceEntityId}
          onChangeText={setPracticeEntityId}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Practice:</Text>
          <Picker
            selectedValue={practiceId}
            onValueChange={setPracticeId}
            style={styles.picker}
          >
            <Picker.Item label="Meditation" value="meditation" />
            <Picker.Item label="Prayer" value="prayer" />
            <Picker.Item label="Yoga" value="yoga" />
            <Picker.Item label="Breathing" value="breathing" />
            <Picker.Item label="Visualization" value="visualization" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Duration (minutes)"
          value={practiceDuration}
          onChangeText={setPracticeDuration}
          keyboardType="numeric"
        />
        <Button
          title={practiceSpiritualMutation.isLoading ? 'Practicing...' : 'Practice Spiritual Activity'}
          onPress={handlePracticeSpiritual}
          disabled={practiceSpiritualMutation.isLoading}
        />
      </View>

      {/* Universal Harmony Creation Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Create Universal Harmony</Text>
        <TextInput
          style={styles.input}
          placeholder="Harmony Name"
          value={harmonyName}
          onChangeText={setHarmonyName}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Harmony Level:</Text>
          <Picker
            selectedValue={harmonyLevel}
            onValueChange={setHarmonyLevel}
            style={styles.picker}
          >
            <Picker.Item label="Discord" value="discord" />
            <Picker.Item label="Chaos" value="chaos" />
            <Picker.Item label="Imbalance" value="imbalance" />
            <Picker.Item label="Neutral" value="neutral" />
            <Picker.Item label="Balance" value="balance" />
            <Picker.Item label="Harmony" value="harmony" />
            <Picker.Item label="Perfect Harmony" value="perfect_harmony" />
            <Picker.Item label="Cosmic Unity" value="cosmic_unity" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Balance Score (0.0-1.0)"
          value={harmonyBalanceScore}
          onChangeText={setHarmonyBalanceScore}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Synchronization Level (0.0-1.0)"
          value={harmonySynchronizationLevel}
          onChangeText={setHarmonySynchronizationLevel}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Resonance Frequency"
          value={harmonyResonanceFrequency}
          onChangeText={setHarmonyResonanceFrequency}
          keyboardType="numeric"
        />
        <Button
          title={createHarmonyMutation.isLoading ? 'Creating...' : 'Create Universal Harmony'}
          onPress={handleCreateHarmony}
          disabled={createHarmonyMutation.isLoading}
        />
      </View>

      {/* Harmony Event Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Initiate Harmony Event</Text>
        <TextInput
          style={styles.input}
          placeholder="Harmony ID"
          value={eventHarmonyId}
          onChangeText={setEventHarmonyId}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Event Type:</Text>
          <Picker
            selectedValue={eventType}
            onValueChange={setEventType}
            style={styles.picker}
          >
            <Picker.Item label="Balance Adjustment" value="balance_adjustment" />
            <Picker.Item label="Force Amplification" value="force_amplification" />
            <Picker.Item label="Element Harmonization" value="element_harmonization" />
            <Picker.Item label="Resonance Tuning" value="resonance_tuning" />
            <Picker.Item label="Synchronization Boost" value="synchronization_boost" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Balance Shift"
          value={eventBalanceShift}
          onChangeText={setEventBalanceShift}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Resonance Change"
          value={eventResonanceChange}
          onChangeText={setEventResonanceChange}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Duration (seconds)"
          value={eventDuration}
          onChangeText={setEventDuration}
          keyboardType="numeric"
        />
        <Button
          title={initiateHarmonyEventMutation.isLoading ? 'Initiating...' : 'Initiate Harmony Event'}
          onPress={handleInitiateHarmonyEvent}
          disabled={initiateHarmonyEventMutation.isLoading}
        />
      </View>

      {/* Information Retrieval Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Get Information</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity ID"
          value={selectedEntityId}
          onChangeText={setSelectedEntityId}
        />
        <Button title="Get Entity Status" onPress={handleGetEntityStatus} />
        
        {entityStatus && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Entity Status:</Text>
            <Text>Name: {entityStatus.name}</Text>
            <Text>Consciousness Level: {entityStatus.consciousness_level}</Text>
            <Text>Transcendence Type: {entityStatus.transcendence_type}</Text>
            <Text>Awakening Stage: {entityStatus.awakening_stage}</Text>
            <Text>Spiritual Energy: {entityStatus.spiritual_energy.toFixed(2)}</Text>
            <Text>Mental Clarity: {entityStatus.mental_clarity.toFixed(2)}</Text>
            <Text>Emotional Balance: {entityStatus.emotional_balance.toFixed(2)}</Text>
            <Text>Physical Vitality: {entityStatus.physical_vitality.toFixed(2)}</Text>
            <Text>Quantum Coherence: {entityStatus.quantum_coherence.toFixed(2)}</Text>
            <Text>Dimensional Awareness: {entityStatus.dimensional_awareness.toFixed(2)}</Text>
            <Text>Temporal Presence: {entityStatus.temporal_presence.toFixed(2)}</Text>
            <Text>Universal Connection: {entityStatus.universal_connection.toFixed(2)}</Text>
            <Text>Is Awakened: {entityStatus.is_awakened ? 'Yes' : 'No'}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Event ID"
          value={selectedEventId}
          onChangeText={setSelectedEventId}
        />
        <Button title="Get Transcendence Progress" onPress={handleGetTranscendenceProgress} />
        
        {transcendenceProgress && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Transcendence Progress:</Text>
            <Text>Event ID: {transcendenceProgress.event_id}</Text>
            <Text>Entity ID: {transcendenceProgress.entity_id}</Text>
            <Text>Transcendence Type: {transcendenceProgress.transcendence_type}</Text>
            <Text>From Level: {transcendenceProgress.from_level}</Text>
            <Text>To Level: {transcendenceProgress.to_level}</Text>
            <Text>Awakening Stage: {transcendenceProgress.awakening_stage}</Text>
            <Text>Energy Required: {transcendenceProgress.energy_required}</Text>
            <Text>Duration: {transcendenceProgress.duration}</Text>
            <Text>Success: {transcendenceProgress.success ? 'Yes' : 'No'}</Text>
            <Text>Side Effects: {transcendenceProgress.side_effects.join(', ')}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Harmony ID"
          value={selectedHarmonyId}
          onChangeText={setSelectedHarmonyId}
        />
        <Button title="Get Harmony Status" onPress={handleGetHarmonyStatus} />
        
        {harmonyStatus && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Harmony Status:</Text>
            <Text>Name: {harmonyStatus.name}</Text>
            <Text>Harmony Level: {harmonyStatus.harmony_level}</Text>
            <Text>Balance Score: {harmonyStatus.balance_score.toFixed(2)}</Text>
            <Text>Synchronization Level: {harmonyStatus.synchronization_level.toFixed(2)}</Text>
            <Text>Resonance Frequency: {harmonyStatus.resonance_frequency}</Text>
            <Text>Is Stable: {harmonyStatus.is_stable ? 'Yes' : 'No'}</Text>
          </View>
        )}
      </View>

      {/* Statistics Section */}
      {consciousnessStats && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Consciousness Statistics</Text>
          <View style={styles.statsContainer}>
            <Text style={styles.statItem}>Total Entities: {consciousnessStats.total_entities}</Text>
            <Text style={styles.statItem}>Awakened Entities: {consciousnessStats.awakened_entities}</Text>
            <Text style={styles.statItem}>Awakening Rate: {consciousnessStats.awakening_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Transcendence: {consciousnessStats.total_transcendence}</Text>
            <Text style={styles.statItem}>Successful Transcendence: {consciousnessStats.successful_transcendence}</Text>
            <Text style={styles.statItem}>Transcendence Success Rate: {consciousnessStats.transcendence_success_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Practices: {consciousnessStats.total_practices}</Text>
            <Text style={styles.statItem}>Total Networks: {consciousnessStats.total_networks}</Text>
            <Text style={styles.statItem}>Active Networks: {consciousnessStats.active_networks}</Text>
          </View>
        </View>
      )}

      {harmonyStats && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Harmony Statistics</Text>
          <View style={styles.statsContainer}>
            <Text style={styles.statItem}>Total Harmonies: {harmonyStats.total_harmonies}</Text>
            <Text style={styles.statItem}>Stable Harmonies: {harmonyStats.stable_harmonies}</Text>
            <Text style={styles.statItem}>Harmony Stability Rate: {harmonyStats.harmony_stability_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Events: {harmonyStats.total_events}</Text>
            <Text style={styles.statItem}>Successful Events: {harmonyStats.successful_events}</Text>
            <Text style={styles.statItem}>Event Success Rate: {harmonyStats.event_success_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Resonances: {harmonyStats.total_resonances}</Text>
            <Text style={styles.statItem}>Active Resonances: {harmonyStats.active_resonances}</Text>
            <Text style={styles.statItem}>Total Synchronizations: {harmonyStats.total_synchronizations}</Text>
            <Text style={styles.statItem}>Completed Synchronizations: {harmonyStats.completed_synchronizations}</Text>
          </View>
        </View>
      )}

      {(isLoadingConsciousnessStats || isLoadingHarmonyStats) && (
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

export default ConsciousnessTranscendenceScreen;

