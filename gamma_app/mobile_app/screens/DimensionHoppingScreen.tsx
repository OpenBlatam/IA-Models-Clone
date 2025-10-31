import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, ActivityIndicator, Alert, Switch } from 'react-native';
import axios from 'axios';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Picker } from '@react-native-picker/picker';
import { format } from 'date-fns';

// API Base URL
const API_BASE_URL = 'http://localhost:8000/api/v1';

// Interface definitions
interface DimensionInfo {
  dimension_id: string;
  name: string;
  dimension_type: string;
  reality_level: string;
  physical_laws: Record<string, any>;
  inhabitants: string[];
  resources: Record<string, number>;
  technology_level: number;
  is_accessible: boolean;
  stability_score: number;
  last_accessed?: string;
}

interface DimensionHop {
  hop_id: string;
  traveler_id: string;
  source_dimension: string;
  target_dimension: string;
  hopping_method: string;
  departure_time: string;
  arrival_time: string;
  success: boolean;
  reality_shift: number;
  side_effects: string[];
}

interface RealityAnomaly {
  anomaly_id: string;
  dimension_id: string;
  anomaly_type: string;
  location: [number, number, number];
  severity: number;
  description: string;
  effects: string[];
  containment_status: string;
  created_at: string;
}

interface InterdimensionalEntity {
  entity_id: string;
  name: string;
  origin_dimension: string;
  current_dimension: string;
  entity_type: string;
  abilities: string[];
  threat_level: number;
  is_hostile: boolean;
  last_seen: string;
  communication_protocol: string;
}

// API functions
const registerDimension = async (dimensionInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/dimension-reality/dimension/register`, dimensionInfo);
  return response.data;
};

const initiateDimensionHop = async (hopInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/dimension-reality/dimension/hop/initiate`, hopInfo);
  return response.data;
};

const detectRealityAnomaly = async (anomalyInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/dimension-reality/dimension/anomalies/detect`, anomalyInfo);
  return response.data;
};

const registerInterdimensionalEntity = async (entityInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/dimension-reality/dimension/entities/register`, entityInfo);
  return response.data;
};

const getDimensionInfo = async (dimensionId: string): Promise<DimensionInfo> => {
  const response = await axios.get(`${API_BASE_URL}/dimension-reality/dimension/${dimensionId}/info`);
  return response.data;
};

const getTravelerStatus = async (travelerId: string) => {
  const response = await axios.get(`${API_BASE_URL}/dimension-reality/dimension/travelers/${travelerId}/status`);
  return response.data;
};

const getDimensionStatistics = async () => {
  const response = await axios.get(`${API_BASE_URL}/dimension-reality/dimension/statistics`);
  return response.data;
};

const DimensionHoppingScreen: React.FC = () => {
  const queryClient = useQueryClient();
  
  // State for dimension registration
  const [newDimensionName, setNewDimensionName] = useState<string>('');
  const [newDimensionType, setNewDimensionType] = useState<string>('parallel');
  const [newRealityLevel, setNewRealityLevel] = useState<string>('stable');
  const [newTechnologyLevel, setNewTechnologyLevel] = useState<string>('5');
  
  // State for dimension hopping
  const [travelerId, setTravelerId] = useState<string>('traveler_123');
  const [sourceDimension, setSourceDimension] = useState<string>('prime_dimension');
  const [targetDimension, setTargetDimension] = useState<string>('parallel_dimension');
  const [hoppingMethod, setHoppingMethod] = useState<string>('quantum_tunnel');
  
  // State for anomaly detection
  const [anomalyDimensionId, setAnomalyDimensionId] = useState<string>('prime_dimension');
  const [anomalyType, setAnomalyType] = useState<string>('spatial_distortion');
  const [anomalySeverity, setAnomalySeverity] = useState<string>('0.5');
  const [anomalyDescription, setAnomalyDescription] = useState<string>('');
  
  // State for entity registration
  const [entityName, setEntityName] = useState<string>('');
  const [entityOriginDimension, setEntityOriginDimension] = useState<string>('unknown');
  const [entityCurrentDimension, setEntityCurrentDimension] = useState<string>('prime_dimension');
  const [entityType, setEntityType] = useState<string>('reality_manipulator');
  const [entityThreatLevel, setEntityThreatLevel] = useState<string>('0.5');
  const [entityIsHostile, setEntityIsHostile] = useState<boolean>(false);
  
  // State for display
  const [selectedDimensionId, setSelectedDimensionId] = useState<string>('prime_dimension');
  const [selectedTravelerId, setSelectedTravelerId] = useState<string>('traveler_123');
  const [dimensionInfo, setDimensionInfo] = useState<DimensionInfo | null>(null);
  const [travelerStatus, setTravelerStatus] = useState<any>(null);
  const [statistics, setStatistics] = useState<any>(null);

  // Queries
  const { data: dimensionStats, isLoading: isLoadingStats, error: statsError } = useQuery(
    'dimensionStatistics',
    getDimensionStatistics,
    { refetchInterval: 5000 }
  );

  // Mutations
  const registerDimensionMutation = useMutation(registerDimension, {
    onSuccess: (data) => {
      Alert.alert('Success', `Dimension registered: ${data.dimension_id}`);
      queryClient.invalidateQueries('dimensionStatistics');
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to register dimension: ${error.message}`);
    },
  });

  const initiateHopMutation = useMutation(initiateDimensionHop, {
    onSuccess: (data) => {
      Alert.alert('Success', `Dimension hop initiated: ${data.hop_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to initiate dimension hop: ${error.message}`);
    },
  });

  const detectAnomalyMutation = useMutation(detectRealityAnomaly, {
    onSuccess: (data) => {
      Alert.alert('Success', `Reality anomaly detected: ${data.anomaly_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to detect reality anomaly: ${error.message}`);
    },
  });

  const registerEntityMutation = useMutation(registerInterdimensionalEntity, {
    onSuccess: (data) => {
      Alert.alert('Success', `Interdimensional entity registered: ${data.entity_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to register interdimensional entity: ${error.message}`);
    },
  });

  // Handlers
  const handleRegisterDimension = () => {
    if (!newDimensionName.trim()) {
      Alert.alert('Input Error', 'Please enter a dimension name.');
      return;
    }

    const dimensionInfo = {
      name: newDimensionName,
      dimension_type: newDimensionType,
      reality_level: newRealityLevel,
      physical_laws: {
        gravity: 9.81,
        speed_of_light: 299792458,
        planck_constant: 6.626e-34,
        entropy: "increasing"
      },
      inhabitants: ["unknown"],
      resources: { matter: 1.0, energy: 0.8, information: 0.9 },
      technology_level: parseInt(newTechnologyLevel, 10)
    };

    registerDimensionMutation.mutate(dimensionInfo);
  };

  const handleInitiateHop = () => {
    if (!travelerId.trim() || !sourceDimension.trim() || !targetDimension.trim()) {
      Alert.alert('Input Error', 'Please fill in all required fields.');
      return;
    }

    const hopInfo = {
      traveler_id: travelerId,
      source_dimension: sourceDimension,
      target_dimension: targetDimension,
      hopping_method: hoppingMethod,
      arrival_time: new Date().toISOString()
    };

    initiateHopMutation.mutate(hopInfo);
  };

  const handleDetectAnomaly = () => {
    if (!anomalyDescription.trim()) {
      Alert.alert('Input Error', 'Please enter an anomaly description.');
      return;
    }

    const anomalyInfo = {
      dimension_id: anomalyDimensionId,
      anomaly_type: anomalyType,
      location: [Math.random() * 100, Math.random() * 100, Math.random() * 100],
      severity: parseFloat(anomalySeverity),
      description: anomalyDescription,
      effects: ["spatial_distortion", "temporal_fluctuation"]
    };

    detectAnomalyMutation.mutate(anomalyInfo);
  };

  const handleRegisterEntity = () => {
    if (!entityName.trim()) {
      Alert.alert('Input Error', 'Please enter an entity name.');
      return;
    }

    const entityInfo = {
      name: entityName,
      origin_dimension: entityOriginDimension,
      current_dimension: entityCurrentDimension,
      entity_type: entityType,
      abilities: ["dimensional_phase", "reality_distortion"],
      threat_level: parseFloat(entityThreatLevel),
      is_hostile: entityIsHostile,
      communication_protocol: "telepathic"
    };

    registerEntityMutation.mutate(entityInfo);
  };

  const handleGetDimensionInfo = async () => {
    try {
      const info = await getDimensionInfo(selectedDimensionId);
      setDimensionInfo(info);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get dimension info: ${error.message}`);
    }
  };

  const handleGetTravelerStatus = async () => {
    try {
      const status = await getTravelerStatus(selectedTravelerId);
      setTravelerStatus(status);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get traveler status: ${error.message}`);
    }
  };

  useEffect(() => {
    if (dimensionStats) {
      setStatistics(dimensionStats);
    }
  }, [dimensionStats]);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Dimension Hopping & Reality Engine</Text>
      
      {/* Dimension Registration Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Register New Dimension</Text>
        <TextInput
          style={styles.input}
          placeholder="Dimension Name"
          value={newDimensionName}
          onChangeText={setNewDimensionName}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Dimension Type:</Text>
          <Picker
            selectedValue={newDimensionType}
            onValueChange={setNewDimensionType}
            style={styles.picker}
          >
            <Picker.Item label="Parallel" value="parallel" />
            <Picker.Item label="Alternate" value="alternate" />
            <Picker.Item label="Mirror" value="mirror" />
            <Picker.Item label="Quantum" value="quantum" />
            <Picker.Item label="Virtual" value="virtual" />
            <Picker.Item label="Dream" value="dream" />
            <Picker.Item label="Pocket" value="pocket" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Reality Level:</Text>
          <Picker
            selectedValue={newRealityLevel}
            onValueChange={setNewRealityLevel}
            style={styles.picker}
          >
            <Picker.Item label="Stable" value="stable" />
            <Picker.Item label="Fluctuating" value="fluctuating" />
            <Picker.Item label="Unstable" value="unstable" />
            <Picker.Item label="Collapsing" value="collapsing" />
            <Picker.Item label="Reconstructing" value="reconstructing" />
            <Picker.Item label="Synthetic" value="synthetic" />
            <Picker.Item label="Hybrid" value="hybrid" />
            <Picker.Item label="Transcendent" value="transcendent" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Technology Level (1-10)"
          value={newTechnologyLevel}
          onChangeText={setNewTechnologyLevel}
          keyboardType="numeric"
        />
        <Button
          title={registerDimensionMutation.isLoading ? 'Registering...' : 'Register Dimension'}
          onPress={handleRegisterDimension}
          disabled={registerDimensionMutation.isLoading}
        />
      </View>

      {/* Dimension Hopping Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Initiate Dimension Hop</Text>
        <TextInput
          style={styles.input}
          placeholder="Traveler ID"
          value={travelerId}
          onChangeText={setTravelerId}
        />
        <TextInput
          style={styles.input}
          placeholder="Source Dimension ID"
          value={sourceDimension}
          onChangeText={setSourceDimension}
        />
        <TextInput
          style={styles.input}
          placeholder="Target Dimension ID"
          value={targetDimension}
          onChangeText={setTargetDimension}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Hopping Method:</Text>
          <Picker
            selectedValue={hoppingMethod}
            onValueChange={setHoppingMethod}
            style={styles.picker}
          >
            <Picker.Item label="Quantum Tunnel" value="quantum_tunnel" />
            <Picker.Item label="Reality Bridge" value="reality_bridge" />
            <Picker.Item label="Consciousness Transfer" value="consciousness_transfer" />
            <Picker.Item label="Matter Phase" value="matter_phase" />
            <Picker.Item label="Energy Vortex" value="energy_vortex" />
            <Picker.Item label="Dimensional Portal" value="dimensional_portal" />
            <Picker.Item label="Reality Anchor" value="reality_anchor" />
            <Picker.Item label="Quantum Leap" value="quantum_leap" />
          </Picker>
        </View>
        <Button
          title={initiateHopMutation.isLoading ? 'Hopping...' : 'Initiate Dimension Hop'}
          onPress={handleInitiateHop}
          disabled={initiateHopMutation.isLoading}
        />
      </View>

      {/* Reality Anomaly Detection Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Detect Reality Anomaly</Text>
        <TextInput
          style={styles.input}
          placeholder="Dimension ID"
          value={anomalyDimensionId}
          onChangeText={setAnomalyDimensionId}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Anomaly Type:</Text>
          <Picker
            selectedValue={anomalyType}
            onValueChange={setAnomalyType}
            style={styles.picker}
          >
            <Picker.Item label="Spatial Distortion" value="spatial_distortion" />
            <Picker.Item label="Temporal Fluctuation" value="temporal_fluctuation" />
            <Picker.Item label="Matter Instability" value="matter_instability" />
            <Picker.Item label="Energy Anomaly" value="energy_anomaly" />
            <Picker.Item label="Reality Bleed" value="reality_bleed" />
            <Picker.Item label="Dimensional Echo" value="dimensional_echo" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Severity (0.0-1.0)"
          value={anomalySeverity}
          onChangeText={setAnomalySeverity}
          keyboardType="numeric"
        />
        <TextInput
          style={[styles.input, styles.textArea]}
          placeholder="Anomaly Description"
          value={anomalyDescription}
          onChangeText={setAnomalyDescription}
          multiline
          numberOfLines={3}
        />
        <Button
          title={detectAnomalyMutation.isLoading ? 'Detecting...' : 'Detect Anomaly'}
          onPress={handleDetectAnomaly}
          disabled={detectAnomalyMutation.isLoading}
        />
      </View>

      {/* Interdimensional Entity Registration Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Register Interdimensional Entity</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity Name"
          value={entityName}
          onChangeText={setEntityName}
        />
        <TextInput
          style={styles.input}
          placeholder="Origin Dimension"
          value={entityOriginDimension}
          onChangeText={setEntityOriginDimension}
        />
        <TextInput
          style={styles.input}
          placeholder="Current Dimension"
          value={entityCurrentDimension}
          onChangeText={setEntityCurrentDimension}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Entity Type:</Text>
          <Picker
            selectedValue={entityType}
            onValueChange={setEntityType}
            style={styles.picker}
          >
            <Picker.Item label="Reality Manipulator" value="reality_manipulator" />
            <Picker.Item label="Dimensional Traveler" value="dimensional_traveler" />
            <Picker.Item label="Consciousness Entity" value="consciousness_entity" />
            <Picker.Item label="Energy Being" value="energy_being" />
            <Picker.Item label="Quantum Entity" value="quantum_entity" />
            <Picker.Item label="Transcendent Being" value="transcendent_being" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Threat Level (0.0-1.0)"
          value={entityThreatLevel}
          onChangeText={setEntityThreatLevel}
          keyboardType="numeric"
        />
        <View style={styles.switchContainer}>
          <Text style={styles.label}>Is Hostile:</Text>
          <Switch
            value={entityIsHostile}
            onValueChange={setEntityIsHostile}
          />
        </View>
        <Button
          title={registerEntityMutation.isLoading ? 'Registering...' : 'Register Entity'}
          onPress={handleRegisterEntity}
          disabled={registerEntityMutation.isLoading}
        />
      </View>

      {/* Information Retrieval Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Get Information</Text>
        <TextInput
          style={styles.input}
          placeholder="Dimension ID"
          value={selectedDimensionId}
          onChangeText={setSelectedDimensionId}
        />
        <Button title="Get Dimension Info" onPress={handleGetDimensionInfo} />
        
        {dimensionInfo && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Dimension Information:</Text>
            <Text>Name: {dimensionInfo.name}</Text>
            <Text>Type: {dimensionInfo.dimension_type}</Text>
            <Text>Reality Level: {dimensionInfo.reality_level}</Text>
            <Text>Technology Level: {dimensionInfo.technology_level}</Text>
            <Text>Stability Score: {dimensionInfo.stability_score.toFixed(2)}</Text>
            <Text>Accessible: {dimensionInfo.is_accessible ? 'Yes' : 'No'}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Traveler ID"
          value={selectedTravelerId}
          onChangeText={setSelectedTravelerId}
        />
        <Button title="Get Traveler Status" onPress={handleGetTravelerStatus} />
        
        {travelerStatus && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Traveler Status:</Text>
            <Text>Current Dimension: {travelerStatus.current_dimension}</Text>
            <Text>Total Hops: {travelerStatus.total_hops}</Text>
            {travelerStatus.recent_hops && travelerStatus.recent_hops.length > 0 && (
              <Text>Last Hop: {travelerStatus.recent_hops[0].target_dimension}</Text>
            )}
          </View>
        )}
      </View>

      {/* Statistics Section */}
      {statistics && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Service Statistics</Text>
          <View style={styles.statsContainer}>
            <Text style={styles.statItem}>Total Dimensions: {statistics.total_dimensions}</Text>
            <Text style={styles.statItem}>Accessible Dimensions: {statistics.accessible_dimensions}</Text>
            <Text style={styles.statItem}>Total Hops: {statistics.total_hops}</Text>
            <Text style={styles.statItem}>Successful Hops: {statistics.successful_hops}</Text>
            <Text style={styles.statItem}>Hop Success Rate: {statistics.hop_success_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Anomalies: {statistics.total_anomalies}</Text>
            <Text style={styles.statItem}>Contained Anomalies: {statistics.contained_anomalies}</Text>
            <Text style={styles.statItem}>Total Entities: {statistics.total_entities}</Text>
            <Text style={styles.statItem}>Hostile Entities: {statistics.hostile_entities}</Text>
            <Text style={styles.statItem}>Active Travelers: {statistics.active_travelers}</Text>
          </View>
        </View>
      )}

      {isLoadingStats && <ActivityIndicator size="large" color="#0000ff" style={styles.activityIndicator} />}
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
  textArea: {
    height: 80,
    textAlignVertical: 'top',
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

export default DimensionHoppingScreen;

