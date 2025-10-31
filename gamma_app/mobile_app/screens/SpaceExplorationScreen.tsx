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
interface Spacecraft {
  spacecraft_id: string;
  name: string;
  spacecraft_type: string;
  mission_type: string;
  current_location: string;
  target_location?: string;
  fuel_level: number;
  power_level: number;
  health_status: string;
  crew_capacity: number;
  cargo_capacity: number;
  launch_date: string;
  mission_duration: number;
  is_active: boolean;
  last_communication?: string;
}

interface SpaceMission {
  mission_id: string;
  name: string;
  mission_type: string;
  spacecraft_id: string;
  target_celestial_body: string;
  objectives: string[];
  start_date: string;
  estimated_duration: number;
  status: string;
  progress: number;
  crew_members: string[];
  resources_required: Record<string, number>;
  risks: string[];
  success_criteria: string[];
  created_at: string;
}

interface SpaceResource {
  resource_id: string;
  name: string;
  celestial_body: string;
  resource_type: string;
  abundance: number;
  extraction_difficulty: number;
  value_per_unit: number;
  discovered_date: string;
  extraction_methods: string[];
  estimated_reserves: number;
}

// API calls
const registerSpacecraft = async (spacecraftInfo: Record<string, any>): Promise<{spacecraft_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/space-time/space/spacecraft/register', spacecraftInfo);
  return response.data;
};

const createSpaceMission = async (missionInfo: Record<string, any>): Promise<{mission_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/space-time/space/missions/create', missionInfo);
  return response.data;
};

const collectSpaceData = async (dataInfo: Record<string, any>): Promise<{data_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/space-time/space/data/collect', dataInfo);
  return response.data;
};

const discoverSpaceResource = async (resourceInfo: Record<string, any>): Promise<{resource_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/space-time/space/resources/discover', resourceInfo);
  return response.data;
};

const getSpacecraftStatus = async (spacecraftId: string): Promise<Spacecraft> => {
  const response = await axios.get(`http://localhost:8000/api/v1/space-time/space/spacecraft/${spacecraftId}/status`);
  return response.data;
};

const getMissionProgress = async (missionId: string): Promise<SpaceMission> => {
  const response = await axios.get(`http://localhost:8000/api/v1/space-time/space/missions/${missionId}/progress`);
  return response.data;
};

const getSpaceStatistics = async (): Promise<Record<string, any>> => {
  const response = await axios.get('http://localhost:8000/api/v1/space-time/space/statistics');
  return response.data;
};

const SpaceExplorationScreen: React.FC = () => {
  const queryClient = useQueryClient();
  
  // State for Spacecraft Registration
  const [spacecraftName, setSpacecraftName] = useState<string>('');
  const [spacecraftType, setSpacecraftType] = useState<string>('satellite');
  const [missionType, setMissionType] = useState<string>('exploration');
  const [currentLocation, setCurrentLocation] = useState<string>('earth');
  const [targetLocation, setTargetLocation] = useState<string>('mars');
  const [fuelLevel, setFuelLevel] = useState<string>('100');
  const [powerLevel, setPowerLevel] = useState<string>('100');
  const [crewCapacity, setCrewCapacity] = useState<string>('0');
  const [cargoCapacity, setCargoCapacity] = useState<string>('0');
  
  // State for Mission Creation
  const [missionName, setMissionName] = useState<string>('');
  const [missionSpacecraftId, setMissionSpacecraftId] = useState<string>('');
  const [targetCelestialBody, setTargetCelestialBody] = useState<string>('mars');
  const [missionObjectives, setMissionObjectives] = useState<string>('');
  const [estimatedDuration, setEstimatedDuration] = useState<string>('365');
  const [crewMembers, setCrewMembers] = useState<string>('');
  
  // State for Data Collection
  const [dataSpacecraftId, setDataSpacecraftId] = useState<string>('');
  const [dataType, setDataType] = useState<string>('environmental');
  const [dataCelestialBody, setDataCelestialBody] = useState<string>('mars');
  const [dataContent, setDataContent] = useState<string>('');
  
  // State for Resource Discovery
  const [resourceName, setResourceName] = useState<string>('');
  const [resourceType, setResourceType] = useState<string>('mineral');
  const [resourceCelestialBody, setResourceCelestialBody] = useState<string>('mars');
  const [abundance, setAbundance] = useState<string>('0.5');
  const [extractionDifficulty, setExtractionDifficulty] = useState<string>('0.5');
  const [valuePerUnit, setValuePerUnit] = useState<string>('100');
  
  // State for tracking
  const [activeSpacecraftId, setActiveSpacecraftId] = useState<string | null>(null);
  const [activeMissionId, setActiveMissionId] = useState<string | null>(null);
  
  // State for modals
  const [showSpacecraftModal, setShowSpacecraftModal] = useState<boolean>(false);
  const [showMissionModal, setShowMissionModal] = useState<boolean>(false);
  const [showDataModal, setShowDataModal] = useState<boolean>(false);
  const [showResourceModal, setShowResourceModal] = useState<boolean>(false);
  
  // Options
  const spacecraftTypes = [
    { value: 'satellite', label: 'Satellite' },
    { value: 'rover', label: 'Rover' },
    { value: 'probe', label: 'Probe' },
    { value: 'station', label: 'Space Station' },
    { value: 'shuttle', label: 'Space Shuttle' },
    { value: 'rocket', label: 'Rocket' },
    { value: 'telescope', label: 'Telescope' },
    { value: 'mining_vehicle', label: 'Mining Vehicle' }
  ];
  
  const missionTypes = [
    { value: 'exploration', label: 'Exploration' },
    { value: 'research', label: 'Research' },
    { value: 'communication', label: 'Communication' },
    { value: 'observation', label: 'Observation' },
    { value: 'mining', label: 'Mining' },
    { value: 'colonization', label: 'Colonization' },
    { value: 'defense', label: 'Defense' },
    { value: 'transport', label: 'Transport' }
  ];
  
  const celestialBodies = [
    { value: 'mercury', label: 'Mercury' },
    { value: 'venus', label: 'Venus' },
    { value: 'earth', label: 'Earth' },
    { value: 'mars', label: 'Mars' },
    { value: 'jupiter', label: 'Jupiter' },
    { value: 'saturn', label: 'Saturn' },
    { value: 'uranus', label: 'Uranus' },
    { value: 'neptune', label: 'Neptune' },
    { value: 'pluto', label: 'Pluto' },
    { value: 'moon', label: 'Moon' },
    { value: 'europa', label: 'Europa' },
    { value: 'titan', label: 'Titan' },
    { value: 'asteroid_belt', label: 'Asteroid Belt' },
    { value: 'kuiper_belt', label: 'Kuiper Belt' }
  ];
  
  const dataTypes = [
    { value: 'environmental', label: 'Environmental' },
    { value: 'geological', label: 'Geological' },
    { value: 'biological', label: 'Biological' },
    { value: 'atmospheric', label: 'Atmospheric' },
    { value: 'magnetic', label: 'Magnetic' },
    { value: 'gravitational', label: 'Gravitational' },
    { value: 'radiation', label: 'Radiation' },
    { value: 'mineral', label: 'Mineral' }
  ];
  
  const resourceTypes = [
    { value: 'mineral', label: 'Mineral' },
    { value: 'water', label: 'Water' },
    { value: 'gas', label: 'Gas' },
    { value: 'metal', label: 'Metal' },
    { value: 'nuclear_fuel', label: 'Nuclear Fuel' },
    { value: 'organic', label: 'Organic' },
    { value: 'rare_earth', label: 'Rare Earth' },
    { value: 'energy', label: 'Energy' }
  ];
  
  // Queries
  const { data: statistics, isLoading: isLoadingStats } = useQuery<Record<string, any>, Error>(
    'spaceStatistics',
    getSpaceStatistics,
    {
      refetchInterval: 30000,
    }
  );
  
  const { data: spacecraftStatus, isLoading: isLoadingSpacecraft } = useQuery<Spacecraft, Error>(
    ['spacecraftStatus', activeSpacecraftId],
    () => activeSpacecraftId ? getSpacecraftStatus(activeSpacecraftId) : Promise.resolve(null),
    {
      enabled: !!activeSpacecraftId,
      refetchInterval: 5000,
    }
  );
  
  const { data: missionProgress, isLoading: isLoadingMission } = useQuery<SpaceMission, Error>(
    ['missionProgress', activeMissionId],
    () => activeMissionId ? getMissionProgress(activeMissionId) : Promise.resolve(null),
    {
      enabled: !!activeMissionId,
      refetchInterval: 10000,
    }
  );
  
  // Mutations
  const registerSpacecraftMutation = useMutation<{spacecraft_id: string}, Error, Record<string, any>>(registerSpacecraft, {
    onSuccess: (data) => {
      setActiveSpacecraftId(data.spacecraft_id);
      Alert.alert('Éxito', `Nave espacial registrada: ${data.spacecraft_id}`);
      setShowSpacecraftModal(false);
      queryClient.invalidateQueries('spaceStatistics');
    },
    onError: (error) => {
      Alert.alert('Error', `Error al registrar nave espacial: ${error.message}`);
    },
  });
  
  const createMissionMutation = useMutation<{mission_id: string}, Error, Record<string, any>>(createSpaceMission, {
    onSuccess: (data) => {
      setActiveMissionId(data.mission_id);
      Alert.alert('Éxito', `Misión espacial creada: ${data.mission_id}`);
      setShowMissionModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al crear misión: ${error.message}`);
    },
  });
  
  const collectDataMutation = useMutation<{data_id: string}, Error, Record<string, any>>(collectSpaceData, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Datos espaciales recolectados: ${data.data_id}`);
      setShowDataModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al recolectar datos: ${error.message}`);
    },
  });
  
  const discoverResourceMutation = useMutation<{resource_id: string}, Error, Record<string, any>>(discoverSpaceResource, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Recurso espacial descubierto: ${data.resource_id}`);
      setShowResourceModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al descubrir recurso: ${error.message}`);
    },
  });
  
  // Handlers
  const handleRegisterSpacecraft = () => {
    if (!spacecraftName.trim()) {
      Alert.alert('Error', 'Por favor ingresa un nombre para la nave espacial.');
      return;
    }
    
    registerSpacecraftMutation.mutate({
      name: spacecraftName,
      spacecraft_type: spacecraftType,
      mission_type: missionType,
      current_location: currentLocation,
      target_location: targetLocation,
      fuel_level: parseFloat(fuelLevel),
      power_level: parseFloat(powerLevel),
      crew_capacity: parseInt(crewCapacity, 10),
      cargo_capacity: parseFloat(cargoCapacity),
      mission_duration: 365
    });
  };
  
  const handleCreateMission = () => {
    if (!missionName.trim() || !missionSpacecraftId.trim()) {
      Alert.alert('Error', 'Por favor completa el nombre de la misión y el ID de la nave espacial.');
      return;
    }
    
    createMissionMutation.mutate({
      name: missionName,
      spacecraft_id: missionSpacecraftId,
      mission_type: missionType,
      target_celestial_body: targetCelestialBody,
      objectives: missionObjectives.split('\n').filter(obj => obj.trim()),
      estimated_duration: parseInt(estimatedDuration, 10),
      crew_members: crewMembers.split(',').map(member => member.trim()).filter(member => member),
      resources_required: {},
      risks: [],
      success_criteria: []
    });
  };
  
  const handleCollectData = () => {
    if (!dataSpacecraftId.trim() || !dataContent.trim()) {
      Alert.alert('Error', 'Por favor completa el ID de la nave espacial y el contenido de los datos.');
      return;
    }
    
    try {
      const content = JSON.parse(dataContent);
      collectDataMutation.mutate({
        spacecraft_id: dataSpacecraftId,
        data_type: dataType,
        celestial_body: dataCelestialBody,
        data_content: content,
        quality_score: 0.8
      });
    } catch (error) {
      Alert.alert('Error', 'El contenido de los datos debe ser un JSON válido.');
    }
  };
  
  const handleDiscoverResource = () => {
    if (!resourceName.trim()) {
      Alert.alert('Error', 'Por favor ingresa un nombre para el recurso.');
      return;
    }
    
    discoverResourceMutation.mutate({
      name: resourceName,
      resource_type: resourceType,
      celestial_body: resourceCelestialBody,
      abundance: parseFloat(abundance),
      extraction_difficulty: parseFloat(extractionDifficulty),
      value_per_unit: parseFloat(valuePerUnit),
      extraction_methods: ['drilling', 'mining'],
      estimated_reserves: 1000.0
    });
  };
  
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Space Exploration</Text>
      <Text style={styles.subtitle}>Exploración Espacial y Gestión de Misiones</Text>
      
      {/* Statistics */}
      {statistics && (
        <View style={styles.statisticsContainer}>
          <Text style={styles.sectionTitle}>Estadísticas del Sistema</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_spacecraft || 0}</Text>
              <Text style={styles.statLabel}>Naves Espaciales</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_missions || 0}</Text>
              <Text style={styles.statLabel}>Misiones</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_data_points || 0}</Text>
              <Text style={styles.statLabel}>Puntos de Datos</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_resources || 0}</Text>
              <Text style={styles.statLabel}>Recursos</Text>
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
            onPress={() => setShowSpacecraftModal(true)}
          >
            <Text style={styles.actionButtonText}>Registrar Nave</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#2196F3' }]}
            onPress={() => setShowMissionModal(true)}
          >
            <Text style={styles.actionButtonText}>Crear Misión</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#FF9800' }]}
            onPress={() => setShowDataModal(true)}
          >
            <Text style={styles.actionButtonText}>Recolectar Datos</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#9C27B0' }]}
            onPress={() => setShowResourceModal(true)}
          >
            <Text style={styles.actionButtonText}>Descubrir Recurso</Text>
          </TouchableOpacity>
        </View>
      </View>
      
      {/* Spacecraft Status */}
      {spacecraftStatus && (
        <View style={styles.statusContainer}>
          <Text style={styles.sectionTitle}>Estado de la Nave Espacial</Text>
          <View style={styles.statusCard}>
            <Text style={styles.statusTitle}>Nave: {spacecraftStatus.name}</Text>
            <Text style={styles.statusInfo}>Tipo: {spacecraftStatus.spacecraft_type}</Text>
            <Text style={styles.statusInfo}>Misión: {spacecraftStatus.mission_type}</Text>
            <Text style={styles.statusInfo}>Ubicación: {spacecraftStatus.current_location}</Text>
            <Text style={styles.statusInfo}>Combustible: {spacecraftStatus.fuel_level.toFixed(1)}%</Text>
            <Text style={styles.statusInfo}>Energía: {spacecraftStatus.power_level.toFixed(1)}%</Text>
            <Text style={[styles.statusInfo, { color: spacecraftStatus.is_active ? '#4CAF50' : '#FF9800' }]}>
              Estado: {spacecraftStatus.is_active ? 'Activa' : 'Inactiva'}
            </Text>
            <Text style={styles.statusInfo}>Tripulación: {spacecraftStatus.crew_capacity}</Text>
            <Text style={styles.statusInfo}>Carga: {spacecraftStatus.cargo_capacity} kg</Text>
          </View>
        </View>
      )}
      
      {/* Mission Progress */}
      {missionProgress && (
        <View style={styles.statusContainer}>
          <Text style={styles.sectionTitle}>Progreso de la Misión</Text>
          <View style={styles.statusCard}>
            <Text style={styles.statusTitle}>Misión: {missionProgress.name}</Text>
            <Text style={styles.statusInfo}>Tipo: {missionProgress.mission_type}</Text>
            <Text style={styles.statusInfo}>Objetivo: {missionProgress.target_celestial_body}</Text>
            <Text style={styles.statusInfo}>Duración: {missionProgress.estimated_duration} días</Text>
            <Text style={[styles.statusInfo, { color: missionProgress.status === 'completed' ? '#4CAF50' : '#FF9800' }]}>
              Estado: {missionProgress.status}
            </Text>
            <Text style={styles.statusInfo}>Progreso: {missionProgress.progress.toFixed(1)}%</Text>
            <Text style={styles.statusInfo}>Tripulación: {missionProgress.crew_members.length}</Text>
            <Text style={styles.statusInfo}>Objetivos: {missionProgress.objectives.length}</Text>
          </View>
        </View>
      )}
      
      {/* Spacecraft Registration Modal */}
      <Modal
        visible={showSpacecraftModal}
        animationType="slide"
        onRequestClose={() => setShowSpacecraftModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Registrar Nave Espacial</Text>
            <Button title="Cerrar" onPress={() => setShowSpacecraftModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Nombre de la Nave:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: Mars Rover Perseverance"
              value={spacecraftName}
              onChangeText={setSpacecraftName}
            />
            
            <Text style={styles.label}>Tipo de Nave:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={spacecraftType}
                onValueChange={setSpacecraftType}
                style={styles.picker}
              >
                {spacecraftTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Tipo de Misión:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={missionType}
                onValueChange={setMissionType}
                style={styles.picker}
              >
                {missionTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Ubicación Actual:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={currentLocation}
                onValueChange={setCurrentLocation}
                style={styles.picker}
              >
                {celestialBodies.map((body) => (
                  <Picker.Item key={body.value} label={body.label} value={body.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Ubicación Objetivo:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={targetLocation}
                onValueChange={setTargetLocation}
                style={styles.picker}
              >
                {celestialBodies.map((body) => (
                  <Picker.Item key={body.value} label={body.label} value={body.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Nivel de Combustible (%):</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={fuelLevel}
                onValueChange={setFuelLevel}
                style={styles.picker}
              >
                <Picker.Item label="25%" value="25" />
                <Picker.Item label="50%" value="50" />
                <Picker.Item label="75%" value="75" />
                <Picker.Item label="100%" value="100" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Nivel de Energía (%):</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={powerLevel}
                onValueChange={setPowerLevel}
                style={styles.picker}
              >
                <Picker.Item label="25%" value="25" />
                <Picker.Item label="50%" value="50" />
                <Picker.Item label="75%" value="75" />
                <Picker.Item label="100%" value="100" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Capacidad de Tripulación:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={crewCapacity}
                onValueChange={setCrewCapacity}
                style={styles.picker}
              >
                <Picker.Item label="0" value="0" />
                <Picker.Item label="1" value="1" />
                <Picker.Item label="2" value="2" />
                <Picker.Item label="3" value="3" />
                <Picker.Item label="4" value="4" />
                <Picker.Item label="5" value="5" />
                <Picker.Item label="6" value="6" />
                <Picker.Item label="7" value="7" />
                <Picker.Item label="8" value="8" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Capacidad de Carga (kg):</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={cargoCapacity}
                onValueChange={setCargoCapacity}
                style={styles.picker}
              >
                <Picker.Item label="0 kg" value="0" />
                <Picker.Item label="100 kg" value="100" />
                <Picker.Item label="500 kg" value="500" />
                <Picker.Item label="1000 kg" value="1000" />
                <Picker.Item label="5000 kg" value="5000" />
                <Picker.Item label="10000 kg" value="10000" />
              </Picker>
            </View>
            
            <Button
              title={registerSpacecraftMutation.isLoading ? 'Registrando...' : 'Registrar Nave Espacial'}
              onPress={handleRegisterSpacecraft}
              disabled={registerSpacecraftMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Mission Creation Modal */}
      <Modal
        visible={showMissionModal}
        animationType="slide"
        onRequestClose={() => setShowMissionModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Crear Misión Espacial</Text>
            <Button title="Cerrar" onPress={() => setShowMissionModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Nombre de la Misión:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: Misión a Marte 2024"
              value={missionName}
              onChangeText={setMissionName}
            />
            
            <Text style={styles.label}>ID de la Nave Espacial:</Text>
            <TextInput
              style={styles.input}
              placeholder="spacecraft_id_here"
              value={missionSpacecraftId}
              onChangeText={setMissionSpacecraftId}
            />
            
            <Text style={styles.label}>Tipo de Misión:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={missionType}
                onValueChange={setMissionType}
                style={styles.picker}
              >
                {missionTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Cuerpo Celeste Objetivo:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={targetCelestialBody}
                onValueChange={setTargetCelestialBody}
                style={styles.picker}
              >
                {celestialBodies.map((body) => (
                  <Picker.Item key={body.value} label={body.label} value={body.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Objetivos (uno por línea):</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Explorar superficie&#10;Recolectar muestras&#10;Establecer comunicación"
              value={missionObjectives}
              onChangeText={setMissionObjectives}
              multiline
              numberOfLines={4}
            />
            
            <Text style={styles.label}>Duración Estimada (días):</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={estimatedDuration}
                onValueChange={setEstimatedDuration}
                style={styles.picker}
              >
                <Picker.Item label="30 días" value="30" />
                <Picker.Item label="90 días" value="90" />
                <Picker.Item label="180 días" value="180" />
                <Picker.Item label="365 días" value="365" />
                <Picker.Item label="730 días" value="730" />
                <Picker.Item label="1095 días" value="1095" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Miembros de la Tripulación (separados por comas):</Text>
            <TextInput
              style={styles.input}
              placeholder="Astronauta 1, Astronauta 2, Astronauta 3"
              value={crewMembers}
              onChangeText={setCrewMembers}
            />
            
            <Button
              title={createMissionMutation.isLoading ? 'Creando...' : 'Crear Misión Espacial'}
              onPress={handleCreateMission}
              disabled={createMissionMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Data Collection Modal */}
      <Modal
        visible={showDataModal}
        animationType="slide"
        onRequestClose={() => setShowDataModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Recolectar Datos Espaciales</Text>
            <Button title="Cerrar" onPress={() => setShowDataModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>ID de la Nave Espacial:</Text>
            <TextInput
              style={styles.input}
              placeholder="spacecraft_id_here"
              value={dataSpacecraftId}
              onChangeText={setDataSpacecraftId}
            />
            
            <Text style={styles.label}>Tipo de Datos:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={dataType}
                onValueChange={setDataType}
                style={styles.picker}
              >
                {dataTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Cuerpo Celeste:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={dataCelestialBody}
                onValueChange={setDataCelestialBody}
                style={styles.picker}
              >
                {celestialBodies.map((body) => (
                  <Picker.Item key={body.value} label={body.label} value={body.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Contenido de los Datos (JSON):</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder='{"temperature": -50, "pressure": 0.6, "atmosphere": "CO2"}'
              value={dataContent}
              onChangeText={setDataContent}
              multiline
              numberOfLines={4}
            />
            
            <Button
              title={collectDataMutation.isLoading ? 'Recolectando...' : 'Recolectar Datos'}
              onPress={handleCollectData}
              disabled={collectDataMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Resource Discovery Modal */}
      <Modal
        visible={showResourceModal}
        animationType="slide"
        onRequestClose={() => setShowResourceModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Descubrir Recurso Espacial</Text>
            <Button title="Cerrar" onPress={() => setShowResourceModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Nombre del Recurso:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: Agua Helada"
              value={resourceName}
              onChangeText={setResourceName}
            />
            
            <Text style={styles.label}>Tipo de Recurso:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={resourceType}
                onValueChange={setResourceType}
                style={styles.picker}
              >
                {resourceTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Cuerpo Celeste:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={resourceCelestialBody}
                onValueChange={setResourceCelestialBody}
                style={styles.picker}
              >
                {celestialBodies.map((body) => (
                  <Picker.Item key={body.value} label={body.label} value={body.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Abundancia (0.0 - 1.0):</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={abundance}
                onValueChange={setAbundance}
                style={styles.picker}
              >
                <Picker.Item label="0.1" value="0.1" />
                <Picker.Item label="0.2" value="0.2" />
                <Picker.Item label="0.3" value="0.3" />
                <Picker.Item label="0.4" value="0.4" />
                <Picker.Item label="0.5" value="0.5" />
                <Picker.Item label="0.6" value="0.6" />
                <Picker.Item label="0.7" value="0.7" />
                <Picker.Item label="0.8" value="0.8" />
                <Picker.Item label="0.9" value="0.9" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Dificultad de Extracción (0.0 - 1.0):</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={extractionDifficulty}
                onValueChange={setExtractionDifficulty}
                style={styles.picker}
              >
                <Picker.Item label="0.1" value="0.1" />
                <Picker.Item label="0.2" value="0.2" />
                <Picker.Item label="0.3" value="0.3" />
                <Picker.Item label="0.4" value="0.4" />
                <Picker.Item label="0.5" value="0.5" />
                <Picker.Item label="0.6" value="0.6" />
                <Picker.Item label="0.7" value="0.7" />
                <Picker.Item label="0.8" value="0.8" />
                <Picker.Item label="0.9" value="0.9" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Valor por Unidad:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={valuePerUnit}
                onValueChange={setValuePerUnit}
                style={styles.picker}
              >
                <Picker.Item label="$10" value="10" />
                <Picker.Item label="$50" value="50" />
                <Picker.Item label="$100" value="100" />
                <Picker.Item label="$500" value="500" />
                <Picker.Item label="$1000" value="1000" />
                <Picker.Item label="$5000" value="5000" />
                <Picker.Item label="$10000" value="10000" />
              </Picker>
            </View>
            
            <Button
              title={discoverResourceMutation.isLoading ? 'Descubriendo...' : 'Descubrir Recurso'}
              onPress={handleDiscoverResource}
              disabled={discoverResourceMutation.isLoading}
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

export default SpaceExplorationScreen;


