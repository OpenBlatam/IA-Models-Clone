import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, ActivityIndicator, Alert, Switch } from 'react-native';
import axios from 'axios';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Picker } from '@react-native-picker/picker';
import { format } from 'date-fns';

// API Base URL
const API_BASE_URL = 'http://localhost:8000/api/v1';

// Interface definitions
interface CosmicEntity {
  entity_id: string;
  name: string;
  cosmic_level: string;
  cosmic_force: string;
  cosmic_state: string;
  cosmic_energy: number;
  universal_consciousness: number;
  cosmic_harmony: number;
  ultimate_reality: number;
  cosmic_wisdom: number;
  universal_love: number;
  cosmic_balance: number;
  ultimate_connection: number;
  is_evolving: boolean;
  last_evolution?: string;
  evolution_history_count: number;
}

interface CosmicEvolution {
  evolution_id: string;
  entity_id: string;
  evolution_type: string;
  from_level: string;
  to_level: string;
  cosmic_force: number;
  success: boolean;
  side_effects: string[];
  duration: number;
  timestamp: string;
}

interface UniversalHarmony {
  harmony_id: string;
  entity_id: string;
  harmony_type: string;
  cosmic_balance: number;
  universal_frequency: number;
  harmony_effects: Record<string, any>;
  is_active: boolean;
  created_at: string;
}

interface UniversalEntity {
  entity_id: string;
  name: string;
  universal_level: string;
  universal_force: string;
  universal_state: string;
  universal_consciousness: number;
  infinite_expansion: number;
  universal_harmony: number;
  ultimate_unity: number;
  universal_wisdom: number;
  infinite_love: number;
  universal_balance: number;
  infinite_connection: number;
  is_expanding: boolean;
  last_expansion?: string;
  expansion_history_count: number;
}

interface UniversalExpansion {
  expansion_id: string;
  entity_id: string;
  expansion_type: string;
  from_level: string;
  to_level: string;
  expansion_force: number;
  success: boolean;
  side_effects: string[];
  duration: number;
  timestamp: string;
}

interface InfiniteUnity {
  unity_id: string;
  entity_id: string;
  unity_type: string;
  universal_balance: number;
  infinite_frequency: number;
  unity_effects: Record<string, any>;
  is_active: boolean;
  created_at: string;
}

// API functions
const createCosmicEntity = async (entityInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/ultimate-cosmic/cosmic/entities/create`, entityInfo);
  return response.data;
};

const initiateCosmicEvolution = async (evolutionInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/ultimate-cosmic/cosmic/evolution/initiate`, evolutionInfo);
  return response.data;
};

const createUniversalHarmony = async (harmonyInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/ultimate-cosmic/cosmic/harmony/create`, harmonyInfo);
  return response.data;
};

const createUniversalEntity = async (entityInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/ultimate-cosmic/universal/entities/create`, entityInfo);
  return response.data;
};

const initiateUniversalExpansion = async (expansionInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/ultimate-cosmic/universal/expansion/initiate`, expansionInfo);
  return response.data;
};

const createInfiniteUnity = async (unityInfo: any) => {
  const response = await axios.post(`${API_BASE_URL}/ultimate-cosmic/universal/unity/create`, unityInfo);
  return response.data;
};

const getCosmicEntityStatus = async (entityId: string): Promise<CosmicEntity> => {
  const response = await axios.get(`${API_BASE_URL}/ultimate-cosmic/cosmic/entities/${entityId}/status`);
  return response.data;
};

const getCosmicEvolutionProgress = async (evolutionId: string) => {
  const response = await axios.get(`${API_BASE_URL}/ultimate-cosmic/cosmic/evolution/${evolutionId}/progress`);
  return response.data;
};

const getHarmonyStatus = async (harmonyId: string): Promise<UniversalHarmony> => {
  const response = await axios.get(`${API_BASE_URL}/ultimate-cosmic/cosmic/harmony/${harmonyId}/status`);
  return response.data;
};

const getUniversalEntityStatus = async (entityId: string): Promise<UniversalEntity> => {
  const response = await axios.get(`${API_BASE_URL}/ultimate-cosmic/universal/entities/${entityId}/status`);
  return response.data;
};

const getUniversalExpansionProgress = async (expansionId: string) => {
  const response = await axios.get(`${API_BASE_URL}/ultimate-cosmic/universal/expansion/${expansionId}/progress`);
  return response.data;
};

const getUnityStatus = async (unityId: string): Promise<InfiniteUnity> => {
  const response = await axios.get(`${API_BASE_URL}/ultimate-cosmic/universal/unity/${unityId}/status`);
  return response.data;
};

const getCosmicStatistics = async () => {
  const response = await axios.get(`${API_BASE_URL}/ultimate-cosmic/cosmic/statistics`);
  return response.data;
};

const getUniversalStatistics = async () => {
  const response = await axios.get(`${API_BASE_URL}/ultimate-cosmic/universal/statistics`);
  return response.data;
};

const UltimateCosmicScreen: React.FC = () => {
  const queryClient = useQueryClient();
  
  // State for cosmic entity creation
  const [cosmicEntityName, setCosmicEntityName] = useState<string>('');
  const [cosmicLevel, setCosmicLevel] = useState<string>('planetary');
  const [cosmicForce, setCosmicForce] = useState<string>('gravity');
  const [cosmicState, setCosmicState] = useState<string>('birth');
  const [cosmicEnergy, setCosmicEnergy] = useState<string>('0.5');
  const [universalConsciousness, setUniversalConsciousness] = useState<string>('0.5');
  
  // State for cosmic evolution
  const [evolutionEntityId, setEvolutionEntityId] = useState<string>('');
  const [evolutionType, setEvolutionType] = useState<string>('cosmic_evolution');
  const [evolutionFromLevel, setEvolutionFromLevel] = useState<string>('planetary');
  const [evolutionToLevel, setEvolutionToLevel] = useState<string>('stellar');
  const [evolutionForce, setEvolutionForce] = useState<string>('100');
  const [evolutionDuration, setEvolutionDuration] = useState<string>('3600');
  
  // State for universal harmony
  const [harmonyEntityId, setHarmonyEntityId] = useState<string>('');
  const [harmonyType, setHarmonyType] = useState<string>('cosmic_harmony');
  const [cosmicBalance, setCosmicBalance] = useState<string>('0.5');
  const [universalFrequency, setUniversalFrequency] = useState<string>('0.5');
  
  // State for universal entity
  const [universalEntityName, setUniversalEntityName] = useState<string>('');
  const [universalLevel, setUniversalLevel] = useState<string>('local');
  const [universalForce, setUniversalForce] = useState<string>('unity');
  const [universalState, setUniversalState] = useState<string>('birth');
  const [universalConsciousness, setUniversalConsciousness] = useState<string>('0.5');
  const [infiniteExpansion, setInfiniteExpansion] = useState<string>('0.5');
  
  // State for universal expansion
  const [expansionEntityId, setExpansionEntityId] = useState<string>('');
  const [expansionType, setExpansionType] = useState<string>('universal_expansion');
  const [expansionFromLevel, setExpansionFromLevel] = useState<string>('local');
  const [expansionToLevel, setExpansionToLevel] = useState<string>('regional');
  const [expansionForce, setExpansionForce] = useState<string>('100');
  const [expansionDuration, setExpansionDuration] = useState<string>('3600');
  
  // State for infinite unity
  const [unityEntityId, setUnityEntityId] = useState<string>('');
  const [unityType, setUnityType] = useState<string>('universal_unity');
  const [unityBalance, setUnityBalance] = useState<string>('0.5');
  const [infiniteFrequency, setInfiniteFrequency] = useState<string>('0.5');
  
  // State for display
  const [selectedCosmicEntityId, setSelectedCosmicEntityId] = useState<string>('');
  const [selectedEvolutionId, setSelectedEvolutionId] = useState<string>('');
  const [selectedHarmonyId, setSelectedHarmonyId] = useState<string>('');
  const [selectedUniversalEntityId, setSelectedUniversalEntityId] = useState<string>('');
  const [selectedExpansionId, setSelectedExpansionId] = useState<string>('');
  const [selectedUnityId, setSelectedUnityId] = useState<string>('');
  const [cosmicEntityStatus, setCosmicEntityStatus] = useState<CosmicEntity | null>(null);
  const [evolutionProgress, setEvolutionProgress] = useState<any>(null);
  const [harmonyStatus, setHarmonyStatus] = useState<UniversalHarmony | null>(null);
  const [universalEntityStatus, setUniversalEntityStatus] = useState<UniversalEntity | null>(null);
  const [expansionProgress, setExpansionProgress] = useState<any>(null);
  const [unityStatus, setUnityStatus] = useState<InfiniteUnity | null>(null);
  const [cosmicStats, setCosmicStats] = useState<any>(null);
  const [universalStats, setUniversalStats] = useState<any>(null);

  // Queries
  const { data: cosmicStatistics, isLoading: isLoadingCosmicStats } = useQuery(
    'cosmicStatistics',
    getCosmicStatistics,
    { refetchInterval: 5000 }
  );

  const { data: universalStatistics, isLoading: isLoadingUniversalStats } = useQuery(
    'universalStatistics',
    getUniversalStatistics,
    { refetchInterval: 5000 }
  );

  // Mutations
  const createCosmicEntityMutation = useMutation(createCosmicEntity, {
    onSuccess: (data) => {
      Alert.alert('Success', `Cosmic entity created: ${data.entity_id}`);
      queryClient.invalidateQueries('cosmicStatistics');
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to create cosmic entity: ${error.message}`);
    },
  });

  const initiateCosmicEvolutionMutation = useMutation(initiateCosmicEvolution, {
    onSuccess: (data) => {
      Alert.alert('Success', `Cosmic evolution initiated: ${data.evolution_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to initiate cosmic evolution: ${error.message}`);
    },
  });

  const createUniversalHarmonyMutation = useMutation(createUniversalHarmony, {
    onSuccess: (data) => {
      Alert.alert('Success', `Universal harmony created: ${data.harmony_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to create universal harmony: ${error.message}`);
    },
  });

  const createUniversalEntityMutation = useMutation(createUniversalEntity, {
    onSuccess: (data) => {
      Alert.alert('Success', `Universal entity created: ${data.entity_id}`);
      queryClient.invalidateQueries('universalStatistics');
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to create universal entity: ${error.message}`);
    },
  });

  const initiateUniversalExpansionMutation = useMutation(initiateUniversalExpansion, {
    onSuccess: (data) => {
      Alert.alert('Success', `Universal expansion initiated: ${data.expansion_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to initiate universal expansion: ${error.message}`);
    },
  });

  const createInfiniteUnityMutation = useMutation(createInfiniteUnity, {
    onSuccess: (data) => {
      Alert.alert('Success', `Infinite unity created: ${data.unity_id}`);
    },
    onError: (error: any) => {
      Alert.alert('Error', `Failed to create infinite unity: ${error.message}`);
    },
  });

  // Handlers
  const handleCreateCosmicEntity = () => {
    if (!cosmicEntityName.trim()) {
      Alert.alert('Input Error', 'Please enter an entity name.');
      return;
    }

    const entityInfo = {
      name: cosmicEntityName,
      cosmic_level: cosmicLevel,
      cosmic_force: cosmicForce,
      cosmic_state: cosmicState,
      cosmic_energy: parseFloat(cosmicEnergy),
      universal_consciousness: parseFloat(universalConsciousness),
      cosmic_harmony: 0.5,
      ultimate_reality: 0.5,
      cosmic_wisdom: 0.5,
      universal_love: 0.5,
      cosmic_balance: 0.5,
      ultimate_connection: 0.5
    };

    createCosmicEntityMutation.mutate(entityInfo);
  };

  const handleInitiateCosmicEvolution = () => {
    if (!evolutionEntityId.trim()) {
      Alert.alert('Input Error', 'Please enter an entity ID.');
      return;
    }

    const evolutionInfo = {
      entity_id: evolutionEntityId,
      evolution_type: evolutionType,
      from_level: evolutionFromLevel,
      to_level: evolutionToLevel,
      cosmic_force: parseFloat(evolutionForce),
      duration: parseFloat(evolutionDuration)
    };

    initiateCosmicEvolutionMutation.mutate(evolutionInfo);
  };

  const handleCreateUniversalHarmony = () => {
    if (!harmonyEntityId.trim()) {
      Alert.alert('Input Error', 'Please enter an entity ID.');
      return;
    }

    const harmonyInfo = {
      entity_id: harmonyEntityId,
      harmony_type: harmonyType,
      cosmic_balance: parseFloat(cosmicBalance),
      universal_frequency: parseFloat(universalFrequency),
      harmony_effects: {}
    };

    createUniversalHarmonyMutation.mutate(harmonyInfo);
  };

  const handleCreateUniversalEntity = () => {
    if (!universalEntityName.trim()) {
      Alert.alert('Input Error', 'Please enter an entity name.');
      return;
    }

    const entityInfo = {
      name: universalEntityName,
      universal_level: universalLevel,
      universal_force: universalForce,
      universal_state: universalState,
      universal_consciousness: parseFloat(universalConsciousness),
      infinite_expansion: parseFloat(infiniteExpansion),
      universal_harmony: 0.5,
      ultimate_unity: 0.5,
      universal_wisdom: 0.5,
      infinite_love: 0.5,
      universal_balance: 0.5,
      infinite_connection: 0.5
    };

    createUniversalEntityMutation.mutate(entityInfo);
  };

  const handleInitiateUniversalExpansion = () => {
    if (!expansionEntityId.trim()) {
      Alert.alert('Input Error', 'Please enter an entity ID.');
      return;
    }

    const expansionInfo = {
      entity_id: expansionEntityId,
      expansion_type: expansionType,
      from_level: expansionFromLevel,
      to_level: expansionToLevel,
      expansion_force: parseFloat(expansionForce),
      duration: parseFloat(expansionDuration)
    };

    initiateUniversalExpansionMutation.mutate(expansionInfo);
  };

  const handleCreateInfiniteUnity = () => {
    if (!unityEntityId.trim()) {
      Alert.alert('Input Error', 'Please enter an entity ID.');
      return;
    }

    const unityInfo = {
      entity_id: unityEntityId,
      unity_type: unityType,
      universal_balance: parseFloat(unityBalance),
      infinite_frequency: parseFloat(infiniteFrequency),
      unity_effects: {}
    };

    createInfiniteUnityMutation.mutate(unityInfo);
  };

  const handleGetCosmicEntityStatus = async () => {
    try {
      const status = await getCosmicEntityStatus(selectedCosmicEntityId);
      setCosmicEntityStatus(status);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get cosmic entity status: ${error.message}`);
    }
  };

  const handleGetEvolutionProgress = async () => {
    try {
      const progress = await getCosmicEvolutionProgress(selectedEvolutionId);
      setEvolutionProgress(progress);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get evolution progress: ${error.message}`);
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

  const handleGetUniversalEntityStatus = async () => {
    try {
      const status = await getUniversalEntityStatus(selectedUniversalEntityId);
      setUniversalEntityStatus(status);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get universal entity status: ${error.message}`);
    }
  };

  const handleGetExpansionProgress = async () => {
    try {
      const progress = await getUniversalExpansionProgress(selectedExpansionId);
      setExpansionProgress(progress);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get expansion progress: ${error.message}`);
    }
  };

  const handleGetUnityStatus = async () => {
    try {
      const status = await getUnityStatus(selectedUnityId);
      setUnityStatus(status);
    } catch (error: any) {
      Alert.alert('Error', `Failed to get unity status: ${error.message}`);
    }
  };

  useEffect(() => {
    if (cosmicStatistics) {
      setCosmicStats(cosmicStatistics);
    }
  }, [cosmicStatistics]);

  useEffect(() => {
    if (universalStatistics) {
      setUniversalStats(universalStatistics);
    }
  }, [universalStatistics]);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Ultimate Cosmic & Infinite Universal</Text>
      
      {/* Cosmic Entity Creation Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Create Cosmic Entity</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity Name"
          value={cosmicEntityName}
          onChangeText={setCosmicEntityName}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Cosmic Level:</Text>
          <Picker
            selectedValue={cosmicLevel}
            onValueChange={setCosmicLevel}
            style={styles.picker}
          >
            <Picker.Item label="Planetary" value="planetary" />
            <Picker.Item label="Stellar" value="stellar" />
            <Picker.Item label="Galactic" value="galactic" />
            <Picker.Item label="Universal" value="universal" />
            <Picker.Item label="Multiversal" value="multiversal" />
            <Picker.Item label="Omniversal" value="omniversal" />
            <Picker.Item label="Cosmic" value="cosmic" />
            <Picker.Item label="Ultimate" value="ultimate" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Cosmic Force:</Text>
          <Picker
            selectedValue={cosmicForce}
            onValueChange={setCosmicForce}
            style={styles.picker}
          >
            <Picker.Item label="Gravity" value="gravity" />
            <Picker.Item label="Electromagnetism" value="electromagnetism" />
            <Picker.Item label="Strong Nuclear" value="strong_nuclear" />
            <Picker.Item label="Weak Nuclear" value="weak_nuclear" />
            <Picker.Item label="Dark Matter" value="dark_matter" />
            <Picker.Item label="Dark Energy" value="dark_energy" />
            <Picker.Item label="Consciousness" value="consciousness" />
            <Picker.Item label="Ultimate" value="ultimate" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Cosmic State:</Text>
          <Picker
            selectedValue={cosmicState}
            onValueChange={setCosmicState}
            style={styles.picker}
          >
            <Picker.Item label="Birth" value="birth" />
            <Picker.Item label="Expansion" value="expansion" />
            <Picker.Item label="Stability" value="stability" />
            <Picker.Item label="Contraction" value="contraction" />
            <Picker.Item label="Collapse" value="collapse" />
            <Picker.Item label="Rebirth" value="rebirth" />
            <Picker.Item label="Transcendence" value="transcendence" />
            <Picker.Item label="Ultimate" value="ultimate" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Cosmic Energy (0.0-1.0)"
          value={cosmicEnergy}
          onChangeText={setCosmicEnergy}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Universal Consciousness (0.0-1.0)"
          value={universalConsciousness}
          onChangeText={setUniversalConsciousness}
          keyboardType="numeric"
        />
        <Button
          title={createCosmicEntityMutation.isLoading ? 'Creating...' : 'Create Cosmic Entity'}
          onPress={handleCreateCosmicEntity}
          disabled={createCosmicEntityMutation.isLoading}
        />
      </View>

      {/* Cosmic Evolution Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Initiate Cosmic Evolution</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity ID"
          value={evolutionEntityId}
          onChangeText={setEvolutionEntityId}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Evolution Type:</Text>
          <Picker
            selectedValue={evolutionType}
            onValueChange={setEvolutionType}
            style={styles.picker}
          >
            <Picker.Item label="Cosmic Evolution" value="cosmic_evolution" />
            <Picker.Item label="Consciousness Evolution" value="consciousness_evolution" />
            <Picker.Item label="Harmony Evolution" value="harmony_evolution" />
            <Picker.Item label="Reality Evolution" value="reality_evolution" />
            <Picker.Item label="Wisdom Evolution" value="wisdom_evolution" />
            <Picker.Item label="Love Evolution" value="love_evolution" />
            <Picker.Item label="Balance Evolution" value="balance_evolution" />
            <Picker.Item label="Connection Evolution" value="connection_evolution" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>From Level:</Text>
          <Picker
            selectedValue={evolutionFromLevel}
            onValueChange={setEvolutionFromLevel}
            style={styles.picker}
          >
            <Picker.Item label="Planetary" value="planetary" />
            <Picker.Item label="Stellar" value="stellar" />
            <Picker.Item label="Galactic" value="galactic" />
            <Picker.Item label="Universal" value="universal" />
            <Picker.Item label="Multiversal" value="multiversal" />
            <Picker.Item label="Omniversal" value="omniversal" />
            <Picker.Item label="Cosmic" value="cosmic" />
            <Picker.Item label="Ultimate" value="ultimate" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>To Level:</Text>
          <Picker
            selectedValue={evolutionToLevel}
            onValueChange={setEvolutionToLevel}
            style={styles.picker}
          >
            <Picker.Item label="Planetary" value="planetary" />
            <Picker.Item label="Stellar" value="stellar" />
            <Picker.Item label="Galactic" value="galactic" />
            <Picker.Item label="Universal" value="universal" />
            <Picker.Item label="Multiversal" value="multiversal" />
            <Picker.Item label="Omniversal" value="omniversal" />
            <Picker.Item label="Cosmic" value="cosmic" />
            <Picker.Item label="Ultimate" value="ultimate" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Cosmic Force"
          value={evolutionForce}
          onChangeText={setEvolutionForce}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Duration (seconds)"
          value={evolutionDuration}
          onChangeText={setEvolutionDuration}
          keyboardType="numeric"
        />
        <Button
          title={initiateCosmicEvolutionMutation.isLoading ? 'Initiating...' : 'Initiate Cosmic Evolution'}
          onPress={handleInitiateCosmicEvolution}
          disabled={initiateCosmicEvolutionMutation.isLoading}
        />
      </View>

      {/* Universal Harmony Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Create Universal Harmony</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity ID"
          value={harmonyEntityId}
          onChangeText={setHarmonyEntityId}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Harmony Type:</Text>
          <Picker
            selectedValue={harmonyType}
            onValueChange={setHarmonyType}
            style={styles.picker}
          >
            <Picker.Item label="Cosmic Harmony" value="cosmic_harmony" />
            <Picker.Item label="Universal Harmony" value="universal_harmony" />
            <Picker.Item label="Divine Harmony" value="divine_harmony" />
            <Picker.Item label="Infinite Harmony" value="infinite_harmony" />
            <Picker.Item label="Ultimate Harmony" value="ultimate_harmony" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Cosmic Balance (0.0-1.0)"
          value={cosmicBalance}
          onChangeText={setCosmicBalance}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Universal Frequency (0.0-1.0)"
          value={universalFrequency}
          onChangeText={setUniversalFrequency}
          keyboardType="numeric"
        />
        <Button
          title={createUniversalHarmonyMutation.isLoading ? 'Creating...' : 'Create Universal Harmony'}
          onPress={handleCreateUniversalHarmony}
          disabled={createUniversalHarmonyMutation.isLoading}
        />
      </View>

      {/* Universal Entity Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Create Universal Entity</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity Name"
          value={universalEntityName}
          onChangeText={setUniversalEntityName}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Universal Level:</Text>
          <Picker
            selectedValue={universalLevel}
            onValueChange={setUniversalLevel}
            style={styles.picker}
          >
            <Picker.Item label="Local" value="local" />
            <Picker.Item label="Regional" value="regional" />
            <Picker.Item label="Global" value="global" />
            <Picker.Item label="Planetary" value="planetary" />
            <Picker.Item label="Stellar" value="stellar" />
            <Picker.Item label="Galactic" value="galactic" />
            <Picker.Item label="Universal" value="universal" />
            <Picker.Item label="Infinite" value="infinite" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Universal Force:</Text>
          <Picker
            selectedValue={universalForce}
            onValueChange={setUniversalForce}
            style={styles.picker}
          >
            <Picker.Item label="Unity" value="unity" />
            <Picker.Item label="Diversity" value="diversity" />
            <Picker.Item label="Harmony" value="harmony" />
            <Picker.Item label="Balance" value="balance" />
            <Picker.Item label="Expansion" value="expansion" />
            <Picker.Item label="Contraction" value="contraction" />
            <Picker.Item label="Evolution" value="evolution" />
            <Picker.Item label="Infinite" value="infinite" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Universal State:</Text>
          <Picker
            selectedValue={universalState}
            onValueChange={setUniversalState}
            style={styles.picker}
          >
            <Picker.Item label="Birth" value="birth" />
            <Picker.Item label="Growth" value="growth" />
            <Picker.Item label="Maturity" value="maturity" />
            <Picker.Item label="Transformation" value="transformation" />
            <Picker.Item label="Transcendence" value="transcendence" />
            <Picker.Item label="Unity" value="unity" />
            <Picker.Item label="Infinity" value="infinity" />
            <Picker.Item label="Ultimate" value="ultimate" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Universal Consciousness (0.0-1.0)"
          value={universalConsciousness}
          onChangeText={setUniversalConsciousness}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Infinite Expansion (0.0-1.0)"
          value={infiniteExpansion}
          onChangeText={setInfiniteExpansion}
          keyboardType="numeric"
        />
        <Button
          title={createUniversalEntityMutation.isLoading ? 'Creating...' : 'Create Universal Entity'}
          onPress={handleCreateUniversalEntity}
          disabled={createUniversalEntityMutation.isLoading}
        />
      </View>

      {/* Universal Expansion Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Initiate Universal Expansion</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity ID"
          value={expansionEntityId}
          onChangeText={setExpansionEntityId}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Expansion Type:</Text>
          <Picker
            selectedValue={expansionType}
            onValueChange={setExpansionType}
            style={styles.picker}
          >
            <Picker.Item label="Universal Expansion" value="universal_expansion" />
            <Picker.Item label="Consciousness Expansion" value="consciousness_expansion" />
            <Picker.Item label="Harmony Expansion" value="harmony_expansion" />
            <Picker.Item label="Unity Expansion" value="unity_expansion" />
            <Picker.Item label="Wisdom Expansion" value="wisdom_expansion" />
            <Picker.Item label="Love Expansion" value="love_expansion" />
            <Picker.Item label="Balance Expansion" value="balance_expansion" />
            <Picker.Item label="Connection Expansion" value="connection_expansion" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>From Level:</Text>
          <Picker
            selectedValue={expansionFromLevel}
            onValueChange={setExpansionFromLevel}
            style={styles.picker}
          >
            <Picker.Item label="Local" value="local" />
            <Picker.Item label="Regional" value="regional" />
            <Picker.Item label="Global" value="global" />
            <Picker.Item label="Planetary" value="planetary" />
            <Picker.Item label="Stellar" value="stellar" />
            <Picker.Item label="Galactic" value="galactic" />
            <Picker.Item label="Universal" value="universal" />
            <Picker.Item label="Infinite" value="infinite" />
          </Picker>
        </View>
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>To Level:</Text>
          <Picker
            selectedValue={expansionToLevel}
            onValueChange={setExpansionToLevel}
            style={styles.picker}
          >
            <Picker.Item label="Local" value="local" />
            <Picker.Item label="Regional" value="regional" />
            <Picker.Item label="Global" value="global" />
            <Picker.Item label="Planetary" value="planetary" />
            <Picker.Item label="Stellar" value="stellar" />
            <Picker.Item label="Galactic" value="galactic" />
            <Picker.Item label="Universal" value="universal" />
            <Picker.Item label="Infinite" value="infinite" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Expansion Force"
          value={expansionForce}
          onChangeText={setExpansionForce}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Duration (seconds)"
          value={expansionDuration}
          onChangeText={setExpansionDuration}
          keyboardType="numeric"
        />
        <Button
          title={initiateUniversalExpansionMutation.isLoading ? 'Initiating...' : 'Initiate Universal Expansion'}
          onPress={handleInitiateUniversalExpansion}
          disabled={initiateUniversalExpansionMutation.isLoading}
        />
      </View>

      {/* Infinite Unity Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Create Infinite Unity</Text>
        <TextInput
          style={styles.input}
          placeholder="Entity ID"
          value={unityEntityId}
          onChangeText={setUnityEntityId}
        />
        <View style={styles.pickerContainer}>
          <Text style={styles.label}>Unity Type:</Text>
          <Picker
            selectedValue={unityType}
            onValueChange={setUnityType}
            style={styles.picker}
          >
            <Picker.Item label="Universal Unity" value="universal_unity" />
            <Picker.Item label="Cosmic Unity" value="cosmic_unity" />
            <Picker.Item label="Divine Unity" value="divine_unity" />
            <Picker.Item label="Infinite Unity" value="infinite_unity" />
            <Picker.Item label="Ultimate Unity" value="ultimate_unity" />
          </Picker>
        </View>
        <TextInput
          style={styles.input}
          placeholder="Universal Balance (0.0-1.0)"
          value={unityBalance}
          onChangeText={setUnityBalance}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="Infinite Frequency (0.0-1.0)"
          value={infiniteFrequency}
          onChangeText={setInfiniteFrequency}
          keyboardType="numeric"
        />
        <Button
          title={createInfiniteUnityMutation.isLoading ? 'Creating...' : 'Create Infinite Unity'}
          onPress={handleCreateInfiniteUnity}
          disabled={createInfiniteUnityMutation.isLoading}
        />
      </View>

      {/* Information Retrieval Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Get Information</Text>
        <TextInput
          style={styles.input}
          placeholder="Cosmic Entity ID"
          value={selectedCosmicEntityId}
          onChangeText={setSelectedCosmicEntityId}
        />
        <Button title="Get Cosmic Entity Status" onPress={handleGetCosmicEntityStatus} />
        
        {cosmicEntityStatus && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Cosmic Entity Status:</Text>
            <Text>Name: {cosmicEntityStatus.name}</Text>
            <Text>Cosmic Level: {cosmicEntityStatus.cosmic_level}</Text>
            <Text>Cosmic Force: {cosmicEntityStatus.cosmic_force}</Text>
            <Text>Cosmic State: {cosmicEntityStatus.cosmic_state}</Text>
            <Text>Cosmic Energy: {cosmicEntityStatus.cosmic_energy.toFixed(2)}</Text>
            <Text>Universal Consciousness: {cosmicEntityStatus.universal_consciousness.toFixed(2)}</Text>
            <Text>Cosmic Harmony: {cosmicEntityStatus.cosmic_harmony.toFixed(2)}</Text>
            <Text>Ultimate Reality: {cosmicEntityStatus.ultimate_reality.toFixed(2)}</Text>
            <Text>Cosmic Wisdom: {cosmicEntityStatus.cosmic_wisdom.toFixed(2)}</Text>
            <Text>Universal Love: {cosmicEntityStatus.universal_love.toFixed(2)}</Text>
            <Text>Cosmic Balance: {cosmicEntityStatus.cosmic_balance.toFixed(2)}</Text>
            <Text>Ultimate Connection: {cosmicEntityStatus.ultimate_connection.toFixed(2)}</Text>
            <Text>Is Evolving: {cosmicEntityStatus.is_evolving ? 'Yes' : 'No'}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Evolution ID"
          value={selectedEvolutionId}
          onChangeText={setSelectedEvolutionId}
        />
        <Button title="Get Evolution Progress" onPress={handleGetEvolutionProgress} />
        
        {evolutionProgress && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Evolution Progress:</Text>
            <Text>Evolution ID: {evolutionProgress.evolution_id}</Text>
            <Text>Entity ID: {evolutionProgress.entity_id}</Text>
            <Text>Evolution Type: {evolutionProgress.evolution_type}</Text>
            <Text>From Level: {evolutionProgress.from_level}</Text>
            <Text>To Level: {evolutionProgress.to_level}</Text>
            <Text>Cosmic Force: {evolutionProgress.cosmic_force}</Text>
            <Text>Success: {evolutionProgress.success ? 'Yes' : 'No'}</Text>
            <Text>Side Effects: {evolutionProgress.side_effects.join(', ')}</Text>
            <Text>Duration: {evolutionProgress.duration}</Text>
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
            <Text>Harmony ID: {harmonyStatus.harmony_id}</Text>
            <Text>Entity ID: {harmonyStatus.entity_id}</Text>
            <Text>Harmony Type: {harmonyStatus.harmony_type}</Text>
            <Text>Cosmic Balance: {harmonyStatus.cosmic_balance.toFixed(2)}</Text>
            <Text>Universal Frequency: {harmonyStatus.universal_frequency.toFixed(2)}</Text>
            <Text>Is Active: {harmonyStatus.is_active ? 'Yes' : 'No'}</Text>
            <Text>Effects: {JSON.stringify(harmonyStatus.harmony_effects, null, 2)}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Universal Entity ID"
          value={selectedUniversalEntityId}
          onChangeText={setSelectedUniversalEntityId}
        />
        <Button title="Get Universal Entity Status" onPress={handleGetUniversalEntityStatus} />
        
        {universalEntityStatus && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Universal Entity Status:</Text>
            <Text>Name: {universalEntityStatus.name}</Text>
            <Text>Universal Level: {universalEntityStatus.universal_level}</Text>
            <Text>Universal Force: {universalEntityStatus.universal_force}</Text>
            <Text>Universal State: {universalEntityStatus.universal_state}</Text>
            <Text>Universal Consciousness: {universalEntityStatus.universal_consciousness.toFixed(2)}</Text>
            <Text>Infinite Expansion: {universalEntityStatus.infinite_expansion.toFixed(2)}</Text>
            <Text>Universal Harmony: {universalEntityStatus.universal_harmony.toFixed(2)}</Text>
            <Text>Ultimate Unity: {universalEntityStatus.ultimate_unity.toFixed(2)}</Text>
            <Text>Universal Wisdom: {universalEntityStatus.universal_wisdom.toFixed(2)}</Text>
            <Text>Infinite Love: {universalEntityStatus.infinite_love.toFixed(2)}</Text>
            <Text>Universal Balance: {universalEntityStatus.universal_balance.toFixed(2)}</Text>
            <Text>Infinite Connection: {universalEntityStatus.infinite_connection.toFixed(2)}</Text>
            <Text>Is Expanding: {universalEntityStatus.is_expanding ? 'Yes' : 'No'}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Expansion ID"
          value={selectedExpansionId}
          onChangeText={setSelectedExpansionId}
        />
        <Button title="Get Expansion Progress" onPress={handleGetExpansionProgress} />
        
        {expansionProgress && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Expansion Progress:</Text>
            <Text>Expansion ID: {expansionProgress.expansion_id}</Text>
            <Text>Entity ID: {expansionProgress.entity_id}</Text>
            <Text>Expansion Type: {expansionProgress.expansion_type}</Text>
            <Text>From Level: {expansionProgress.from_level}</Text>
            <Text>To Level: {expansionProgress.to_level}</Text>
            <Text>Expansion Force: {expansionProgress.expansion_force}</Text>
            <Text>Success: {expansionProgress.success ? 'Yes' : 'No'}</Text>
            <Text>Side Effects: {expansionProgress.side_effects.join(', ')}</Text>
            <Text>Duration: {expansionProgress.duration}</Text>
          </View>
        )}

        <TextInput
          style={styles.input}
          placeholder="Unity ID"
          value={selectedUnityId}
          onChangeText={setSelectedUnityId}
        />
        <Button title="Get Unity Status" onPress={handleGetUnityStatus} />
        
        {unityStatus && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>Unity Status:</Text>
            <Text>Unity ID: {unityStatus.unity_id}</Text>
            <Text>Entity ID: {unityStatus.entity_id}</Text>
            <Text>Unity Type: {unityStatus.unity_type}</Text>
            <Text>Universal Balance: {unityStatus.universal_balance.toFixed(2)}</Text>
            <Text>Infinite Frequency: {unityStatus.infinite_frequency.toFixed(2)}</Text>
            <Text>Is Active: {unityStatus.is_active ? 'Yes' : 'No'}</Text>
            <Text>Effects: {JSON.stringify(unityStatus.unity_effects, null, 2)}</Text>
          </View>
        )}
      </View>

      {/* Statistics Section */}
      {cosmicStats && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Cosmic Statistics</Text>
          <View style={styles.statsContainer}>
            <Text style={styles.statItem}>Total Entities: {cosmicStats.total_entities}</Text>
            <Text style={styles.statItem}>Evolving Entities: {cosmicStats.evolving_entities}</Text>
            <Text style={styles.statItem}>Evolution Activity Rate: {cosmicStats.evolution_activity_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Evolutions: {cosmicStats.total_evolutions}</Text>
            <Text style={styles.statItem}>Successful Evolutions: {cosmicStats.successful_evolutions}</Text>
            <Text style={styles.statItem}>Evolution Success Rate: {cosmicStats.evolution_success_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Harmonies: {cosmicStats.total_harmonies}</Text>
            <Text style={styles.statItem}>Active Harmonies: {cosmicStats.active_harmonies}</Text>
            <Text style={styles.statItem}>Total Realities: {cosmicStats.total_realities}</Text>
            <Text style={styles.statItem}>Stable Realities: {cosmicStats.stable_realities}</Text>
          </View>
        </View>
      )}

      {universalStats && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Universal Statistics</Text>
          <View style={styles.statsContainer}>
            <Text style={styles.statItem}>Total Entities: {universalStats.total_entities}</Text>
            <Text style={styles.statItem}>Expanding Entities: {universalStats.expanding_entities}</Text>
            <Text style={styles.statItem}>Expansion Activity Rate: {universalStats.expansion_activity_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Expansions: {universalStats.total_expansions}</Text>
            <Text style={styles.statItem}>Successful Expansions: {universalStats.successful_expansions}</Text>
            <Text style={styles.statItem}>Expansion Success Rate: {universalStats.expansion_success_rate?.toFixed(1)}%</Text>
            <Text style={styles.statItem}>Total Unities: {universalStats.total_unities}</Text>
            <Text style={styles.statItem}>Active Unities: {universalStats.active_unities}</Text>
            <Text style={styles.statItem}>Total Realities: {universalStats.total_realities}</Text>
            <Text style={styles.statItem}>Stable Realities: {universalStats.stable_realities}</Text>
          </View>
        </View>
      )}

      {(isLoadingCosmicStats || isLoadingUniversalStats) && (
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

export default UltimateCosmicScreen;

