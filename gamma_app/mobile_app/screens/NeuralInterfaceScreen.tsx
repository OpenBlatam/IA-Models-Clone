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
interface NeuralDevice {
  device_id: string;
  name: string;
  interface_type: string;
  signal_types: string[];
  num_channels: number;
  sampling_rate: number;
  resolution: number;
  is_connected: boolean;
  battery_level?: number;
  last_calibration?: string;
  created_at: string;
}

interface NeuralSession {
  session_id: string;
  device_id: string;
  user_id: string;
  session_type: string;
  start_time: string;
  end_time?: string;
  is_active: boolean;
  signal_quality: number;
  artifacts_detected: string[];
  device_info: {
    name: string;
    interface_type: string;
    num_channels: number;
  };
}

interface NeuralCommand {
  command_id: string;
  session_id: string;
  command_type: string;
  intent: string;
  confidence: number;
  parameters: Record<string, any>;
  timestamp: string;
  executed: boolean;
  execution_time?: number;
}

interface NeuralPattern {
  pattern_id: string;
  user_id: string;
  pattern_type: string;
  features: Record<string, any>;
  classification: string;
  confidence: number;
  training_data: any[];
  model_accuracy: number;
  created_at: string;
}

// API calls
const registerNeuralDevice = async (deviceInfo: Record<string, any>): Promise<{device_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/quantum-neural/neural/devices/register', deviceInfo);
  return response.data;
};

const startNeuralSession = async (sessionInfo: Record<string, any>): Promise<{session_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/quantum-neural/neural/sessions/start', sessionInfo);
  return response.data;
};

const processNeuralSignal = async (signalData: Record<string, any>): Promise<{signal_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/quantum-neural/neural/signals/process', signalData);
  return response.data;
};

const generateNeuralCommand = async (commandInfo: Record<string, any>): Promise<{command_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/quantum-neural/neural/commands/generate', commandInfo);
  return response.data;
};

const trainNeuralPattern = async (patternInfo: Record<string, any>): Promise<{pattern_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/quantum-neural/neural/patterns/train', patternInfo);
  return response.data;
};

const recognizeNeuralPattern = async (patternData: Record<string, any>): Promise<{result: any}> => {
  const response = await axios.post('http://localhost:8000/api/v1/quantum-neural/neural/patterns/recognize', patternData);
  return response.data;
};

const getNeuralSessionStatus = async (sessionId: string): Promise<NeuralSession> => {
  const response = await axios.get(`http://localhost:8000/api/v1/quantum-neural/neural/sessions/${sessionId}/status`);
  return response.data;
};

const getNeuralStatistics = async (): Promise<Record<string, any>> => {
  const response = await axios.get('http://localhost:8000/api/v1/quantum-neural/neural/statistics');
  return response.data;
};

const NeuralInterfaceScreen: React.FC = () => {
  const queryClient = useQueryClient();
  
  // State for Device Registration
  const [deviceName, setDeviceName] = useState<string>('');
  const [interfaceType, setInterfaceType] = useState<string>('non_invasive');
  const [signalTypes, setSignalTypes] = useState<string[]>(['eeg']);
  const [numChannels, setNumChannels] = useState<string>('64');
  const [samplingRate, setSamplingRate] = useState<string>('1000');
  const [resolution, setResolution] = useState<string>('16');
  
  // State for Session Management
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [sessionType, setSessionType] = useState<string>('general');
  const [userId, setUserId] = useState<string>('neural_user_123');
  
  // State for Signal Processing
  const [signalData, setSignalData] = useState<string>('');
  const [signalType, setSignalType] = useState<string>('eeg');
  const [channel, setChannel] = useState<string>('0');
  
  // State for Command Generation
  const [commandType, setCommandType] = useState<string>('movement');
  const [commandIntent, setCommandIntent] = useState<string>('');
  const [commandConfidence, setCommandConfidence] = useState<string>('0.8');
  
  // State for Pattern Recognition
  const [patternType, setPatternType] = useState<string>('movement');
  const [patternFeatures, setPatternFeatures] = useState<string>('');
  const [patternClassification, setPatternClassification] = useState<string>('');
  
  // State for tracking
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [activePatternId, setActivePatternId] = useState<string | null>(null);
  
  // State for modals
  const [showDeviceModal, setShowDeviceModal] = useState<boolean>(false);
  const [showSessionModal, setShowSessionModal] = useState<boolean>(false);
  const [showSignalModal, setShowSignalModal] = useState<boolean>(false);
  const [showCommandModal, setShowCommandModal] = useState<boolean>(false);
  const [showPatternModal, setShowPatternModal] = useState<boolean>(false);
  
  // Options
  const interfaceTypes = [
    { value: 'non_invasive', label: 'No Invasivo' },
    { value: 'invasive', label: 'Invasivo' },
    { value: 'partially_invasive', label: 'Parcialmente Invasivo' },
    { value: 'optical', label: 'Óptico' },
    { value: 'electrical', label: 'Eléctrico' },
    { value: 'magnetic', label: 'Magnético' },
    { value: 'ultrasound', label: 'Ultrasonido' }
  ];
  
  const signalTypeOptions = [
    { value: 'eeg', label: 'EEG (Electroencefalografía)' },
    { value: 'ecog', label: 'ECoG (Electrocorticografía)' },
    { value: 'lfp', label: 'LFP (Local Field Potential)' },
    { value: 'spike', label: 'Spike' },
    { value: 'fmri', label: 'fMRI (Functional MRI)' },
    { value: 'nirs', label: 'NIRS (Near-Infrared Spectroscopy)' },
    { value: 'meg', label: 'MEG (Magnetoencefalografía)' },
    { value: 'optical', label: 'Óptico' }
  ];
  
  const sessionTypes = [
    { value: 'general', label: 'General' },
    { value: 'movement', label: 'Control de Movimiento' },
    { value: 'speech', label: 'Control de Voz' },
    { value: 'vision', label: 'Control Visual' },
    { value: 'cognition', label: 'Cognición' },
    { value: 'emotion', label: 'Emoción' },
    { value: 'memory', label: 'Memoria' },
    { value: 'attention', label: 'Atención' }
  ];
  
  const commandTypes = [
    { value: 'movement', label: 'Movimiento' },
    { value: 'speech', label: 'Voz' },
    { value: 'vision', label: 'Visión' },
    { value: 'cognition', label: 'Cognición' },
    { value: 'emotion', label: 'Emoción' },
    { value: 'memory', label: 'Memoria' },
    { value: 'attention', label: 'Atención' },
    { value: 'creativity', label: 'Creatividad' }
  ];
  
  const patternTypes = [
    { value: 'movement', label: 'Movimiento' },
    { value: 'speech', label: 'Voz' },
    { value: 'emotion', label: 'Emoción' },
    { value: 'attention', label: 'Atención' },
    { value: 'memory', label: 'Memoria' },
    { value: 'cognition', label: 'Cognición' },
    { value: 'creativity', label: 'Creatividad' },
    { value: 'sleep', label: 'Sueño' }
  ];
  
  // Queries
  const { data: statistics, isLoading: isLoadingStats } = useQuery<Record<string, any>, Error>(
    'neuralStatistics',
    getNeuralStatistics,
    {
      refetchInterval: 30000,
    }
  );
  
  const { data: sessionStatus, isLoading: isLoadingSession } = useQuery<NeuralSession, Error>(
    ['neuralSessionStatus', activeSessionId],
    () => activeSessionId ? getNeuralSessionStatus(activeSessionId) : Promise.resolve(null),
    {
      enabled: !!activeSessionId,
      refetchInterval: 2000,
    }
  );
  
  // Mutations
  const registerDeviceMutation = useMutation<{device_id: string}, Error, Record<string, any>>(registerNeuralDevice, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Dispositivo neural registrado: ${data.device_id}`);
      setShowDeviceModal(false);
      queryClient.invalidateQueries('neuralStatistics');
    },
    onError: (error) => {
      Alert.alert('Error', `Error al registrar dispositivo: ${error.message}`);
    },
  });
  
  const startSessionMutation = useMutation<{session_id: string}, Error, Record<string, any>>(startNeuralSession, {
    onSuccess: (data) => {
      setActiveSessionId(data.session_id);
      Alert.alert('Éxito', `Sesión neural iniciada: ${data.session_id}`);
      setShowSessionModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al iniciar sesión: ${error.message}`);
    },
  });
  
  const processSignalMutation = useMutation<{signal_id: string}, Error, Record<string, any>>(processNeuralSignal, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Señal neural procesada: ${data.signal_id}`);
      setShowSignalModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al procesar señal: ${error.message}`);
    },
  });
  
  const generateCommandMutation = useMutation<{command_id: string}, Error, Record<string, any>>(generateNeuralCommand, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Comando neural generado: ${data.command_id}`);
      setShowCommandModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al generar comando: ${error.message}`);
    },
  });
  
  const trainPatternMutation = useMutation<{pattern_id: string}, Error, Record<string, any>>(trainNeuralPattern, {
    onSuccess: (data) => {
      setActivePatternId(data.pattern_id);
      Alert.alert('Éxito', `Patrón neural entrenado: ${data.pattern_id}`);
      setShowPatternModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al entrenar patrón: ${error.message}`);
    },
  });
  
  // Handlers
  const handleRegisterDevice = () => {
    if (!deviceName.trim()) {
      Alert.alert('Error', 'Por favor ingresa un nombre para el dispositivo.');
      return;
    }
    
    registerDeviceMutation.mutate({
      name: deviceName,
      interface_type: interfaceType,
      signal_types: signalTypes,
      num_channels: parseInt(numChannels, 10),
      sampling_rate: parseFloat(samplingRate),
      resolution: parseFloat(resolution)
    });
  };
  
  const handleStartSession = () => {
    if (!selectedDevice.trim()) {
      Alert.alert('Error', 'Por favor selecciona un dispositivo.');
      return;
    }
    
    startSessionMutation.mutate({
      device_id: selectedDevice,
      user_id: userId,
      session_type: sessionType
    });
  };
  
  const handleProcessSignal = () => {
    if (!signalData.trim() || !activeSessionId) {
      Alert.alert('Error', 'Por favor ingresa datos de señal y asegúrate de tener una sesión activa.');
      return;
    }
    
    try {
      const data = JSON.parse(signalData);
      processSignalMutation.mutate({
        session_id: activeSessionId,
        signal_type: signalType,
        channel: parseInt(channel, 10),
        data: data
      });
    } catch (error) {
      Alert.alert('Error', 'Los datos de señal deben ser un JSON válido.');
    }
  };
  
  const handleGenerateCommand = () => {
    if (!commandIntent.trim() || !activeSessionId) {
      Alert.alert('Error', 'Por favor ingresa una intención y asegúrate de tener una sesión activa.');
      return;
    }
    
    generateCommandMutation.mutate({
      session_id: activeSessionId,
      command_type: commandType,
      intent: commandIntent,
      confidence: parseFloat(commandConfidence),
      parameters: {}
    });
  };
  
  const handleTrainPattern = () => {
    if (!patternFeatures.trim() || !patternClassification.trim()) {
      Alert.alert('Error', 'Por favor completa todos los campos del patrón.');
      return;
    }
    
    try {
      const features = JSON.parse(patternFeatures);
      trainPatternMutation.mutate({
        user_id: userId,
        pattern_type: patternType,
        features: features,
        classification: patternClassification,
        training_data: []
      });
    } catch (error) {
      Alert.alert('Error', 'Las características deben ser un JSON válido.');
    }
  };
  
  const handleRecognizePattern = async () => {
    if (!patternFeatures.trim()) {
      Alert.alert('Error', 'Por favor ingresa las características del patrón.');
      return;
    }
    
    try {
      const features = JSON.parse(patternFeatures);
      const result = await recognizeNeuralPattern({
        user_id: userId,
        features: features
      });
      
      Alert.alert(
        'Reconocimiento de Patrón',
        `Clasificación: ${result.result.classification}\nConfianza: ${(result.result.confidence * 100).toFixed(1)}%`
      );
    } catch (error) {
      Alert.alert('Error', `Error al reconocer patrón: ${error.message}`);
    }
  };
  
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Neural Interface</Text>
      <Text style={styles.subtitle}>Interfaz Cerebro-Computadora Avanzada</Text>
      
      {/* Statistics */}
      {statistics && (
        <View style={styles.statisticsContainer}>
          <Text style={styles.sectionTitle}>Estadísticas del Sistema</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_neural_devices || 0}</Text>
              <Text style={styles.statLabel}>Dispositivos</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_neural_sessions || 0}</Text>
              <Text style={styles.statLabel}>Sesiones</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_neural_signals || 0}</Text>
              <Text style={styles.statLabel}>Señales</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_neural_commands || 0}</Text>
              <Text style={styles.statLabel}>Comandos</Text>
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
            onPress={() => setShowDeviceModal(true)}
          >
            <Text style={styles.actionButtonText}>Registrar Dispositivo</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#2196F3' }]}
            onPress={() => setShowSessionModal(true)}
          >
            <Text style={styles.actionButtonText}>Iniciar Sesión</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#FF9800' }]}
            onPress={() => setShowSignalModal(true)}
          >
            <Text style={styles.actionButtonText}>Procesar Señal</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#9C27B0' }]}
            onPress={() => setShowCommandModal(true)}
          >
            <Text style={styles.actionButtonText}>Generar Comando</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#E91E63' }]}
            onPress={() => setShowPatternModal(true)}
          >
            <Text style={styles.actionButtonText}>Entrenar Patrón</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#00BCD4' }]}
            onPress={handleRecognizePattern}
          >
            <Text style={styles.actionButtonText}>Reconocer Patrón</Text>
          </TouchableOpacity>
        </View>
      </View>
      
      {/* Session Status */}
      {sessionStatus && (
        <View style={styles.statusContainer}>
          <Text style={styles.sectionTitle}>Estado de la Sesión Neural</Text>
          <View style={styles.statusCard}>
            <Text style={styles.statusTitle}>Sesión: {sessionStatus.session_id}</Text>
            <Text style={styles.statusInfo}>Usuario: {sessionStatus.user_id}</Text>
            <Text style={styles.statusInfo}>Tipo: {sessionStatus.session_type}</Text>
            <Text style={styles.statusInfo}>Dispositivo: {sessionStatus.device_info.name}</Text>
            <Text style={styles.statusInfo}>Canales: {sessionStatus.device_info.num_channels}</Text>
            <Text style={[styles.statusInfo, { color: sessionStatus.is_active ? '#4CAF50' : '#FF9800' }]}>
              Estado: {sessionStatus.is_active ? 'Activa' : 'Inactiva'}
            </Text>
            <Text style={styles.statusInfo}>
              Calidad de Señal: {(sessionStatus.signal_quality * 100).toFixed(1)}%
            </Text>
            {sessionStatus.artifacts_detected.length > 0 && (
              <Text style={styles.statusInfo}>
                Artefactos: {sessionStatus.artifacts_detected.join(', ')}
              </Text>
            )}
          </View>
        </View>
      )}
      
      {/* Device Registration Modal */}
      <Modal
        visible={showDeviceModal}
        animationType="slide"
        onRequestClose={() => setShowDeviceModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Registrar Dispositivo Neural</Text>
            <Button title="Cerrar" onPress={() => setShowDeviceModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Nombre del Dispositivo:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: EEG Headset Pro"
              value={deviceName}
              onChangeText={setDeviceName}
            />
            
            <Text style={styles.label}>Tipo de Interfaz:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={interfaceType}
                onValueChange={setInterfaceType}
                style={styles.picker}
              >
                {interfaceTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Tipos de Señal:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={signalTypes[0]}
                onValueChange={(value) => setSignalTypes([value])}
                style={styles.picker}
              >
                {signalTypeOptions.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Número de Canales:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={numChannels}
                onValueChange={setNumChannels}
                style={styles.picker}
              >
                <Picker.Item label="8" value="8" />
                <Picker.Item label="16" value="16" />
                <Picker.Item label="32" value="32" />
                <Picker.Item label="64" value="64" />
                <Picker.Item label="128" value="128" />
                <Picker.Item label="256" value="256" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Frecuencia de Muestreo (Hz):</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={samplingRate}
                onValueChange={setSamplingRate}
                style={styles.picker}
              >
                <Picker.Item label="250 Hz" value="250" />
                <Picker.Item label="500 Hz" value="500" />
                <Picker.Item label="1000 Hz" value="1000" />
                <Picker.Item label="2000 Hz" value="2000" />
                <Picker.Item label="5000 Hz" value="5000" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Resolución (bits):</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={resolution}
                onValueChange={setResolution}
                style={styles.picker}
              >
                <Picker.Item label="8 bits" value="8" />
                <Picker.Item label="16 bits" value="16" />
                <Picker.Item label="24 bits" value="24" />
                <Picker.Item label="32 bits" value="32" />
              </Picker>
            </View>
            
            <Button
              title={registerDeviceMutation.isLoading ? 'Registrando...' : 'Registrar Dispositivo'}
              onPress={handleRegisterDevice}
              disabled={registerDeviceMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Session Start Modal */}
      <Modal
        visible={showSessionModal}
        animationType="slide"
        onRequestClose={() => setShowSessionModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Iniciar Sesión Neural</Text>
            <Button title="Cerrar" onPress={() => setShowSessionModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>ID del Dispositivo:</Text>
            <TextInput
              style={styles.input}
              placeholder="device_id_here"
              value={selectedDevice}
              onChangeText={setSelectedDevice}
            />
            
            <Text style={styles.label}>ID del Usuario:</Text>
            <TextInput
              style={styles.input}
              placeholder="user_id_here"
              value={userId}
              onChangeText={setUserId}
            />
            
            <Text style={styles.label}>Tipo de Sesión:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={sessionType}
                onValueChange={setSessionType}
                style={styles.picker}
              >
                {sessionTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Button
              title={startSessionMutation.isLoading ? 'Iniciando...' : 'Iniciar Sesión'}
              onPress={handleStartSession}
              disabled={startSessionMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Signal Processing Modal */}
      <Modal
        visible={showSignalModal}
        animationType="slide"
        onRequestClose={() => setShowSignalModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Procesar Señal Neural</Text>
            <Button title="Cerrar" onPress={() => setShowSignalModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Tipo de Señal:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={signalType}
                onValueChange={setSignalType}
                style={styles.picker}
              >
                {signalTypeOptions.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Canal:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={channel}
                onValueChange={setChannel}
                style={styles.picker}
              >
                {Array.from({ length: 64 }, (_, i) => (
                  <Picker.Item key={i} label={`Canal ${i}`} value={i.toString()} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Datos de Señal (JSON):</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder='[0.1, 0.2, 0.3, 0.4, 0.5, ...]'
              value={signalData}
              onChangeText={setSignalData}
              multiline
              numberOfLines={4}
            />
            
            <Button
              title={processSignalMutation.isLoading ? 'Procesando...' : 'Procesar Señal'}
              onPress={handleProcessSignal}
              disabled={processSignalMutation.isLoading || !activeSessionId}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Command Generation Modal */}
      <Modal
        visible={showCommandModal}
        animationType="slide"
        onRequestClose={() => setShowCommandModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Generar Comando Neural</Text>
            <Button title="Cerrar" onPress={() => setShowCommandModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Tipo de Comando:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={commandType}
                onValueChange={setCommandType}
                style={styles.picker}
              >
                {commandTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Intención:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: mover brazo derecho hacia adelante"
              value={commandIntent}
              onChangeText={setCommandIntent}
            />
            
            <Text style={styles.label}>Confianza (0.0 - 1.0):</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={commandConfidence}
                onValueChange={setCommandConfidence}
                style={styles.picker}
              >
                <Picker.Item label="0.5" value="0.5" />
                <Picker.Item label="0.6" value="0.6" />
                <Picker.Item label="0.7" value="0.7" />
                <Picker.Item label="0.8" value="0.8" />
                <Picker.Item label="0.9" value="0.9" />
                <Picker.Item label="1.0" value="1.0" />
              </Picker>
            </View>
            
            <Button
              title={generateCommandMutation.isLoading ? 'Generando...' : 'Generar Comando'}
              onPress={handleGenerateCommand}
              disabled={generateCommandMutation.isLoading || !activeSessionId}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* Pattern Training Modal */}
      <Modal
        visible={showPatternModal}
        animationType="slide"
        onRequestClose={() => setShowPatternModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Entrenar Patrón Neural</Text>
            <Button title="Cerrar" onPress={() => setShowPatternModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Tipo de Patrón:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={patternType}
                onValueChange={setPatternType}
                style={styles.picker}
              >
                {patternTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Clasificación:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: movimiento_adelante"
              value={patternClassification}
              onChangeText={setPatternClassification}
            />
            
            <Text style={styles.label}>Características (JSON):</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder='{"alpha_power": 0.8, "beta_power": 0.6, "gamma_power": 0.4}'
              value={patternFeatures}
              onChangeText={setPatternFeatures}
              multiline
              numberOfLines={4}
            />
            
            <Button
              title={trainPatternMutation.isLoading ? 'Entrenando...' : 'Entrenar Patrón'}
              onPress={handleTrainPattern}
              disabled={trainPatternMutation.isLoading}
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

export default NeuralInterfaceScreen;


