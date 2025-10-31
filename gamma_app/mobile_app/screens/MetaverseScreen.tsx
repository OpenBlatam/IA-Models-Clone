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
interface VRDevice {
  device_id: string;
  device_type: string;
  name: string;
  resolution: [number, number];
  refresh_rate: number;
  field_of_view: number;
  tracking_type: string;
  controllers: string[];
  is_connected: boolean;
  battery_level?: number;
  last_seen: string;
}

interface ARSession {
  session_id: string;
  platform: string;
  device_info: Record<string, any>;
  tracking_state: string;
  lighting_estimation: Record<string, any>;
  plane_detection: Record<string, any>[];
  anchors: string[];
  is_active: boolean;
  created_at: string;
}

interface VR3DContent {
  content_id: string;
  content_type: string;
  name: string;
  description: string;
  file_path: string;
  file_format: string;
  file_size: number;
  dimensions: [number, number, number];
  vertices_count: number;
  textures: string[];
  materials: string[];
  animations: string[];
  created_at: string;
}

interface VRScene {
  scene_id: string;
  name: string;
  description: string;
  content_objects: string[];
  lighting: Record<string, any>;
  physics: Record<string, any>;
  audio: Record<string, any>;
  interactions: string[];
  environment_settings: Record<string, any>;
  active_users: number;
  users: any[];
  is_published: boolean;
  created_at: string;
}

// API calls
const registerVRDevice = async (deviceInfo: Record<string, any>): Promise<{device_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/metaverse-consciousness/vr/devices/register', deviceInfo);
  return response.data;
};

const startARSession = async (platform: string, deviceInfo: Record<string, any>): Promise<{session_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/metaverse-consciousness/ar/sessions/start', {
    platform,
    device_info: deviceInfo
  });
  return response.data;
};

const upload3DContent = async (contentInfo: Record<string, any>): Promise<{content_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/metaverse-consciousness/content/3d/upload', contentInfo);
  return response.data;
};

const createVRScene = async (sceneInfo: Record<string, any>): Promise<{scene_id: string}> => {
  const response = await axios.post('http://localhost:8000/api/v1/metaverse-consciousness/scenes/create', sceneInfo);
  return response.data;
};

const getAvailableContent = async (): Promise<VR3DContent[]> => {
  const response = await axios.get('http://localhost:8000/api/v1/metaverse-consciousness/content/3d');
  return response.data.content || [];
};

const getVRScenes = async (): Promise<VRScene[]> => {
  // This would be a custom endpoint to list scenes
  return [];
};

const getMetaverseStatistics = async (): Promise<Record<string, any>> => {
  const response = await axios.get('http://localhost:8000/api/v1/metaverse-consciousness/metaverse/statistics');
  return response.data;
};

const MetaverseScreen: React.FC = () => {
  const queryClient = useQueryClient();
  
  // State for VR Device Registration
  const [deviceName, setDeviceName] = useState<string>('');
  const [deviceType, setDeviceType] = useState<string>('oculus_quest');
  const [resolution, setResolution] = useState<string>('1920x1080');
  const [refreshRate, setRefreshRate] = useState<string>('90');
  const [fieldOfView, setFieldOfView] = useState<string>('110');
  
  // State for AR Session
  const [arPlatform, setArPlatform] = useState<string>('arkit');
  const [arDeviceInfo, setArDeviceInfo] = useState<string>('');
  
  // State for 3D Content Upload
  const [contentName, setContentName] = useState<string>('');
  const [contentDescription, setContentDescription] = useState<string>('');
  const [contentType, setContentType] = useState<string>('3d_model');
  const [filePath, setFilePath] = useState<string>('');
  const [fileFormat, setFileFormat] = useState<string>('gltf');
  
  // State for VR Scene Creation
  const [sceneName, setSceneName] = useState<string>('');
  const [sceneDescription, setSceneDescription] = useState<string>('');
  const [selectedContent, setSelectedContent] = useState<string[]>([]);
  
  // State for modals
  const [showDeviceModal, setShowDeviceModal] = useState<boolean>(false);
  const [showARModal, setShowARModal] = useState<boolean>(false);
  const [showContentModal, setShowContentModal] = useState<boolean>(false);
  const [showSceneModal, setShowSceneModal] = useState<boolean>(false);
  
  // Device types and platforms
  const deviceTypes = [
    { value: 'oculus_quest', label: 'Oculus Quest' },
    { value: 'oculus_rift', label: 'Oculus Rift' },
    { value: 'htc_vive', label: 'HTC Vive' },
    { value: 'valve_index', label: 'Valve Index' },
    { value: 'playstation_vr', label: 'PlayStation VR' },
    { value: 'windows_mr', label: 'Windows Mixed Reality' },
    { value: 'cardboard', label: 'Google Cardboard' },
    { value: 'gear_vr', label: 'Samsung Gear VR' }
  ];
  
  const arPlatforms = [
    { value: 'arkit', label: 'ARKit (iOS)' },
    { value: 'arcore', label: 'ARCore (Android)' },
    { value: 'windows_mr', label: 'Windows Mixed Reality' },
    { value: 'hololens', label: 'Microsoft HoloLens' },
    { value: 'magic_leap', label: 'Magic Leap' },
    { value: 'webxr', label: 'WebXR' }
  ];
  
  const contentTypes = [
    { value: '3d_model', label: '3D Model' },
    { value: 'animation', label: 'Animation' },
    { value: 'environment', label: 'Environment' },
    { value: 'avatar', label: 'Avatar' },
    { value: 'interactive_object', label: 'Interactive Object' },
    { value: 'ui_element', label: 'UI Element' },
    { value: 'audio_spatial', label: 'Spatial Audio' },
    { value: 'haptic_feedback', label: 'Haptic Feedback' }
  ];
  
  const fileFormats = [
    { value: 'gltf', label: 'glTF' },
    { value: 'fbx', label: 'FBX' },
    { value: 'obj', label: 'OBJ' },
    { value: 'dae', label: 'Collada' },
    { value: 'blend', label: 'Blender' },
    { value: 'max', label: '3ds Max' }
  ];
  
  // Queries
  const { data: availableContent, isLoading: isLoadingContent } = useQuery<VR3DContent[], Error>(
    'metaverseContent',
    getAvailableContent,
    {
      refetchInterval: 30000,
    }
  );
  
  const { data: statistics, isLoading: isLoadingStats } = useQuery<Record<string, any>, Error>(
    'metaverseStatistics',
    getMetaverseStatistics,
    {
      refetchInterval: 60000,
    }
  );
  
  // Mutations
  const registerDeviceMutation = useMutation<{device_id: string}, Error, Record<string, any>>(registerVRDevice, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Dispositivo VR registrado: ${data.device_id}`);
      setShowDeviceModal(false);
      queryClient.invalidateQueries('metaverseStatistics');
    },
    onError: (error) => {
      Alert.alert('Error', `Error al registrar dispositivo: ${error.message}`);
    },
  });
  
  const startARMutation = useMutation<{session_id: string}, Error, {platform: string, device_info: Record<string, any>}>(startARSession, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Sesión AR iniciada: ${data.session_id}`);
      setShowARModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al iniciar sesión AR: ${error.message}`);
    },
  });
  
  const uploadContentMutation = useMutation<{content_id: string}, Error, Record<string, any>>(upload3DContent, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Contenido 3D subido: ${data.content_id}`);
      setShowContentModal(false);
      queryClient.invalidateQueries('metaverseContent');
    },
    onError: (error) => {
      Alert.alert('Error', `Error al subir contenido: ${error.message}`);
    },
  });
  
  const createSceneMutation = useMutation<{scene_id: string}, Error, Record<string, any>>(createVRScene, {
    onSuccess: (data) => {
      Alert.alert('Éxito', `Escena VR creada: ${data.scene_id}`);
      setShowSceneModal(false);
    },
    onError: (error) => {
      Alert.alert('Error', `Error al crear escena: ${error.message}`);
    },
  });
  
  // Handlers
  const handleRegisterDevice = () => {
    if (!deviceName.trim()) {
      Alert.alert('Error', 'Por favor ingresa un nombre para el dispositivo.');
      return;
    }
    
    const [width, height] = resolution.split('x').map(Number);
    
    registerDeviceMutation.mutate({
      name: deviceName,
      device_type: deviceType,
      resolution: [width, height],
      refresh_rate: parseInt(refreshRate, 10),
      field_of_view: parseFloat(fieldOfView),
      tracking_type: 'inside_out',
      controllers: ['left_controller', 'right_controller']
    });
  };
  
  const handleStartARSession = () => {
    if (!arDeviceInfo.trim()) {
      Alert.alert('Error', 'Por favor ingresa información del dispositivo AR.');
      return;
    }
    
    startARMutation.mutate({
      platform: arPlatform,
      device_info: {
        device_name: arDeviceInfo,
        capabilities: ['plane_detection', 'light_estimation', 'anchor_tracking']
      }
    });
  };
  
  const handleUploadContent = () => {
    if (!contentName.trim() || !filePath.trim()) {
      Alert.alert('Error', 'Por favor completa todos los campos requeridos.');
      return;
    }
    
    uploadContentMutation.mutate({
      name: contentName,
      description: contentDescription,
      content_type: contentType,
      file_path: filePath,
      file_format: fileFormat,
      file_size: 1024 * 1024, // 1MB placeholder
      dimensions: [1.0, 1.0, 1.0],
      vertices_count: 1000,
      textures: [],
      materials: [],
      animations: []
    });
  };
  
  const handleCreateScene = () => {
    if (!sceneName.trim()) {
      Alert.alert('Error', 'Por favor ingresa un nombre para la escena.');
      return;
    }
    
    createSceneMutation.mutate({
      name: sceneName,
      description: sceneDescription,
      content_objects: selectedContent,
      lighting: {
        ambient_light: { color: [1.0, 1.0, 1.0], intensity: 0.3 },
        directional_light: { color: [1.0, 1.0, 1.0], intensity: 1.0, direction: [0, -1, 0] }
      },
      physics: {
        gravity: [0, -9.81, 0],
        collision_detection: true,
        physics_engine: 'bullet'
      },
      audio: {
        spatial_audio: true,
        reverb: 'medium',
        ambient_sound: null
      },
      interactions: [],
      environment_settings: {
        skybox: 'default',
        fog: false,
        weather: 'clear'
      }
    });
  };
  
  const renderContentItem = ({ item }: { item: VR3DContent }) => (
    <TouchableOpacity style={styles.contentCard}>
      <Text style={styles.contentTitle}>{item.name}</Text>
      <Text style={styles.contentInfo}>
        {item.content_type} • {item.file_format} • {item.file_size} bytes
      </Text>
      <Text style={styles.contentDescription}>{item.description}</Text>
      <Text style={styles.contentDate}>
        {format(new Date(item.created_at), 'dd/MM/yyyy HH:mm')}
      </Text>
    </TouchableOpacity>
  );
  
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Metaverse AR/VR</Text>
      <Text style={styles.subtitle}>Experiencias Inmersivas y Realidad Aumentada</Text>
      
      {/* Statistics */}
      {statistics && (
        <View style={styles.statisticsContainer}>
          <Text style={styles.sectionTitle}>Estadísticas del Sistema</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_vr_devices || 0}</Text>
              <Text style={styles.statLabel}>Dispositivos VR</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_3d_content || 0}</Text>
              <Text style={styles.statLabel}>Contenido 3D</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_vr_scenes || 0}</Text>
              <Text style={styles.statLabel}>Escenas VR</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statNumber}>{statistics.total_vr_users || 0}</Text>
              <Text style={styles.statLabel}>Usuarios VR</Text>
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
            <Text style={styles.actionButtonText}>Registrar VR</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#2196F3' }]}
            onPress={() => setShowARModal(true)}
          >
            <Text style={styles.actionButtonText}>Iniciar AR</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#FF9800' }]}
            onPress={() => setShowContentModal(true)}
          >
            <Text style={styles.actionButtonText}>Subir 3D</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.actionButton, { backgroundColor: '#9C27B0' }]}
            onPress={() => setShowSceneModal(true)}
          >
            <Text style={styles.actionButtonText}>Crear Escena</Text>
          </TouchableOpacity>
        </View>
      </View>
      
      {/* Available Content */}
      <View style={styles.contentSection}>
        <Text style={styles.sectionTitle}>Contenido 3D Disponible</Text>
        {isLoadingContent && <ActivityIndicator size="large" color="#0000ff" style={styles.activityIndicator} />}
        {availableContent && availableContent.length > 0 ? (
          <FlatList
            data={availableContent}
            renderItem={renderContentItem}
            keyExtractor={(item) => item.content_id}
            style={styles.contentList}
            scrollEnabled={false}
          />
        ) : (
          <Text style={styles.noContentText}>No hay contenido 3D disponible aún.</Text>
        )}
      </View>
      
      {/* VR Device Registration Modal */}
      <Modal
        visible={showDeviceModal}
        animationType="slide"
        onRequestClose={() => setShowDeviceModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Registrar Dispositivo VR</Text>
            <Button title="Cerrar" onPress={() => setShowDeviceModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Nombre del Dispositivo:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: Oculus Quest 2"
              value={deviceName}
              onChangeText={setDeviceName}
            />
            
            <Text style={styles.label}>Tipo de Dispositivo:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={deviceType}
                onValueChange={setDeviceType}
                style={styles.picker}
              >
                {deviceTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Resolución:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={resolution}
                onValueChange={setResolution}
                style={styles.picker}
              >
                <Picker.Item label="1920x1080" value="1920x1080" />
                <Picker.Item label="2560x1440" value="2560x1440" />
                <Picker.Item label="3840x2160" value="3840x2160" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Frecuencia de Actualización:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={refreshRate}
                onValueChange={setRefreshRate}
                style={styles.picker}
              >
                <Picker.Item label="72 Hz" value="72" />
                <Picker.Item label="90 Hz" value="90" />
                <Picker.Item label="120 Hz" value="120" />
                <Picker.Item label="144 Hz" value="144" />
              </Picker>
            </View>
            
            <Text style={styles.label}>Campo de Visión:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={fieldOfView}
                onValueChange={setFieldOfView}
                style={styles.picker}
              >
                <Picker.Item label="90°" value="90" />
                <Picker.Item label="100°" value="100" />
                <Picker.Item label="110°" value="110" />
                <Picker.Item label="120°" value="120" />
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
      
      {/* AR Session Modal */}
      <Modal
        visible={showARModal}
        animationType="slide"
        onRequestClose={() => setShowARModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Iniciar Sesión AR</Text>
            <Button title="Cerrar" onPress={() => setShowARModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Plataforma AR:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={arPlatform}
                onValueChange={setArPlatform}
                style={styles.picker}
              >
                {arPlatforms.map((platform) => (
                  <Picker.Item key={platform.value} label={platform.label} value={platform.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Información del Dispositivo:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: iPhone 12 Pro con LiDAR"
              value={arDeviceInfo}
              onChangeText={setArDeviceInfo}
            />
            
            <Button
              title={startARMutation.isLoading ? 'Iniciando...' : 'Iniciar Sesión AR'}
              onPress={handleStartARSession}
              disabled={startARMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* 3D Content Upload Modal */}
      <Modal
        visible={showContentModal}
        animationType="slide"
        onRequestClose={() => setShowContentModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Subir Contenido 3D</Text>
            <Button title="Cerrar" onPress={() => setShowContentModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Nombre del Contenido:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: Modelo de Casa"
              value={contentName}
              onChangeText={setContentName}
            />
            
            <Text style={styles.label}>Descripción:</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Descripción del contenido 3D..."
              value={contentDescription}
              onChangeText={setContentDescription}
              multiline
              numberOfLines={3}
            />
            
            <Text style={styles.label}>Tipo de Contenido:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={contentType}
                onValueChange={setContentType}
                style={styles.picker}
              >
                {contentTypes.map((type) => (
                  <Picker.Item key={type.value} label={type.label} value={type.value} />
                ))}
              </Picker>
            </View>
            
            <Text style={styles.label}>Ruta del Archivo:</Text>
            <TextInput
              style={styles.input}
              placeholder="/path/to/model.gltf"
              value={filePath}
              onChangeText={setFilePath}
            />
            
            <Text style={styles.label}>Formato del Archivo:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={fileFormat}
                onValueChange={setFileFormat}
                style={styles.picker}
              >
                {fileFormats.map((format) => (
                  <Picker.Item key={format.value} label={format.label} value={format.value} />
                ))}
              </Picker>
            </View>
            
            <Button
              title={uploadContentMutation.isLoading ? 'Subiendo...' : 'Subir Contenido'}
              onPress={handleUploadContent}
              disabled={uploadContentMutation.isLoading}
            />
          </ScrollView>
        </View>
      </Modal>
      
      {/* VR Scene Creation Modal */}
      <Modal
        visible={showSceneModal}
        animationType="slide"
        onRequestClose={() => setShowSceneModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Crear Escena VR</Text>
            <Button title="Cerrar" onPress={() => setShowSceneModal(false)} />
          </View>
          <ScrollView style={styles.modalContent}>
            <Text style={styles.label}>Nombre de la Escena:</Text>
            <TextInput
              style={styles.input}
              placeholder="Ej: Sala de Reuniones Virtual"
              value={sceneName}
              onChangeText={setSceneName}
            />
            
            <Text style={styles.label}>Descripción:</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Descripción de la escena VR..."
              value={sceneDescription}
              onChangeText={setSceneDescription}
              multiline
              numberOfLines={3}
            />
            
            <Text style={styles.label}>Contenido 3D Disponible:</Text>
            {availableContent && availableContent.length > 0 ? (
              <View style={styles.contentSelection}>
                {availableContent.map((content) => (
                  <TouchableOpacity
                    key={content.content_id}
                    style={[
                      styles.contentOption,
                      selectedContent.includes(content.content_id) && styles.contentOptionSelected
                    ]}
                    onPress={() => {
                      if (selectedContent.includes(content.content_id)) {
                        setSelectedContent(selectedContent.filter(id => id !== content.content_id));
                      } else {
                        setSelectedContent([...selectedContent, content.content_id]);
                      }
                    }}
                  >
                    <Text style={styles.contentOptionText}>{content.name}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            ) : (
              <Text style={styles.noContentText}>No hay contenido 3D disponible.</Text>
            )}
            
            <Button
              title={createSceneMutation.isLoading ? 'Creando...' : 'Crear Escena VR'}
              onPress={handleCreateScene}
              disabled={createSceneMutation.isLoading}
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
  contentSection: {
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
  contentList: {
    maxHeight: 300,
  },
  contentCard: {
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#2196F3',
  },
  contentTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  contentInfo: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  contentDescription: {
    fontSize: 14,
    color: '#333',
    marginBottom: 4,
  },
  contentDate: {
    fontSize: 12,
    color: '#999',
  },
  noContentText: {
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
  contentSelection: {
    maxHeight: 200,
    marginBottom: 16,
  },
  contentOption: {
    padding: 12,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    marginBottom: 8,
    backgroundColor: '#f9f9f9',
  },
  contentOptionSelected: {
    backgroundColor: '#e3f2fd',
    borderColor: '#2196F3',
  },
  contentOptionText: {
    fontSize: 14,
    color: '#333',
  },
});

export default MetaverseScreen;


