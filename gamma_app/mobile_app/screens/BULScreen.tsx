import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  TextInput, 
  Button, 
  StyleSheet, 
  ScrollView, 
  ActivityIndicator, 
  Alert, 
  FlatList,
  TouchableOpacity,
  Modal
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import axios from 'axios';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { format } from 'date-fns';

// Types
interface BULDocumentRequest {
  query: string;
  business_area?: string;
  document_type?: string;
  priority: number;
  user_id?: string;
  metadata?: Record<string, any>;
}

interface BULDocumentResponse {
  task_id: string;
  status: string;
  message: string;
  estimated_time?: number;
  document_id?: string;
}

interface BULTask {
  task_id: string;
  status: string;
  progress: number;
  created_at: string;
  completed_at?: string;
  business_area?: string;
  document_type?: string;
  query: string;
  priority: number;
  result?: any;
  error?: string;
}

interface BULDocument {
  document_id: string;
  title: string;
  content: string;
  format: string;
  word_count: number;
  business_area: string;
  document_type: string;
  query: string;
  generated_at: string;
  metadata?: Record<string, any>;
}

// API calls
const generateDocument = async (request: BULDocumentRequest): Promise<BULDocumentResponse> => {
  const response = await axios.post('http://localhost:8000/bul/documents/generate', request);
  return response.data;
};

const getTaskStatus = async (taskId: string): Promise<BULTask> => {
  const response = await axios.get(`http://localhost:8000/bul/tasks/${taskId}/status`);
  return response.data;
};

const getDocuments = async (): Promise<BULDocument[]> => {
  const response = await axios.get('http://localhost:8000/bul/documents');
  return response.data.documents || [];
};

const getDocument = async (documentId: string): Promise<BULDocument> => {
  const response = await axios.get(`http://localhost:8000/bul/documents/${documentId}`);
  return response.data;
};

const searchDocuments = async (query: string): Promise<BULDocument[]> => {
  const response = await axios.get(`http://localhost:8000/bul/documents/search?q=${encodeURIComponent(query)}`);
  return response.data.documents || [];
};

const BULScreen: React.FC = () => {
  const queryClient = useQueryClient();
  
  // State for document generation
  const [query, setQuery] = useState<string>('');
  const [selectedBusinessArea, setSelectedBusinessArea] = useState<string>('marketing');
  const [selectedDocumentType, setSelectedDocumentType] = useState<string>('estrategia');
  const [priority, setPriority] = useState<number>(1);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  
  // State for document viewing
  const [selectedDocument, setSelectedDocument] = useState<BULDocument | null>(null);
  const [showDocumentModal, setShowDocumentModal] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  
  // Business areas and document types
  const businessAreas = [
    { value: 'marketing', label: 'Marketing' },
    { value: 'ventas', label: 'Ventas' },
    { value: 'operaciones', label: 'Operaciones' },
    { value: 'rrhh', label: 'RRHH' },
    { value: 'finanzas', label: 'Finanzas' },
    { value: 'legal', label: 'Legal' },
    { value: 'técnico', label: 'Técnico' },
    { value: 'contenido', label: 'Contenido' },
    { value: 'estrategia', label: 'Estrategia' },
    { value: 'atencion_cliente', label: 'Atención al Cliente' }
  ];
  
  const documentTypes = [
    { value: 'estrategia', label: 'Estrategia' },
    { value: 'propuesta', label: 'Propuesta' },
    { value: 'manual', label: 'Manual' },
    { value: 'política', label: 'Política' },
    { value: 'reporte', label: 'Reporte' },
    { value: 'plantilla', label: 'Plantilla' }
  ];
  
  // Queries
  const { data: documents, isLoading: isLoadingDocuments, error: documentsError } = useQuery<BULDocument[], Error>(
    'bulDocuments',
    getDocuments,
    {
      refetchInterval: 30000, // Refetch every 30 seconds
    }
  );
  
  const { data: searchResults, isLoading: isSearching } = useQuery<BULDocument[], Error>(
    ['bulSearch', searchQuery],
    () => searchDocuments(searchQuery),
    {
      enabled: searchQuery.length > 2,
    }
  );
  
  // Mutations
  const generateDocumentMutation = useMutation<BULDocumentResponse, Error, BULDocumentRequest>(generateDocument, {
    onSuccess: (data) => {
      setActiveTaskId(data.task_id);
      Alert.alert('Éxito', `Generación de documento iniciada. ID de tarea: ${data.task_id}`);
      queryClient.invalidateQueries('bulDocuments');
    },
    onError: (error) => {
      Alert.alert('Error', `Error al generar documento: ${error.message}`);
      console.error('BUL document generation error:', error);
    },
  });
  
  // Task status polling
  const { data: taskStatus, isLoading: isLoadingTask } = useQuery<BULTask, Error>(
    ['bulTaskStatus', activeTaskId],
    () => getTaskStatus(activeTaskId!),
    {
      enabled: !!activeTaskId,
      refetchInterval: (data) => {
        if (data?.status === 'completado' || data?.status === 'fallido') {
          return false; // Stop polling when completed or failed
        }
        return 2000; // Poll every 2 seconds
      },
    }
  );
  
  // Effects
  useEffect(() => {
    if (taskStatus?.status === 'completado') {
      setActiveTaskId(null);
      queryClient.invalidateQueries('bulDocuments');
      Alert.alert('Completado', 'Documento generado exitosamente');
    } else if (taskStatus?.status === 'fallido') {
      setActiveTaskId(null);
      Alert.alert('Error', `Error en la generación: ${taskStatus.error || 'Error desconocido'}`);
    }
  }, [taskStatus, queryClient]);
  
  // Handlers
  const handleGenerateDocument = () => {
    if (!query.trim()) {
      Alert.alert('Error de entrada', 'Por favor ingresa una consulta.');
      return;
    }
    
    generateDocumentMutation.mutate({
      query: query.trim(),
      business_area: selectedBusinessArea,
      document_type: selectedDocumentType,
      priority: priority,
      user_id: 'mobile_user_123',
    });
  };
  
  const handleViewDocument = async (documentId: string) => {
    try {
      const document = await getDocument(documentId);
      setSelectedDocument(document);
      setShowDocumentModal(true);
    } catch (error) {
      Alert.alert('Error', 'No se pudo cargar el documento');
      console.error('Error loading document:', error);
    }
  };
  
  const handleSearch = () => {
    if (searchQuery.trim().length > 2) {
      queryClient.invalidateQueries(['bulSearch', searchQuery]);
    }
  };
  
  const renderDocument = ({ item }: { item: BULDocument }) => (
    <TouchableOpacity 
      style={styles.documentCard} 
      onPress={() => handleViewDocument(item.document_id)}
    >
      <Text style={styles.documentTitle}>{item.title}</Text>
      <Text style={styles.documentInfo}>
        {item.business_area} • {item.document_type} • {item.word_count} palabras
      </Text>
      <Text style={styles.documentQuery}>{item.query}</Text>
      <Text style={styles.documentDate}>
        {format(new Date(item.generated_at), 'dd/MM/yyyy HH:mm')}
      </Text>
    </TouchableOpacity>
  );
  
  const renderSearchResult = ({ item }: { item: BULDocument }) => (
    <TouchableOpacity 
      style={styles.searchResultCard} 
      onPress={() => handleViewDocument(item.document_id)}
    >
      <Text style={styles.searchResultTitle}>{item.title}</Text>
      <Text style={styles.searchResultInfo}>
        {item.business_area} • {item.document_type}
      </Text>
    </TouchableOpacity>
  );
  
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>BUL - Business Universal Language</Text>
      <Text style={styles.subtitle}>Generación Avanzada de Documentos Empresariales</Text>
      
      {/* Document Generation Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Generar Nuevo Documento</Text>
        
        <Text style={styles.label}>Consulta de Negocio:</Text>
        <TextInput
          style={styles.textInput}
          placeholder="Describe el documento que necesitas generar..."
          value={query}
          onChangeText={setQuery}
          multiline
          numberOfLines={3}
        />
        
        <Text style={styles.label}>Área de Negocio:</Text>
        <View style={styles.pickerContainer}>
          <Picker
            selectedValue={selectedBusinessArea}
            onValueChange={setSelectedBusinessArea}
            style={styles.picker}
          >
            {businessAreas.map((area) => (
              <Picker.Item key={area.value} label={area.label} value={area.value} />
            ))}
          </Picker>
        </View>
        
        <Text style={styles.label}>Tipo de Documento:</Text>
        <View style={styles.pickerContainer}>
          <Picker
            selectedValue={selectedDocumentType}
            onValueChange={setSelectedDocumentType}
            style={styles.picker}
          >
            {documentTypes.map((type) => (
              <Picker.Item key={type.value} label={type.label} value={type.value} />
            ))}
          </Picker>
        </View>
        
        <Text style={styles.label}>Prioridad: {priority}</Text>
        <View style={styles.pickerContainer}>
          <Picker
            selectedValue={priority}
            onValueChange={setPriority}
            style={styles.picker}
          >
            <Picker.Item label="Baja (1)" value={1} />
            <Picker.Item label="Media (2)" value={2} />
            <Picker.Item label="Normal (3)" value={3} />
            <Picker.Item label="Alta (4)" value={4} />
            <Picker.Item label="Crítica (5)" value={5} />
          </Picker>
        </View>
        
        <Button
          title={generateDocumentMutation.isLoading ? 'Generando...' : 'Generar Documento'}
          onPress={handleGenerateDocument}
          disabled={generateDocumentMutation.isLoading || !query.trim()}
        />
        
        {generateDocumentMutation.isLoading && (
          <ActivityIndicator size="large" color="#0000ff" style={styles.activityIndicator} />
        )}
        
        {/* Task Status */}
        {activeTaskId && taskStatus && (
          <View style={styles.taskStatusContainer}>
            <Text style={styles.taskStatusTitle}>Estado de la Tarea:</Text>
            <Text style={styles.taskStatusText}>ID: {taskStatus.task_id}</Text>
            <Text style={styles.taskStatusText}>Estado: {taskStatus.status}</Text>
            <Text style={styles.taskStatusText}>Progreso: {taskStatus.progress}%</Text>
            {isLoadingTask && <ActivityIndicator size="small" color="#0000ff" />}
          </View>
        )}
      </View>
      
      {/* Search Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Buscar Documentos</Text>
        
        <View style={styles.searchContainer}>
          <TextInput
            style={styles.searchInput}
            placeholder="Buscar en documentos..."
            value={searchQuery}
            onChangeText={setSearchQuery}
          />
          <Button title="Buscar" onPress={handleSearch} />
        </View>
        
        {isSearching && <ActivityIndicator size="large" color="#0000ff" style={styles.activityIndicator} />}
        
        {searchResults && searchResults.length > 0 && (
          <View style={styles.searchResultsContainer}>
            <Text style={styles.searchResultsTitle}>Resultados de Búsqueda:</Text>
            <FlatList
              data={searchResults}
              renderItem={renderSearchResult}
              keyExtractor={(item) => item.document_id}
              style={styles.searchResultsList}
            />
          </View>
        )}
      </View>
      
      {/* Documents List Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Documentos Generados</Text>
        
        {isLoadingDocuments && <ActivityIndicator size="large" color="#0000ff" style={styles.activityIndicator} />}
        
        {documentsError && (
          <Text style={styles.errorText}>Error al cargar documentos: {documentsError.message}</Text>
        )}
        
        {documents && documents.length > 0 ? (
          <FlatList
            data={documents}
            renderItem={renderDocument}
            keyExtractor={(item) => item.document_id}
            style={styles.documentsList}
            scrollEnabled={false}
          />
        ) : (
          <Text style={styles.noDocumentsText}>No hay documentos generados aún.</Text>
        )}
      </View>
      
      {/* Document Modal */}
      <Modal
        visible={showDocumentModal}
        animationType="slide"
        onRequestClose={() => setShowDocumentModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Documento</Text>
            <Button title="Cerrar" onPress={() => setShowDocumentModal(false)} />
          </View>
          
          {selectedDocument && (
            <ScrollView style={styles.modalContent}>
              <Text style={styles.modalDocumentTitle}>{selectedDocument.title}</Text>
              <Text style={styles.modalDocumentInfo}>
                {selectedDocument.business_area} • {selectedDocument.document_type} • {selectedDocument.word_count} palabras
              </Text>
              <Text style={styles.modalDocumentQuery}>Consulta: {selectedDocument.query}</Text>
              <Text style={styles.modalDocumentDate}>
                Generado: {format(new Date(selectedDocument.generated_at), 'dd/MM/yyyy HH:mm')}
              </Text>
              <Text style={styles.modalDocumentContent}>{selectedDocument.content}</Text>
            </ScrollView>
          )}
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
  section: {
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
  label: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: '#333',
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    fontSize: 16,
    backgroundColor: '#f9f9f9',
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
  activityIndicator: {
    marginTop: 16,
  },
  taskStatusContainer: {
    backgroundColor: '#e3f2fd',
    padding: 12,
    borderRadius: 8,
    marginTop: 16,
  },
  taskStatusTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#1976d2',
  },
  taskStatusText: {
    fontSize: 14,
    marginBottom: 4,
    color: '#333',
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  searchInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    marginRight: 8,
    fontSize: 16,
    backgroundColor: '#f9f9f9',
  },
  searchResultsContainer: {
    marginTop: 16,
  },
  searchResultsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
  },
  searchResultsList: {
    maxHeight: 200,
  },
  searchResultCard: {
    backgroundColor: '#f0f8ff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
  },
  searchResultTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  searchResultInfo: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  documentsList: {
    maxHeight: 400,
  },
  documentCard: {
    backgroundColor: '#f8f9fa',
    padding: 16,
    borderRadius: 8,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#007bff',
  },
  documentTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  documentInfo: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  documentQuery: {
    fontSize: 14,
    color: '#333',
    fontStyle: 'italic',
    marginBottom: 4,
  },
  documentDate: {
    fontSize: 12,
    color: '#999',
  },
  noDocumentsText: {
    textAlign: 'center',
    fontSize: 16,
    color: '#666',
    fontStyle: 'italic',
    marginTop: 20,
  },
  errorText: {
    color: 'red',
    textAlign: 'center',
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
  modalDocumentTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  modalDocumentInfo: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  modalDocumentQuery: {
    fontSize: 14,
    color: '#333',
    fontStyle: 'italic',
    marginBottom: 8,
  },
  modalDocumentDate: {
    fontSize: 12,
    color: '#999',
    marginBottom: 16,
  },
  modalDocumentContent: {
    fontSize: 16,
    lineHeight: 24,
    color: '#333',
  },
});

export default BULScreen;





















