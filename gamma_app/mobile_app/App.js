/**
 * Gamma App - Mobile Application
 * React Native app for AI-powered content generation
 */

import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
  FlatList,
  Image,
  Dimensions,
  RefreshControl,
} from 'react-native';
import {
  Colors,
  DebugInstructions,
  Header,
  LearnMoreLinks,
  ReloadInstructions,
} from 'react-native/Libraries/NewAppScreen';
import Icon from 'react-native-vector-icons/MaterialIcons';
import LinearGradient from 'react-native-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width, height } = Dimensions.get('window');

const API_BASE_URL = 'http://localhost:8000/api/v1';

const App = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [user, setUser] = useState(null);
  const [content, setContent] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTab, setSelectedTab] = useState('home');

  const backgroundStyle = {
    backgroundColor: isDarkMode ? Colors.darker : Colors.lighter,
  };

  useEffect(() => {
    loadUserData();
    loadContent();
  }, []);

  const loadUserData = async () => {
    try {
      const userData = await AsyncStorage.getItem('user');
      if (userData) {
        setUser(JSON.parse(userData));
      }
    } catch (error) {
      console.error('Error loading user data:', error);
    }
  };

  const loadContent = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/content/list`);
      const data = await response.json();
      setContent(data.content || []);
    } catch (error) {
      console.error('Error loading content:', error);
      Alert.alert('Error', 'Failed to load content');
    } finally {
      setIsLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadContent();
    setRefreshing(false);
  };

  const generateContent = async (type, topic, style) => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/content/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content_type: type,
          topic: topic,
          style: style,
          length: 'medium',
          language: 'es'
        }),
      });

      const result = await response.json();
      
      if (result.success) {
        Alert.alert('Success', 'Content generated successfully!');
        loadContent();
      } else {
        Alert.alert('Error', result.error || 'Failed to generate content');
      }
    } catch (error) {
      console.error('Error generating content:', error);
      Alert.alert('Error', 'Failed to generate content');
    } finally {
      setIsLoading(false);
    }
  };

  const renderHeader = () => (
    <LinearGradient
      colors={['#667eea', '#764ba2']}
      style={styles.header}
    >
      <View style={styles.headerContent}>
        <View style={styles.headerLeft}>
          <Icon name="auto-awesome" size={28} color="white" />
          <Text style={styles.headerTitle}>Gamma App</Text>
        </View>
        <TouchableOpacity style={styles.profileButton}>
          <Icon name="account-circle" size={28} color="white" />
        </TouchableOpacity>
      </View>
    </LinearGradient>
  );

  const renderTabBar = () => (
    <View style={styles.tabBar}>
      <TouchableOpacity
        style={[styles.tab, selectedTab === 'home' && styles.activeTab]}
        onPress={() => setSelectedTab('home')}
      >
        <Icon
          name="home"
          size={24}
          color={selectedTab === 'home' ? '#667eea' : '#666'}
        />
        <Text style={[styles.tabText, selectedTab === 'home' && styles.activeTabText]}>
          Inicio
        </Text>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={[styles.tab, selectedTab === 'create' && styles.activeTab]}
        onPress={() => setSelectedTab('create')}
      >
        <Icon
          name="add-circle"
          size={24}
          color={selectedTab === 'create' ? '#667eea' : '#666'}
        />
        <Text style={[styles.tabText, selectedTab === 'create' && styles.activeTabText]}>
          Crear
        </Text>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={[styles.tab, selectedTab === 'library' && styles.activeTab]}
        onPress={() => setSelectedTab('library')}
      >
        <Icon
          name="library-books"
          size={24}
          color={selectedTab === 'library' ? '#667eea' : '#666'}
        />
        <Text style={[styles.tabText, selectedTab === 'library' && styles.activeTabText]}>
          Biblioteca
        </Text>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={[styles.tab, selectedTab === 'profile' && styles.activeTab]}
        onPress={() => setSelectedTab('profile')}
      >
        <Icon
          name="person"
          size={24}
          color={selectedTab === 'profile' ? '#667eea' : '#666'}
        />
        <Text style={[styles.tabText, selectedTab === 'profile' && styles.activeTabText]}>
          Perfil
        </Text>
      </TouchableOpacity>
    </View>
  );

  const renderHomeScreen = () => (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <View style={styles.welcomeSection}>
        <Text style={styles.welcomeTitle}>¡Bienvenido a Gamma App!</Text>
        <Text style={styles.welcomeSubtitle}>
          Genera contenido profesional con IA
        </Text>
      </View>

      <View style={styles.quickActions}>
        <Text style={styles.sectionTitle}>Acciones Rápidas</Text>
        <View style={styles.actionGrid}>
          <TouchableOpacity
            style={styles.actionCard}
            onPress={() => generateContent('presentation', 'Presentación de Ventas', 'modern')}
          >
            <LinearGradient
              colors={['#667eea', '#764ba2']}
              style={styles.actionGradient}
            >
              <Icon name="slideshow" size={32} color="white" />
              <Text style={styles.actionText}>Presentación</Text>
            </LinearGradient>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionCard}
            onPress={() => generateContent('document', 'Reporte Mensual', 'professional')}
          >
            <LinearGradient
              colors={['#f093fb', '#f5576c']}
              style={styles.actionGradient}
            >
              <Icon name="description" size={32} color="white" />
              <Text style={styles.actionText}>Documento</Text>
            </LinearGradient>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionCard}
            onPress={() => generateContent('webpage', 'Landing Page', 'creative')}
          >
            <LinearGradient
              colors={['#4facfe', '#00f2fe']}
              style={styles.actionGradient}
            >
              <Icon name="web" size={32} color="white" />
              <Text style={styles.actionText}>Página Web</Text>
            </LinearGradient>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionCard}
            onPress={() => generateContent('blog', 'Artículo de Blog', 'casual')}
          >
            <LinearGradient
              colors={['#43e97b', '#38f9d7']}
              style={styles.actionGradient}
            >
              <Icon name="article" size={32} color="white" />
              <Text style={styles.actionText}>Blog</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.recentContent}>
        <Text style={styles.sectionTitle}>Contenido Reciente</Text>
        {isLoading ? (
          <ActivityIndicator size="large" color="#667eea" />
        ) : (
          <FlatList
            data={content.slice(0, 5)}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <View style={styles.contentItem}>
                <View style={styles.contentIcon}>
                  <Icon
                    name={
                      item.type === 'presentation' ? 'slideshow' :
                      item.type === 'document' ? 'description' :
                      item.type === 'webpage' ? 'web' : 'article'
                    }
                    size={24}
                    color="#667eea"
                  />
                </View>
                <View style={styles.contentInfo}>
                  <Text style={styles.contentTitle}>{item.title}</Text>
                  <Text style={styles.contentType}>{item.type}</Text>
                  <Text style={styles.contentDate}>
                    {new Date(item.created_at).toLocaleDateString()}
                  </Text>
                </View>
                <TouchableOpacity style={styles.contentAction}>
                  <Icon name="more-vert" size={20} color="#666" />
                </TouchableOpacity>
              </View>
            )}
            scrollEnabled={false}
          />
        )}
      </View>
    </ScrollView>
  );

  const renderCreateScreen = () => (
    <ScrollView style={styles.container}>
      <View style={styles.createSection}>
        <Text style={styles.sectionTitle}>Crear Nuevo Contenido</Text>
        
        <View style={styles.createForm}>
          <Text style={styles.inputLabel}>Tipo de Contenido</Text>
          <View style={styles.typeSelector}>
            {['presentation', 'document', 'webpage', 'blog'].map((type) => (
              <TouchableOpacity
                key={type}
                style={styles.typeOption}
                onPress={() => generateContent(type, 'Nuevo Contenido', 'professional')}
              >
                <Text style={styles.typeText}>{type}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <Text style={styles.inputLabel}>Tema</Text>
          <TextInput
            style={styles.textInput}
            placeholder="Ingresa el tema de tu contenido"
            placeholderTextColor="#999"
          />

          <Text style={styles.inputLabel}>Estilo</Text>
          <View style={styles.styleSelector}>
            {['modern', 'professional', 'creative', 'minimalist'].map((style) => (
              <TouchableOpacity
                key={style}
                style={styles.styleOption}
                onPress={() => generateContent('document', 'Nuevo Contenido', style)}
              >
                <Text style={styles.styleText}>{style}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <TouchableOpacity
            style={styles.generateButton}
            onPress={() => generateContent('document', 'Nuevo Contenido', 'professional')}
            disabled={isLoading}
          >
            <LinearGradient
              colors={['#667eea', '#764ba2']}
              style={styles.generateGradient}
            >
              {isLoading ? (
                <ActivityIndicator color="white" />
              ) : (
                <>
                  <Icon name="auto-awesome" size={20} color="white" />
                  <Text style={styles.generateText}>Generar Contenido</Text>
                </>
              )}
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );

  const renderLibraryScreen = () => (
    <ScrollView style={styles.container}>
      <View style={styles.librarySection}>
        <Text style={styles.sectionTitle}>Mi Biblioteca</Text>
        
        {isLoading ? (
          <ActivityIndicator size="large" color="#667eea" />
        ) : (
          <FlatList
            data={content}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <View style={styles.libraryItem}>
                <View style={styles.libraryIcon}>
                  <Icon
                    name={
                      item.type === 'presentation' ? 'slideshow' :
                      item.type === 'document' ? 'description' :
                      item.type === 'webpage' ? 'web' : 'article'
                    }
                    size={28}
                    color="#667eea"
                  />
                </View>
                <View style={styles.libraryInfo}>
                  <Text style={styles.libraryTitle}>{item.title}</Text>
                  <Text style={styles.libraryType}>{item.type}</Text>
                  <Text style={styles.libraryDate}>
                    {new Date(item.created_at).toLocaleDateString()}
                  </Text>
                </View>
                <View style={styles.libraryActions}>
                  <TouchableOpacity style={styles.libraryAction}>
                    <Icon name="download" size={20} color="#667eea" />
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.libraryAction}>
                    <Icon name="share" size={20} color="#667eea" />
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.libraryAction}>
                    <Icon name="delete" size={20} color="#f5576c" />
                  </TouchableOpacity>
                </View>
              </View>
            )}
            scrollEnabled={false}
          />
        )}
      </View>
    </ScrollView>
  );

  const renderProfileScreen = () => (
    <ScrollView style={styles.container}>
      <View style={styles.profileSection}>
        <View style={styles.profileHeader}>
          <View style={styles.profileAvatar}>
            <Icon name="account-circle" size={80} color="#667eea" />
          </View>
          <Text style={styles.profileName}>
            {user?.name || 'Usuario'}
          </Text>
          <Text style={styles.profileEmail}>
            {user?.email || 'usuario@example.com'}
          </Text>
        </View>

        <View style={styles.profileStats}>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{content.length}</Text>
            <Text style={styles.statLabel}>Contenidos</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>12</Text>
            <Text style={styles.statLabel}>Este Mes</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>5</Text>
            <Text style={styles.statLabel}>Favoritos</Text>
          </View>
        </View>

        <View style={styles.profileActions}>
          <TouchableOpacity style={styles.profileAction}>
            <Icon name="settings" size={24} color="#667eea" />
            <Text style={styles.profileActionText}>Configuración</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.profileAction}>
            <Icon name="help" size={24} color="#667eea" />
            <Text style={styles.profileActionText}>Ayuda</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.profileAction}>
            <Icon name="info" size={24} color="#667eea" />
            <Text style={styles.profileActionText}>Acerca de</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.profileAction}>
            <Icon name="logout" size={24} color="#f5576c" />
            <Text style={[styles.profileActionText, { color: '#f5576c' }]}>
              Cerrar Sesión
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );

  const renderContent = () => {
    switch (selectedTab) {
      case 'home':
        return renderHomeScreen();
      case 'create':
        return renderCreateScreen();
      case 'library':
        return renderLibraryScreen();
      case 'profile':
        return renderProfileScreen();
      default:
        return renderHomeScreen();
    }
  };

  return (
    <SafeAreaView style={backgroundStyle}>
      <StatusBar
        barStyle={isDarkMode ? 'light-content' : 'dark-content'}
        backgroundColor="#667eea"
      />
      {renderHeader()}
      {renderContent()}
      {renderTabBar()}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    paddingTop: 20,
    paddingBottom: 15,
    paddingHorizontal: 20,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginLeft: 10,
  },
  profileButton: {
    padding: 5,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    paddingVertical: 10,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 5,
  },
  activeTab: {
    backgroundColor: '#f0f0f0',
  },
  tabText: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  activeTabText: {
    color: '#667eea',
    fontWeight: 'bold',
  },
  welcomeSection: {
    padding: 20,
    alignItems: 'center',
  },
  welcomeTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 10,
  },
  welcomeSubtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  quickActions: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  actionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  actionCard: {
    width: (width - 60) / 2,
    height: 120,
    marginBottom: 15,
    borderRadius: 15,
    overflow: 'hidden',
  },
  actionGradient: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  actionText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 8,
  },
  recentContent: {
    padding: 20,
  },
  contentItem: {
    flexDirection: 'row',
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  contentIcon: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  contentInfo: {
    flex: 1,
  },
  contentTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  contentType: {
    fontSize: 14,
    color: '#667eea',
    textTransform: 'capitalize',
  },
  contentDate: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  contentAction: {
    padding: 5,
  },
  createSection: {
    padding: 20,
  },
  createForm: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    marginTop: 15,
  },
  typeSelector: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  typeOption: {
    width: (width - 80) / 2,
    backgroundColor: '#f0f0f0',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 10,
  },
  typeText: {
    fontSize: 14,
    color: '#333',
    textTransform: 'capitalize',
  },
  styleSelector: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  styleOption: {
    width: (width - 100) / 2,
    backgroundColor: '#f0f0f0',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 8,
  },
  styleText: {
    fontSize: 12,
    color: '#333',
    textTransform: 'capitalize',
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 10,
    padding: 15,
    fontSize: 16,
    backgroundColor: 'white',
  },
  generateButton: {
    marginTop: 20,
    borderRadius: 15,
    overflow: 'hidden',
  },
  generateGradient: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 15,
  },
  generateText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  librarySection: {
    padding: 20,
  },
  libraryItem: {
    flexDirection: 'row',
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  libraryIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  libraryInfo: {
    flex: 1,
  },
  libraryTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  libraryType: {
    fontSize: 14,
    color: '#667eea',
    textTransform: 'capitalize',
  },
  libraryDate: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  libraryActions: {
    flexDirection: 'row',
  },
  libraryAction: {
    padding: 8,
    marginLeft: 5,
  },
  profileSection: {
    padding: 20,
  },
  profileHeader: {
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 30,
    borderRadius: 15,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  profileAvatar: {
    marginBottom: 15,
  },
  profileName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  profileEmail: {
    fontSize: 16,
    color: '#666',
  },
  profileStats: {
    flexDirection: 'row',
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 15,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#667eea',
  },
  statLabel: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  profileActions: {
    backgroundColor: 'white',
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  profileAction: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  profileActionText: {
    fontSize: 16,
    color: '#333',
    marginLeft: 15,
  },
});

export default App;



