import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  Button, 
  StyleSheet, 
  ScrollView, 
  TouchableOpacity,
  Image,
  Dimensions,
  Alert
} from 'react-native';
import { StackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

type RootStackParamList = {
  Home: undefined;
  ContentGeneration: undefined;
  Chatbot: undefined;
  VideoProcessing: undefined;
  IoT: undefined;
  QuantumComputing: undefined;
  AIService: undefined;
  Analytics: undefined;
  Workflow: undefined;
  ML: undefined;
  Agents: undefined;
  BI: undefined;
  CloudNative: undefined;
  Security: undefined;
  Blockchain: undefined;
  BUL: undefined;
  Metaverse: undefined;
  Consciousness: undefined;
  QuantumAI: undefined;
  NeuralInterface: undefined;
      SpaceExploration: undefined;
      DimensionHopping: undefined;
      ConsciousnessTranscendence: undefined;
      InfiniteEvolution: undefined;
      TranscendentAbsolute: undefined;
      UltimateCosmic: undefined;
      EternalOmnipotent: undefined;
      InfiniteDivine: undefined;
    };

type Props = {
  navigation: StackNavigationProp<RootStackParamList, 'Home'>;
};

interface FeatureCard {
  id: string;
  title: string;
  description: string;
  icon: string;
  color: string;
  route: keyof RootStackParamList;
  category: 'ai' | 'analytics' | 'integration' | 'advanced' | 'security' | 'blockchain' | 'bul';
}

const HomeScreen: React.FC<Props> = ({ navigation }) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [userStats, setUserStats] = useState({
    totalDocuments: 0,
    activeProjects: 0,
    aiInteractions: 0,
    lastActivity: 'Nunca'
  });

  const features: FeatureCard[] = [
    // AI & Content Generation
    {
      id: 'content-generation',
      title: 'AI Content Generation',
      description: 'Generate presentations, documents, and web pages with advanced AI',
      icon: 'document-text',
      color: '#4CAF50',
      route: 'ContentGeneration',
      category: 'ai'
    },
    {
      id: 'ai-chatbot',
      title: 'AI Chatbot',
      description: 'Intelligent conversational AI with contextual memory',
      icon: 'chatbubbles',
      color: '#2196F3',
      route: 'Chatbot',
      category: 'ai'
    },
    {
      id: 'ai-service',
      title: 'Advanced AI Services',
      description: 'Advanced AI models, training, and optimization',
      icon: 'brain',
      color: '#9C27B0',
      route: 'AIService',
      category: 'ai'
    },
    {
      id: 'ml-engine',
      title: 'Machine Learning Engine',
      description: 'Advanced ML algorithms and model training',
      icon: 'analytics',
      color: '#FF9800',
      route: 'ML',
      category: 'ai'
    },

    // Analytics & Intelligence
    {
      id: 'analytics',
      title: 'Advanced Analytics',
      description: 'Real-time analytics and predictive insights',
      icon: 'bar-chart',
      color: '#00BCD4',
      route: 'Analytics',
      category: 'analytics'
    },
    {
      id: 'business-intelligence',
      title: 'Business Intelligence',
      description: 'Data warehousing, ETL, and business insights',
      icon: 'trending-up',
      color: '#795548',
      route: 'BI',
      category: 'analytics'
    },

    // Integration & Automation
    {
      id: 'workflow',
      title: 'Workflow Automation',
      description: 'Complex workflows and orchestration',
      icon: 'git-network',
      color: '#607D8B',
      route: 'Workflow',
      category: 'integration'
    },
    {
      id: 'ai-agents',
      title: 'AI Agents',
      description: 'Autonomous agents and multi-agent systems',
      icon: 'people',
      color: '#E91E63',
      route: 'Agents',
      category: 'integration'
    },

    // Advanced Technologies
    {
      id: 'video-processing',
      title: 'Video Processing',
      description: 'AI-powered video effects and optimization',
      icon: 'videocam',
      color: '#F44336',
      route: 'VideoProcessing',
      category: 'advanced'
    },
    {
      id: 'iot-edge',
      title: 'IoT & Edge Computing',
      description: 'Device management and edge processing',
      icon: 'hardware-chip',
      color: '#3F51B5',
      route: 'IoT',
      category: 'advanced'
    },
    {
      id: 'quantum-computing',
      title: 'Quantum Computing',
      description: 'Quantum algorithms and simulations',
      icon: 'nuclear',
      color: '#673AB7',
      route: 'QuantumComputing',
      category: 'advanced'
    },
    {
      id: 'cloud-native',
      title: 'Cloud-Native Deployment',
      description: 'Kubernetes, CI/CD, and auto-scaling',
      icon: 'cloud',
      color: '#009688',
      route: 'CloudNative',
      category: 'advanced'
    },

    // Security & Blockchain
    {
      id: 'security',
      title: 'Advanced Security',
      description: 'Threat detection and compliance',
      icon: 'shield-checkmark',
      color: '#FF5722',
      route: 'Security',
      category: 'security'
    },
    {
      id: 'blockchain',
      title: 'Blockchain & NFTs',
      description: 'Smart contracts and decentralized identity',
      icon: 'link',
      color: '#FFC107',
      route: 'Blockchain',
      category: 'blockchain'
    },

    // BUL Integration
    {
      id: 'bul',
      title: 'BUL - Document Generation',
      description: 'Business Universal Language for enterprise documents',
      icon: 'business',
      color: '#4CAF50',
      route: 'BUL',
      category: 'bul'
    },

    // Metaverse AR/VR
    {
      id: 'metaverse',
      title: 'Metaverse AR/VR',
      description: 'Immersive experiences and augmented reality',
      icon: 'cube',
      color: '#673AB7',
      route: 'Metaverse',
      category: 'advanced'
    },

    // Consciousness AI
    {
      id: 'consciousness',
      title: 'Consciousness AI',
      description: 'Self-aware and empathetic artificial intelligence',
      icon: 'brain',
      color: '#E91E63',
      route: 'Consciousness',
      category: 'ai'
    },

    // Quantum AI
    {
      id: 'quantum_ai',
      title: 'Quantum AI',
      description: 'Quantum computing and artificial intelligence',
      icon: 'nuclear',
      color: '#FF5722',
      route: 'QuantumAI',
      category: 'advanced'
    },

    // Neural Interface
    {
      id: 'neural_interface',
      title: 'Neural Interface',
      description: 'Brain-computer interface and neural communication',
      icon: 'pulse',
      color: '#3F51B5',
      route: 'NeuralInterface',
      category: 'advanced'
    },

      // Space Exploration
      {
        id: 'space_exploration',
        title: 'Space Exploration',
        description: 'Space mission management and interplanetary exploration',
        icon: 'rocket',
        color: '#FF6B35',
        route: 'SpaceExploration',
        category: 'advanced'
      },

      // Dimension Hopping
      {
        id: 'dimension_hopping',
        title: 'Dimension Hopping',
        description: 'Interdimensional travel and reality manipulation',
        icon: 'layers',
        color: '#9C27B0',
        route: 'DimensionHopping',
        category: 'advanced'
      },

      // Consciousness Transcendence
      {
        id: 'consciousness_transcendence',
        title: 'Consciousness Transcendence',
        description: 'Consciousness evolution and universal harmony',
        icon: 'flower',
        color: '#FF9800',
        route: 'ConsciousnessTranscendence',
        category: 'advanced'
      },

      // Infinite Evolution
      {
        id: 'infinite_evolution',
        title: 'Infinite Evolution',
        description: 'Infinite evolution and omnipotent creation',
        icon: 'infinite',
        color: '#E91E63',
        route: 'InfiniteEvolution',
        category: 'advanced'
      },

      // Transcendent Absolute
      {
        id: 'transcendent_absolute',
        title: 'Transcendent Absolute',
        description: 'Transcendent omniverse and absolute divine',
        icon: 'star',
        color: '#9C27B0',
        route: 'TranscendentAbsolute',
        category: 'advanced'
      },

      // Ultimate Cosmic
      {
        id: 'ultimate_cosmic',
        title: 'Ultimate Cosmic',
        description: 'Ultimate cosmic and infinite universal',
        icon: 'planet',
        color: '#FF5722',
        route: 'UltimateCosmic',
        category: 'advanced'
      },

      // Eternal Omnipotent
      {
        id: 'eternal_omnipotent',
        title: 'Eternal Omnipotent',
        description: 'Eternal infinite and omnipotent ultimate',
        icon: 'infinity',
        color: '#673AB7',
        route: 'EternalOmnipotent',
        category: 'advanced'
      },

      // Infinite Divine
      {
        id: 'infinite_divine',
        title: 'Infinite Divine',
        description: 'Infinite absolute and ultimate divine',
        icon: 'star',
        color: '#E91E63',
        route: 'InfiniteDivine',
        category: 'advanced'
      }
  ];

  const categories = [
    { id: 'all', name: 'All Features', icon: 'grid' },
    { id: 'ai', name: 'AI & ML', icon: 'brain' },
    { id: 'analytics', name: 'Analytics', icon: 'analytics' },
    { id: 'integration', name: 'Integration', icon: 'git-network' },
    { id: 'advanced', name: 'Advanced', icon: 'rocket' },
    { id: 'security', name: 'Security', icon: 'shield' },
    { id: 'blockchain', name: 'Blockchain', icon: 'link' },
    { id: 'bul', name: 'BUL', icon: 'business' }
  ];

  const filteredFeatures = selectedCategory === 'all' 
    ? features 
    : features.filter(feature => feature.category === selectedCategory);

  useEffect(() => {
    // Simulate loading user stats
    const loadUserStats = async () => {
      // In a real app, this would fetch from API
      setUserStats({
        totalDocuments: 47,
        activeProjects: 12,
        aiInteractions: 156,
        lastActivity: '2 horas atrás'
      });
    };
    
    loadUserStats();
  }, []);

  const renderFeatureCard = (feature: FeatureCard) => (
    <TouchableOpacity
      key={feature.id}
      style={[styles.featureCard, { borderLeftColor: feature.color }]}
      onPress={() => navigation.navigate(feature.route)}
    >
      <View style={styles.featureHeader}>
        <View style={[styles.iconContainer, { backgroundColor: feature.color }]}>
          <Ionicons name={feature.icon as any} size={24} color="white" />
        </View>
        <Text style={styles.featureTitle}>{feature.title}</Text>
      </View>
      <Text style={styles.featureDescription}>{feature.description}</Text>
      <View style={styles.featureFooter}>
        <Ionicons name="chevron-forward" size={16} color="#666" />
      </View>
    </TouchableOpacity>
  );

  const renderCategoryButton = (category: typeof categories[0]) => (
    <TouchableOpacity
      key={category.id}
      style={[
        styles.categoryButton,
        selectedCategory === category.id && styles.categoryButtonActive
      ]}
      onPress={() => setSelectedCategory(category.id)}
    >
      <Ionicons 
        name={category.icon as any} 
        size={20} 
        color={selectedCategory === category.id ? '#fff' : '#666'} 
      />
      <Text style={[
        styles.categoryButtonText,
        selectedCategory === category.id && styles.categoryButtonTextActive
      ]}>
        {category.name}
      </Text>
    </TouchableOpacity>
  );

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <Text style={styles.welcomeText}>Welcome to</Text>
          <Text style={styles.appTitle}>Gamma App</Text>
          <Text style={styles.subtitle}>Enterprise AI Platform</Text>
        </View>
        <View style={styles.headerStats}>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{userStats.totalDocuments}</Text>
            <Text style={styles.statLabel}>Documents</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{userStats.activeProjects}</Text>
            <Text style={styles.statLabel}>Projects</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{userStats.aiInteractions}</Text>
            <Text style={styles.statLabel}>AI Calls</Text>
          </View>
        </View>
      </View>

      {/* Quick Actions */}
      <View style={styles.quickActions}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.quickActionButtons}>
          <TouchableOpacity 
            style={[styles.quickActionButton, { backgroundColor: '#4CAF50' }]}
            onPress={() => navigation.navigate('ContentGeneration')}
          >
            <Ionicons name="add" size={24} color="white" />
            <Text style={styles.quickActionText}>New Document</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.quickActionButton, { backgroundColor: '#2196F3' }]}
            onPress={() => navigation.navigate('Chatbot')}
          >
            <Ionicons name="chatbubbles" size={24} color="white" />
            <Text style={styles.quickActionText}>AI Chat</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.quickActionButton, { backgroundColor: '#FF9800' }]}
            onPress={() => navigation.navigate('BUL')}
          >
            <Ionicons name="business" size={24} color="white" />
            <Text style={styles.quickActionText}>BUL Generate</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Categories */}
      <View style={styles.categoriesSection}>
        <Text style={styles.sectionTitle}>Categories</Text>
        <ScrollView 
          horizontal 
          showsHorizontalScrollIndicator={false}
          style={styles.categoriesScroll}
        >
          {categories.map(renderCategoryButton)}
        </ScrollView>
      </View>

      {/* Features Grid */}
      <View style={styles.featuresSection}>
        <View style={styles.featuresHeader}>
          <Text style={styles.sectionTitle}>
            {selectedCategory === 'all' ? 'All Features' : categories.find(c => c.id === selectedCategory)?.name}
          </Text>
          <Text style={styles.featuresCount}>
            {filteredFeatures.length} features
          </Text>
        </View>
        
        <View style={styles.featuresGrid}>
          {filteredFeatures.map(renderFeatureCard)}
        </View>
      </View>

      {/* System Status */}
      <View style={styles.systemStatus}>
        <Text style={styles.sectionTitle}>System Status</Text>
        <View style={styles.statusItems}>
          <View style={styles.statusItem}>
            <View style={[styles.statusDot, { backgroundColor: '#4CAF50' }]} />
            <Text style={styles.statusText}>All Systems Operational</Text>
          </View>
          <View style={styles.statusItem}>
            <View style={[styles.statusDot, { backgroundColor: '#4CAF50' }]} />
            <Text style={styles.statusText}>BUL Integration Active</Text>
          </View>
          <View style={styles.statusItem}>
            <View style={[styles.statusDot, { backgroundColor: '#4CAF50' }]} />
            <Text style={styles.statusText}>26 Services Running</Text>
          </View>
        </View>
      </View>

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Last activity: {userStats.lastActivity}
        </Text>
        <Text style={styles.footerText}>
          Gamma App v1.0.0 • BUL Integration v1.0.0
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    backgroundColor: '#fff',
    padding: 20,
    paddingTop: 40,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  headerContent: {
    marginBottom: 20,
  },
  welcomeText: {
    fontSize: 16,
    color: '#666',
    marginBottom: 4,
  },
  appTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
  },
  headerStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  quickActions: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  quickActionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  quickActionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    marginHorizontal: 4,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  quickActionText: {
    color: 'white',
    fontWeight: '600',
    marginLeft: 8,
  },
  categoriesSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  categoriesScroll: {
    marginTop: 8,
  },
  categoryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 12,
    borderRadius: 20,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  categoryButtonActive: {
    backgroundColor: '#2196F3',
    borderColor: '#2196F3',
  },
  categoryButtonText: {
    marginLeft: 6,
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  categoryButtonTextActive: {
    color: '#fff',
  },
  featuresSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  featuresHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  featuresCount: {
    fontSize: 14,
    color: '#666',
  },
  featuresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  featureCard: {
    width: (width - 60) / 2,
    backgroundColor: '#fff',
    padding: 16,
    marginBottom: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  featureHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  featureTitle: {
    flex: 1,
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  featureDescription: {
    fontSize: 12,
    color: '#666',
    lineHeight: 16,
    marginBottom: 12,
  },
  featureFooter: {
    alignItems: 'flex-end',
  },
  systemStatus: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  statusItems: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statusItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 12,
  },
  statusText: {
    fontSize: 14,
    color: '#333',
  },
  footer: {
    padding: 20,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    color: '#999',
    marginBottom: 4,
  },
});

export default HomeScreen;

