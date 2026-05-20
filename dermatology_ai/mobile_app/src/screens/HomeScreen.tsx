import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../types';
import { useScreenTracking } from '../hooks/useScreenTracking';

type HomeScreenNavigationProp = StackNavigationProp<RootStackParamList>;

interface QuickAction {
  id: number;
  title: string;
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
  screen: keyof RootStackParamList;
}

const HomeScreen: React.FC = () => {
  const navigation = useNavigation<HomeScreenNavigationProp>();
  useScreenTracking('Home');

  const quickActions: QuickAction[] = [
    {
      id: 1,
      title: 'Análisis Rápido',
      icon: 'camera',
      color: '#6366f1',
      screen: 'MainTabs',
    },
    {
      id: 2,
      title: 'Escaneo en Tiempo Real',
      icon: 'scan',
      color: '#8b5cf6',
      screen: 'RealTimeScan',
    },
    {
      id: 3,
      title: 'Ver Historial',
      icon: 'time',
      color: '#ec4899',
      screen: 'MainTabs',
    },
    {
      id: 4,
      title: 'Recomendaciones',
      icon: 'sparkles',
      color: '#f59e0b',
      screen: 'MainTabs',
    },
  ];

  return (
    <ScrollView style={styles.container}>
      <LinearGradient
        colors={['#6366f1', '#8b5cf6']}
        style={styles.header}
      >
        <Text style={styles.headerTitle}>Dermatology AI</Text>
        <Text style={styles.headerSubtitle}>
          Análisis inteligente de tu piel
        </Text>
      </LinearGradient>

      <View style={styles.content}>
        <View style={styles.quickActionsContainer}>
          <Text style={styles.sectionTitle}>Acciones Rápidas</Text>
          <View style={styles.quickActionsGrid}>
            {quickActions.map((action) => (
              <TouchableOpacity
                key={action.id}
                style={styles.quickActionCard}
                onPress={() => navigation.navigate(action.screen)}
              >
                <View
                  style={[
                    styles.quickActionIcon,
                    { backgroundColor: `${action.color}20` },
                  ]}
                >
                  <Ionicons
                    name={action.icon}
                    size={32}
                    color={action.color}
                  />
                </View>
                <Text style={styles.quickActionText}>{action.title}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.infoSection}>
          <Text style={styles.sectionTitle}>¿Cómo funciona?</Text>
          <View style={styles.infoCard}>
            <Ionicons name="camera-outline" size={24} color="#6366f1" />
            <View style={styles.infoTextContainer}>
              <Text style={styles.infoTitle}>1. Toma una foto o video</Text>
              <Text style={styles.infoDescription}>
                Captura una imagen de tu piel usando la cámara o selecciona una
                de tu galería
              </Text>
            </View>
          </View>

          <View style={styles.infoCard}>
            <Ionicons name="analytics-outline" size={24} color="#6366f1" />
            <View style={styles.infoTextContainer}>
              <Text style={styles.infoTitle}>2. Análisis automático</Text>
              <Text style={styles.infoDescription}>
                Nuestra IA analiza tu piel y detecta condiciones, textura,
                hidratación y más
              </Text>
            </View>
          </View>

          <View style={styles.infoCard}>
            <Ionicons name="bulb-outline" size={24} color="#6366f1" />
            <View style={styles.infoTextContainer}>
              <Text style={styles.infoTitle}>3. Obtén recomendaciones</Text>
              <Text style={styles.infoDescription}>
                Recibe recomendaciones personalizadas de productos y rutinas de
                cuidado
              </Text>
            </View>
          </View>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    paddingTop: 60,
    paddingBottom: 30,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
  },
  headerTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#fff',
    opacity: 0.9,
  },
  content: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
  },
  quickActionsContainer: {
    marginBottom: 30,
  },
  quickActionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  quickActionCard: {
    width: '48%',
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  quickActionIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  quickActionText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    textAlign: 'center',
  },
  infoSection: {
    marginTop: 10,
  },
  infoCard: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  infoTextContainer: {
    flex: 1,
    marginLeft: 12,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  infoDescription: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
  },
});

export default HomeScreen;

