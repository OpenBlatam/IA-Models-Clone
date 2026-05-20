import React, { useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  ActivityIndicator,
  TouchableOpacity,
  Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRoute, useNavigation } from '@react-navigation/native';
import { StackNavigationProp, RouteProp } from '@react-navigation/stack';
import { RootStackParamList, AnalysisResult } from '../types';
import { useAnalysis } from '../hooks/useAnalysis';
import ScoreCard from '../components/ScoreCard';
import RadarChart from '../components/RadarChart';
import { metrics } from '../utils/metrics';

type AnalysisScreenRouteProp = RouteProp<RootStackParamList, 'Analysis'>;
type AnalysisScreenNavigationProp = StackNavigationProp<RootStackParamList>;

const AnalysisScreen: React.FC = () => {
  const route = useRoute<AnalysisScreenRouteProp>();
  const navigation = useNavigation<AnalysisScreenNavigationProp>();
  const { imageUri, videoUri, analysis: initialAnalysis } = route.params || {};

  const { analysis, isAnalyzing, error, analyzeImage, analyzeVideo, getRecommendations } = useAnalysis();
  const fadeAnim = React.useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (imageUri || videoUri) {
      if (initialAnalysis) {
        // If analysis is already provided, use it
        metrics.trackAnalysis(imageUri || videoUri || '', 'provided');
        return;
      }
      if (imageUri) {
        metrics.trackAnalysis(imageUri, 'image');
        analyzeImage(imageUri, { enhance: true });
      } else if (videoUri) {
        metrics.trackAnalysis(videoUri, 'video');
        analyzeVideo(videoUri);
      }
    }
  }, [imageUri, videoUri]);

  useEffect(() => {
    if (analysis) {
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }).start();
    }
  }, [analysis]);

  const handleGetRecommendations = async () => {
    if (!analysis) return;

    try {
      const recommendations = await getRecommendations(analysis.id, imageUri);
      navigation.navigate('Recommendations', {
        recommendations,
        analysis,
      });
    } catch (err) {
      // Error already handled in hook
    }
  };

  const renderScores = () => {
    const currentAnalysis = analysis || initialAnalysis;
    if (!currentAnalysis?.quality_scores) return null;

    const scores = currentAnalysis.quality_scores;
    const scoreItems = [
      { label: 'General', value: scores.overall_score, color: '#6366f1' },
      { label: 'Textura', value: scores.texture_score, color: '#8b5cf6' },
      { label: 'Hidratación', value: scores.hydration_score, color: '#ec4899' },
      { label: 'Elasticidad', value: scores.elasticity_score, color: '#f59e0b' },
      { label: 'Pigmentación', value: scores.pigmentation_score, color: '#10b981' },
      { label: 'Poros', value: scores.pore_size_score, color: '#3b82f6' },
      { label: 'Arrugas', value: scores.wrinkles_score, color: '#ef4444' },
      { label: 'Enrojecimiento', value: scores.redness_score, color: '#f97316' },
    ];

    return (
      <Animated.View style={[styles.scoresContainer, { opacity: fadeAnim }]}>
        <Text style={styles.sectionTitle}>Puntuaciones de Calidad</Text>
        <View style={styles.scoresGrid}>
          {scoreItems.map((item, index) => (
            <ScoreCard
              key={index}
              label={item.label}
              value={item.value}
              color={item.color}
            />
          ))}
        </View>
      </Animated.View>
    );
  };

  const renderConditions = () => {
    const currentAnalysis = analysis || initialAnalysis;
    if (!currentAnalysis?.conditions || currentAnalysis.conditions.length === 0) {
      return null;
    }

    return (
      <Animated.View style={[styles.conditionsContainer, { opacity: fadeAnim }]}>
        <Text style={styles.sectionTitle}>Condiciones Detectadas</Text>
        {currentAnalysis.conditions.map((condition, index) => (
          <View key={index} style={styles.conditionCard}>
            <Ionicons name="warning" size={24} color="#f59e0b" />
            <View style={styles.conditionText}>
              <Text style={styles.conditionName}>{condition.name}</Text>
              {condition.severity && (
                <Text style={styles.conditionSeverity}>
                  Severidad: {condition.severity}
                </Text>
              )}
            </View>
          </View>
        ))}
      </Animated.View>
    );
  };

  const currentAnalysis = analysis || initialAnalysis;

  if (isAnalyzing && !currentAnalysis) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#6366f1" />
        <Text style={styles.loadingText}>Analizando tu piel...</Text>
        <Text style={styles.loadingSubtext}>
          Esto puede tomar unos segundos
        </Text>
      </View>
    );
  }

  if (error && !currentAnalysis) {
    return (
      <View style={styles.errorContainer}>
        <Ionicons name="alert-circle" size={64} color="#ef4444" />
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity
          style={styles.retryButton}
          onPress={() => {
            if (imageUri) {
              analyzeImage(imageUri, { enhance: true });
            } else if (videoUri) {
              analyzeVideo(videoUri);
            }
          }}
        >
          <Text style={styles.retryButtonText}>Intentar de Nuevo</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {imageUri && (
        <Image source={{ uri: imageUri }} style={styles.image} />
      )}

      {currentAnalysis && (
        <>
          <LinearGradient
            colors={['#6366f1', '#8b5cf6']}
            style={styles.header}
          >
            <Ionicons name="checkmark-circle" size={32} color="#fff" />
            <Text style={styles.headerTitle}>Análisis Completado</Text>
            <Text style={styles.headerSubtitle}>
              Tipo de piel: {currentAnalysis.skin_type || 'No detectado'}
            </Text>
          </LinearGradient>

          <View style={styles.content}>
            {renderScores()}
            {renderConditions()}

            {currentAnalysis.recommendations_priority && (
              <Animated.View style={[styles.priorityContainer, { opacity: fadeAnim }]}>
                <Text style={styles.sectionTitle}>Prioridades</Text>
                <View style={styles.priorityTags}>
                  {currentAnalysis.recommendations_priority.map((priority, index) => (
                    <View key={index} style={styles.priorityTag}>
                      <Text style={styles.priorityText}>{priority}</Text>
                    </View>
                  ))}
                </View>
              </Animated.View>
            )}

            <TouchableOpacity
              style={styles.recommendationsButton}
              onPress={handleGetRecommendations}
              disabled={isAnalyzing}
              activeOpacity={0.8}
            >
              <LinearGradient
                colors={['#6366f1', '#8b5cf6']}
                style={styles.buttonGradient}
              >
                <Ionicons name="sparkles" size={24} color="#fff" />
                <Text style={styles.recommendationsButtonText}>
                  Ver Recomendaciones
                </Text>
              </LinearGradient>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.reportButton}
              onPress={() => navigation.navigate('Report', { analysis: currentAnalysis })}
              activeOpacity={0.8}
            >
              <Ionicons name="document-text" size={24} color="#6366f1" />
              <Text style={styles.reportButtonText}>Ver Reporte Completo</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  loadingText: {
    marginTop: 20,
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
  },
  loadingSubtext: {
    marginTop: 8,
    fontSize: 14,
    color: '#6b7280',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  errorText: {
    marginTop: 20,
    fontSize: 16,
    color: '#ef4444',
    textAlign: 'center',
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: '#6366f1',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  image: {
    width: '100%',
    height: 300,
    resizeMode: 'cover',
  },
  header: {
    padding: 20,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 12,
    marginBottom: 4,
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
  scoresContainer: {
    marginBottom: 24,
  },
  scoresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  conditionsContainer: {
    marginBottom: 24,
  },
  conditionCard: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  conditionText: {
    marginLeft: 12,
    flex: 1,
  },
  conditionName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  conditionSeverity: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 4,
  },
  priorityContainer: {
    marginBottom: 24,
  },
  priorityTags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  priorityTag: {
    backgroundColor: '#fef3c7',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    marginBottom: 8,
  },
  priorityText: {
    fontSize: 14,
    color: '#92400e',
    fontWeight: '500',
  },
  recommendationsButton: {
    marginTop: 8,
    marginBottom: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  buttonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  recommendationsButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },
  reportButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#6366f1',
  },
  reportButtonText: {
    color: '#6366f1',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
});

export default AnalysisScreen;

