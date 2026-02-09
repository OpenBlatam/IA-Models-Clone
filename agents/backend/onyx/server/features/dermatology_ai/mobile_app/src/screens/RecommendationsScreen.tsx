import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRoute, useNavigation } from '@react-navigation/native';
import { RouteProp, StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList, Recommendations, Product, Routine } from '../types';
import ApiService from '../services/apiService';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorView from '../components/ErrorView';
import { metrics } from '../utils/metrics';

type RecommendationsScreenRouteProp = RouteProp<RootStackParamList, 'Recommendations'>;
type RecommendationsScreenNavigationProp = StackNavigationProp<RootStackParamList>;

const RecommendationsScreen: React.FC = () => {
  const route = useRoute<RecommendationsScreenRouteProp>();
  const navigation = useNavigation<RecommendationsScreenNavigationProp>();
  const { recommendations: initialRecommendations, analysis } = route.params || {};

  const [recommendations, setRecommendations] = useState<Recommendations | null>(
    initialRecommendations || null
  );
  const [isLoading, setIsLoading] = useState(!initialRecommendations);
  const [error, setError] = useState<string | null>(null);
  const fadeAnim = React.useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (!initialRecommendations && analysis) {
      loadRecommendations();
    } else if (initialRecommendations) {
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }).start();
    }
  }, []);

  const loadRecommendations = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const result = await ApiService.getRecommendations(
        { analysisId: analysis?.id },
        { includeRoutine: true }
      );
      const recommendationsData = result.data || result.recommendations || result;
      setRecommendations(recommendationsData);
      metrics.trackRecommendationViewed();
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }).start();
    } catch (err: any) {
      setError(err.message || 'Error al cargar recomendaciones');
    } finally {
      setIsLoading(false);
    }
  };

  const renderProduct = (product: Product, index: number) => (
    <Animated.View
      key={product.id || index}
      style={[styles.productCard, { opacity: fadeAnim }]}
    >
      <View style={styles.productHeader}>
        <View style={styles.productIcon}>
          <Ionicons name="cube" size={24} color="#6366f1" />
        </View>
        <View style={styles.productInfo}>
          <Text style={styles.productName}>{product.name || 'Producto'}</Text>
          {product.category && (
            <Text style={styles.productCategory}>{product.category}</Text>
          )}
        </View>
      </View>
      {product.description && (
        <Text style={styles.productDescription}>{product.description}</Text>
      )}
      {product.benefits && product.benefits.length > 0 && (
        <View style={styles.benefitsContainer}>
          {product.benefits.map((benefit, idx) => (
            <View key={idx} style={styles.benefitTag}>
              <Ionicons name="checkmark-circle" size={16} color="#10b981" />
              <Text style={styles.benefitText}>{benefit}</Text>
            </View>
          ))}
        </View>
      )}
      {product.price && (
        <View style={styles.priceContainer}>
          <Text style={styles.productPrice}>${product.price}</Text>
          <TouchableOpacity style={styles.buyButton}>
            <Text style={styles.buyButtonText}>Comprar</Text>
          </TouchableOpacity>
        </View>
      )}
    </Animated.View>
  );

  const renderRoutine = (routine: Routine) => {
    if (!routine || !routine.steps) return null;

    return (
      <Animated.View style={[styles.routineContainer, { opacity: fadeAnim }]}>
        <View style={styles.routineHeader}>
          <Ionicons name="calendar" size={24} color="#6366f1" />
          <Text style={styles.sectionTitle}>Rutina Recomendada</Text>
        </View>
        {routine.duration && (
          <Text style={styles.routineDuration}>Duración: {routine.duration}</Text>
        )}
        {routine.frequency && (
          <Text style={styles.routineFrequency}>Frecuencia: {routine.frequency}</Text>
        )}
        {routine.steps.map((step, index) => (
          <View key={index} style={styles.routineStep}>
            <View style={styles.stepNumber}>
              <Text style={styles.stepNumberText}>{index + 1}</Text>
            </View>
            <View style={styles.stepContent}>
              <Text style={styles.stepTitle}>{step.name || step}</Text>
              {typeof step !== 'string' && step.description && (
                <Text style={styles.stepDescription}>{step.description}</Text>
              )}
              {typeof step !== 'string' && step.time && (
                <View style={styles.stepTimeContainer}>
                  <Ionicons name="time-outline" size={14} color="#9ca3af" />
                  <Text style={styles.stepTime}>{step.time}</Text>
                </View>
              )}
            </View>
          </View>
        ))}
      </Animated.View>
    );
  };

  if (isLoading) {
    return <LoadingSpinner message="Generando recomendaciones personalizadas..." />;
  }

  if (error && !recommendations) {
    return (
      <ErrorView
        message={error}
        onRetry={loadRecommendations}
        retryText="Cargar Recomendaciones"
      />
    );
  }

  return (
    <ScrollView style={styles.container}>
      <LinearGradient
        colors={['#6366f1', '#8b5cf6']}
        style={styles.header}
      >
        <Ionicons name="sparkles" size={32} color="#fff" />
        <Text style={styles.headerTitle}>Recomendaciones</Text>
        <Text style={styles.headerSubtitle}>
          Personalizadas para tu tipo de piel
        </Text>
      </LinearGradient>

      <View style={styles.content}>
        {recommendations?.products && recommendations.products.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Productos Recomendados</Text>
            {recommendations.products.map((product, index) =>
              renderProduct(product, index)
            )}
          </View>
        )}

        {recommendations?.routine && renderRoutine(recommendations.routine)}

        {recommendations?.tips && recommendations.tips.length > 0 && (
          <Animated.View style={[styles.section, { opacity: fadeAnim }]}>
            <Text style={styles.sectionTitle}>Consejos</Text>
            {recommendations.tips.map((tip, index) => (
              <View key={index} style={styles.tipCard}>
                <Ionicons name="bulb" size={20} color="#f59e0b" />
                <Text style={styles.tipText}>{tip}</Text>
              </View>
            ))}
          </Animated.View>
        )}

        {recommendations?.warnings && recommendations.warnings.length > 0 && (
          <Animated.View style={[styles.section, { opacity: fadeAnim }]}>
            <Text style={styles.sectionTitle}>Advertencias</Text>
            {recommendations.warnings.map((warning, index) => (
              <View key={index} style={styles.warningCard}>
                <Ionicons name="warning" size={20} color="#ef4444" />
                <Text style={styles.warningText}>{warning}</Text>
              </View>
            ))}
          </Animated.View>
        )}

        <TouchableOpacity
          style={styles.compareButton}
          onPress={() => navigation.navigate('History')}
          activeOpacity={0.8}
        >
          <Ionicons name="git-compare" size={24} color="#6366f1" />
          <Text style={styles.compareButtonText}>Comparar Productos</Text>
        </TouchableOpacity>
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
    padding: 30,
    paddingTop: 60,
    alignItems: 'center',
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 12,
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
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
  },
  productCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  productHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  productIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#f3f4f6',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  productInfo: {
    flex: 1,
  },
  productName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  productCategory: {
    fontSize: 14,
    color: '#6b7280',
  },
  productDescription: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
    marginBottom: 12,
  },
  benefitsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 12,
  },
  benefitTag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0fdf4',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
    marginBottom: 8,
  },
  benefitText: {
    fontSize: 12,
    color: '#166534',
    marginLeft: 4,
  },
  priceContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
  },
  productPrice: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#6366f1',
  },
  buyButton: {
    backgroundColor: '#6366f1',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  buyButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  routineContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
  },
  routineHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  routineDuration: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 4,
  },
  routineFrequency: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 16,
  },
  routineStep: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  stepNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#6366f1',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  stepNumberText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  stepDescription: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
  },
  stepTimeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  stepTime: {
    fontSize: 12,
    color: '#9ca3af',
    marginLeft: 4,
  },
  tipCard: {
    flexDirection: 'row',
    backgroundColor: '#fffbeb',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    alignItems: 'flex-start',
  },
  tipText: {
    flex: 1,
    fontSize: 14,
    color: '#92400e',
    marginLeft: 12,
    lineHeight: 20,
  },
  warningCard: {
    flexDirection: 'row',
    backgroundColor: '#fef2f2',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    alignItems: 'flex-start',
  },
  warningText: {
    flex: 1,
    fontSize: 14,
    color: '#991b1b',
    marginLeft: 12,
    lineHeight: 20,
  },
  compareButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#6366f1',
    marginTop: 8,
  },
  compareButtonText: {
    color: '#6366f1',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
});

export default RecommendationsScreen;

