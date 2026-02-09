import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute } from '@react-navigation/native';
import { StackNavigationProp, RouteProp } from '@react-navigation/stack';
import { RootStackParamList, HistoryItem, AnalysisResult } from '../types';
import { useSelector } from 'react-redux';
import { RootState } from '../types';
import ComparisonView from '../components/ComparisonView';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorView from '../components/ErrorView';
import ApiService from '../services/apiService';
import { formatDate } from '../utils/helpers';
import { metrics } from '../utils/metrics';

type ComparisonScreenRouteProp = RouteProp<RootStackParamList, 'Comparison'>;
type ComparisonScreenNavigationProp = StackNavigationProp<RootStackParamList>;

const ComparisonScreen: React.FC = () => {
  const navigation = useNavigation<ComparisonScreenNavigationProp>();
  const route = useRoute<ComparisonScreenRouteProp>();
  const { history } = useSelector((state: RootState) => state.history);

  const [analysis1, setAnalysis1] = useState<AnalysisResult | null>(null);
  const [analysis2, setAnalysis2] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // If route params provide analyses, use them
    if (route.params?.analysis1 && route.params?.analysis2) {
      setAnalysis1(route.params.analysis1);
      setAnalysis2(route.params.analysis2);
    } else if (route.params?.recordId1 && route.params?.recordId2) {
      loadComparison(route.params.recordId1, route.params.recordId2);
    } else if (history.length >= 2) {
      // Use last two analyses
      const lastTwo = history.slice(0, 2);
      setAnalysis1(lastTwo[0].analysis || lastTwo[0]);
      setAnalysis2(lastTwo[1].analysis || lastTwo[1]);
    }
  }, []);

  const loadComparison = async (id1: string, id2: string) => {
    try {
      setIsLoading(true);
      setError(null);
      const result = await ApiService.compareHistory(id1, id2);
      const comparisonData = result.data || result;
      if (comparisonData.analysis1 && comparisonData.analysis2) {
        setAnalysis1(comparisonData.analysis1);
        setAnalysis2(comparisonData.analysis2);
        metrics.trackComparisonPerformed();
      }
    } catch (err: any) {
      setError(err.message || 'Error al cargar la comparación');
    } finally {
      setIsLoading(false);
    }
  };

  const selectFromHistory = () => {
    navigation.navigate('History');
  };

  if (isLoading) {
    return <LoadingSpinner message="Cargando comparación..." />;
  }

  if (error) {
    return (
      <ErrorView
        message={error}
        onRetry={() => {
          if (route.params?.recordId1 && route.params?.recordId2) {
            loadComparison(route.params.recordId1, route.params.recordId2);
          }
        }}
      />
    );
  }

  if (!analysis1 || !analysis2) {
    return (
      <View style={styles.emptyContainer}>
        <Ionicons name="git-compare" size={64} color="#9ca3af" />
        <Text style={styles.emptyText}>Selecciona análisis para comparar</Text>
        <Text style={styles.emptySubtext}>
          Necesitas al menos 2 análisis para comparar
        </Text>
        <TouchableOpacity
          style={styles.selectButton}
          onPress={selectFromHistory}
        >
          <Text style={styles.selectButtonText}>Seleccionar del Historial</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const label1 = route.params?.label1 || formatDate(analysis1.timestamp || new Date().toISOString());
  const label2 = route.params?.label2 || formatDate(analysis2.timestamp || new Date().toISOString());

  return (
    <View style={styles.container}>
      <ComparisonView
        analysis1={analysis1}
        analysis2={analysis2}
        label1={label1}
        label2={label2}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1f2937',
    marginTop: 16,
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 24,
  },
  selectButton: {
    backgroundColor: '#6366f1',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  selectButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default ComparisonScreen;

