import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  RefreshControl,
  TextInput,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { useSelector, useDispatch } from 'react-redux';
import { RootStackParamList, RootState, HistoryItem } from '../types';
import ApiService from '../services/apiService';
import { formatDate, getScoreColor, getScoreLabel } from '../utils/helpers';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorView from '../components/ErrorView';
import EmptyState from '../components/EmptyState';
import { useDebounce } from '../hooks/useDebounce';

type HistoryScreenNavigationProp = StackNavigationProp<RootStackParamList>;

const HistoryScreen: React.FC = () => {
  const navigation = useNavigation<HistoryScreenNavigationProp>();
  const dispatch = useDispatch();
  const { history, isLoading, error } = useSelector((state: RootState) => state.history);
  const { userId } = useSelector((state: RootState) => state.user);

  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredHistory, setFilteredHistory] = useState<HistoryItem[]>([]);
  const debouncedSearchQuery = useDebounce(searchQuery, 300);

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    if (debouncedSearchQuery.trim() === '') {
      setFilteredHistory(history);
    } else {
      const filtered = history.filter((item) => {
        const query = debouncedSearchQuery.toLowerCase();
        return (
          item.skin_type?.toLowerCase().includes(query) ||
          item.body_area?.toLowerCase().includes(query) ||
          formatDate(item.timestamp).toLowerCase().includes(query)
        );
      });
      setFilteredHistory(filtered);
    }
  }, [debouncedSearchQuery, history]);

  const loadHistory = async () => {
    try {
      dispatch({ type: 'HISTORY_LOAD_START' });
      const result = await ApiService.getHistory(userId || 'default_user');
      const historyData = result.data?.history || result.history || result || [];
      dispatch({
        type: 'HISTORY_LOAD_SUCCESS',
        payload: Array.isArray(historyData) ? historyData : [],
      });
    } catch (err: any) {
      console.error('Error loading history:', err);
      dispatch({
        type: 'HISTORY_LOAD_FAILURE',
        payload: err.message || 'Error al cargar el historial',
      });
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadHistory();
    setRefreshing(false);
  };

  const handleCompare = (item1: HistoryItem, item2: HistoryItem) => {
    navigation.navigate('Comparison', {
      analysis1: item1.analysis || item1,
      analysis2: item2.analysis || item2,
      label1: formatDate(item1.timestamp),
      label2: formatDate(item2.timestamp),
    });
  };

  const renderHistoryItem = (item: HistoryItem, index: number) => {
    const date = formatDate(item.timestamp);
    const overallScore =
      item.quality_scores?.overall_score ||
      item.analysis?.quality_scores?.overall_score ||
      0;
    const scoreColor = getScoreColor(overallScore);
    const scoreLabel = getScoreLabel(overallScore);

    return (
      <TouchableOpacity
        key={item.id || index}
        style={styles.historyItem}
        onPress={() => {
          navigation.navigate('Analysis', {
            analysis: item.analysis || item,
            fromHistory: true,
          });
        }}
        activeOpacity={0.7}
      >
        {item.image_uri && (
          <Image
            source={{ uri: item.image_uri }}
            style={styles.historyImage}
          />
        )}
        <View style={styles.historyContent}>
          <View style={styles.historyHeader}>
            <Text style={styles.historyDate}>{date}</Text>
            <View style={[styles.scoreBadge, { backgroundColor: scoreColor }]}>
              <Text style={styles.scoreText}>{Math.round(overallScore)}</Text>
            </View>
          </View>
          {item.skin_type && (
            <View style={styles.skinTypeContainer}>
              <Ionicons name="body" size={16} color="#6366f1" />
              <Text style={styles.skinType}>Tipo: {item.skin_type}</Text>
            </View>
          )}
          {item.body_area && (
            <View style={styles.bodyAreaContainer}>
              <Ionicons name="location" size={16} color="#8b5cf6" />
              <Text style={styles.bodyArea}>Área: {item.body_area}</Text>
            </View>
          )}
          {item.conditions && item.conditions.length > 0 && (
            <View style={styles.conditionsBadge}>
              <Ionicons name="warning" size={16} color="#f59e0b" />
              <Text style={styles.conditionsText}>
                {item.conditions.length} condición(es) detectada(s)
              </Text>
            </View>
          )}
          <View style={styles.scoreLabelContainer}>
            <Text style={[styles.scoreLabel, { color: scoreColor }]}>
              {scoreLabel}
            </Text>
          </View>
        </View>
        <View style={styles.itemActions}>
          {index < filteredHistory.length - 1 && (
            <TouchableOpacity
              style={styles.compareButton}
              onPress={(e) => {
                e.stopPropagation();
                handleCompare(item, filteredHistory[index + 1]);
              }}
            >
              <Ionicons name="git-compare" size={20} color="#6366f1" />
            </TouchableOpacity>
          )}
          <Ionicons name="chevron-forward" size={24} color="#9ca3af" />
        </View>
      </TouchableOpacity>
    );
  };

  if (isLoading && history.length === 0) {
    return <LoadingSpinner message="Cargando historial..." />;
  }

  if (error && history.length === 0) {
    return (
      <ErrorView
        message={error}
        onRetry={loadHistory}
        retryText="Cargar Historial"
      />
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Historial de Análisis</Text>
        <Text style={styles.headerSubtitle}>
          {history.length} análisis realizados
        </Text>
      </View>

      {history.length > 0 && (
        <View style={styles.searchContainer}>
          <Ionicons name="search" size={20} color="#9ca3af" style={styles.searchIcon} />
          <TextInput
            style={styles.searchInput}
            placeholder="Buscar en historial..."
            placeholderTextColor="#9ca3af"
            value={searchQuery}
            onChangeText={setSearchQuery}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity
              onPress={() => setSearchQuery('')}
              style={styles.clearButton}
            >
              <Ionicons name="close-circle" size={20} color="#9ca3af" />
            </TouchableOpacity>
          )}
        </View>
      )}

      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {history.length === 0 ? (
          <EmptyState
            icon="time-outline"
            title="No hay análisis aún"
            message="Realiza tu primer análisis para ver tu historial aquí"
            actionLabel="Realizar Análisis"
            onAction={() => navigation.navigate('MainTabs')}
          />
        ) : filteredHistory.length === 0 ? (
          <EmptyState
            icon="search-outline"
            title="No se encontraron resultados"
            message="Intenta con otros términos de búsqueda"
          />
        ) : (
          <View style={styles.content}>
            {filteredHistory.map((item, index) => renderHistoryItem(item, index))}
          </View>
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#fff',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#6b7280',
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    margin: 16,
    marginBottom: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  searchIcon: {
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
    color: '#1f2937',
  },
  clearButton: {
    marginLeft: 8,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 16,
  },
  historyItem: {
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
  historyImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginRight: 12,
  },
  historyContent: {
    flex: 1,
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  historyDate: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  scoreBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  scoreText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  skinTypeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  skinType: {
    fontSize: 12,
    color: '#6b7280',
    marginLeft: 6,
    textTransform: 'capitalize',
  },
  bodyAreaContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  bodyArea: {
    fontSize: 12,
    color: '#6b7280',
    marginLeft: 6,
  },
  conditionsBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  conditionsText: {
    fontSize: 12,
    color: '#f59e0b',
    marginLeft: 4,
  },
  scoreLabelContainer: {
    marginTop: 8,
  },
  scoreLabel: {
    fontSize: 12,
    fontWeight: '600',
  },
  itemActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  compareButton: {
    marginRight: 12,
    padding: 8,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
    minHeight: 400,
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
  emptyButton: {
    backgroundColor: '#6366f1',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  emptyButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default HistoryScreen;

