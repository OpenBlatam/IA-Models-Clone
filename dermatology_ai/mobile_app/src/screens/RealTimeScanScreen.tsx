import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Animated,
} from 'react-native';
import { Camera } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../types';
import { useRealTimeScan } from '../hooks/useRealTimeScan';
import { getScoreColor, getScoreLabel } from '../utils/helpers';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorView from '../components/ErrorView';

type RealTimeScanScreenNavigationProp = StackNavigationProp<RootStackParamList>;

const RealTimeScanScreen: React.FC = () => {
  const navigation = useNavigation<RealTimeScanScreenNavigationProp>();
  const {
    hasPermission,
    isScanning,
    lastAnalysis,
    isAnalyzing,
    cameraRef,
    startScanning,
    stopScanning,
    captureFullAnalysis,
    requestPermissions,
  } = useRealTimeScan();

  const pulseAnim = React.useRef(new Animated.Value(1)).current;

  React.useEffect(() => {
    if (isScanning) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.2,
            duration: 1000,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [isScanning]);

  const handleFullAnalysis = async () => {
    const uri = await captureFullAnalysis();
    if (uri) {
      navigation.navigate('Analysis', { imageUri: uri });
    }
  };

  if (hasPermission === null) {
    return <LoadingSpinner message="Solicitando permisos..." />;
  }

  if (hasPermission === false) {
    return (
      <ErrorView
        message="Necesitamos acceso a tu cámara para el escaneo en tiempo real"
        onRetry={requestPermissions}
        retryText="Solicitar Permiso"
      />
    );
  }

  const overallScore = lastAnalysis?.quality_scores?.overall_score || 0;
  const scoreColor = getScoreColor(overallScore);
  const scoreLabel = getScoreLabel(overallScore);

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={styles.camera}
        type={Camera.Constants.Type.front}
        ratio="16:9"
      >
        <View style={styles.overlay}>
          {/* Scanning indicator */}
          {isScanning && (
            <Animated.View
              style={[
                styles.scanningIndicator,
                { transform: [{ scale: pulseAnim }] },
              ]}
            >
              <View style={styles.scanningDot} />
              <Text style={styles.scanningText}>Escaneando...</Text>
            </Animated.View>
          )}

          {/* Last analysis results overlay */}
          {lastAnalysis && lastAnalysis.quality_scores && (
            <View style={styles.resultsOverlay}>
              <View style={[styles.resultCard, { borderColor: scoreColor }]}>
                <Text style={styles.resultTitle}>Puntuación General</Text>
                <Text style={[styles.resultScore, { color: scoreColor }]}>
                  {Math.round(overallScore)}
                </Text>
                <Text style={[styles.resultLabel, { color: scoreColor }]}>
                  {scoreLabel}
                </Text>
              </View>
              {lastAnalysis.skin_type && (
                <View style={styles.resultCard}>
                  <Text style={styles.resultTitle}>Tipo de Piel</Text>
                  <Text style={styles.resultText}>
                    {lastAnalysis.skin_type}
                  </Text>
                </View>
              )}
              {lastAnalysis.conditions && lastAnalysis.conditions.length > 0 && (
                <View style={[styles.resultCard, styles.warningCard]}>
                  <Ionicons name="warning" size={20} color="#f59e0b" />
                  <Text style={styles.warningText}>
                    {lastAnalysis.conditions.length} condición(es)
                  </Text>
                </View>
              )}
            </View>
          )}

          {/* Controls */}
          <View style={styles.controls}>
            <TouchableOpacity
              style={[
                styles.controlButton,
                isScanning && styles.controlButtonActive,
              ]}
              onPress={isScanning ? stopScanning : startScanning}
              activeOpacity={0.8}
            >
              <View
                style={[
                  styles.playButton,
                  isScanning && styles.playButtonActive,
                ]}
              >
                <Ionicons
                  name={isScanning ? 'stop-circle' : 'play-circle'}
                  size={48}
                  color="#fff"
                />
              </View>
              <Text style={styles.controlText}>
                {isScanning ? 'Detener Escaneo' : 'Iniciar Escaneo'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.fullAnalysisButton}
              onPress={handleFullAnalysis}
              disabled={isAnalyzing}
              activeOpacity={0.8}
            >
              <Ionicons name="camera" size={32} color="#fff" />
              <Text style={styles.fullAnalysisText}>Análisis Completo</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Camera>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'transparent',
    justifyContent: 'space-between',
  },
  scanningIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(99, 102, 241, 0.9)',
    padding: 12,
    marginTop: 60,
    marginHorizontal: 20,
    borderRadius: 12,
  },
  scanningDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#fff',
    marginRight: 8,
  },
  scanningText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  resultsOverlay: {
    position: 'absolute',
    top: 100,
    right: 20,
    width: 160,
  },
  resultCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
    borderWidth: 2,
    borderColor: '#e5e7eb',
  },
  resultTitle: {
    color: '#6b7280',
    fontSize: 11,
    fontWeight: '600',
    marginBottom: 4,
    textTransform: 'uppercase',
  },
  resultScore: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  resultLabel: {
    fontSize: 11,
    fontWeight: '600',
  },
  resultText: {
    color: '#1f2937',
    fontSize: 14,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  warningCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(245, 158, 11, 0.1)',
    borderColor: '#f59e0b',
  },
  warningText: {
    color: '#92400e',
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 6,
  },
  controls: {
    padding: 30,
    paddingBottom: 50,
    alignItems: 'center',
  },
  controlButton: {
    alignItems: 'center',
    marginBottom: 20,
  },
  controlButtonActive: {
    opacity: 0.9,
  },
  playButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(99, 102, 241, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#6366f1',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  playButtonActive: {
    backgroundColor: 'rgba(239, 68, 68, 0.9)',
    shadowColor: '#ef4444',
  },
  controlText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 12,
    textAlign: 'center',
  },
  fullAnalysisButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(99, 102, 241, 0.9)',
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderRadius: 12,
    shadowColor: '#6366f1',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  fullAnalysisText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
});

export default RealTimeScanScreen;

