import { useState, useCallback } from 'react';
import { Alert } from 'react-native';
import { useDispatch } from 'react-redux';
import ApiService from '../services/apiService';
import { AnalysisResult } from '../types';

interface UseAnalysisReturn {
  analysis: AnalysisResult | null;
  isAnalyzing: boolean;
  error: string | null;
  analyzeImage: (imageUri: string, options?: { enhance?: boolean; bodyArea?: string }) => Promise<void>;
  analyzeVideo: (videoUri: string) => Promise<void>;
  getRecommendations: (analysisId?: string, imageUri?: string) => Promise<any>;
  clearAnalysis: () => void;
}

export const useAnalysis = (): UseAnalysisReturn => {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dispatch = useDispatch();

  const analyzeImage = useCallback(async (
    imageUri: string,
    options: { enhance?: boolean; bodyArea?: string } = {}
  ) => {
    try {
      setIsAnalyzing(true);
      setError(null);

      const result = await ApiService.analyzeImage(imageUri, {
        enhance: options.enhance ?? true,
        bodyArea: options.bodyArea,
      });

      if (result && (result.success || result.data)) {
        const analysisData = result.data || result.analysis || result;
        setAnalysis(analysisData);
        dispatch({
          type: 'ANALYSIS_SUCCESS',
          payload: analysisData,
        });
      } else {
        throw new Error('Análisis fallido');
      }
    } catch (err: any) {
      const errorMessage = err.message || 'Error al analizar la imagen';
      setError(errorMessage);
      Alert.alert('Error', 'No se pudo analizar la imagen. Intenta de nuevo.');
    } finally {
      setIsAnalyzing(false);
    }
  }, [dispatch]);

  const analyzeVideo = useCallback(async (videoUri: string) => {
    try {
      setIsAnalyzing(true);
      setError(null);

      const result = await ApiService.analyzeVideo(videoUri);

      if (result && (result.success || result.data)) {
        const analysisData = result.data || result.analysis || result;
        setAnalysis(analysisData);
        dispatch({
          type: 'ANALYSIS_SUCCESS',
          payload: analysisData,
        });
      } else {
        throw new Error('Análisis fallido');
      }
    } catch (err: any) {
      const errorMessage = err.message || 'Error al analizar el video';
      setError(errorMessage);
      Alert.alert('Error', 'No se pudo analizar el video. Intenta de nuevo.');
    } finally {
      setIsAnalyzing(false);
    }
  }, [dispatch]);

  const getRecommendations = useCallback(async (
    analysisId?: string,
    imageUri?: string
  ) => {
    try {
      setIsAnalyzing(true);
      const result = await ApiService.getRecommendations(
        { imageUri, analysisId },
        { includeRoutine: true }
      );
      return result.data || result.recommendations || result;
    } catch (err: any) {
      Alert.alert('Error', 'No se pudieron obtener las recomendaciones');
      throw err;
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const clearAnalysis = useCallback(() => {
    setAnalysis(null);
    setError(null);
    dispatch({ type: 'CLEAR_ANALYSIS' });
  }, [dispatch]);

  return {
    analysis,
    isAnalyzing,
    error,
    analyzeImage,
    analyzeVideo,
    getRecommendations,
    clearAnalysis,
  };
};

