import { useState, useRef, useEffect, useCallback } from 'react';
import { Camera } from 'expo-camera';
import { AnalysisResult } from '../types';
import ApiService from '../services/apiService';

interface UseRealTimeScanReturn {
  hasPermission: boolean | null;
  isScanning: boolean;
  lastAnalysis: AnalysisResult | null;
  isAnalyzing: boolean;
  cameraRef: React.RefObject<Camera>;
  startScanning: () => void;
  stopScanning: () => void;
  captureFullAnalysis: () => Promise<string | null>;
  requestPermissions: () => Promise<void>;
}

const SCAN_INTERVAL = 3000; // 3 seconds

export const useRealTimeScan = (): UseRealTimeScanReturn => {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [lastAnalysis, setLastAnalysis] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const cameraRef = useRef<Camera>(null);
  const scanIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    requestPermissions();
  }, []);

  useEffect(() => {
    return () => {
      if (scanIntervalRef.current) {
        clearInterval(scanIntervalRef.current);
      }
    };
  }, []);

  const requestPermissions = async () => {
    try {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    } catch (error) {
      console.error('Error requesting camera permissions:', error);
      setHasPermission(false);
    }
  };

  const captureAndAnalyze = useCallback(async () => {
    if (!cameraRef.current || isAnalyzing) return;

    try {
      setIsAnalyzing(true);
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.5,
        base64: false,
        skipProcessing: true,
      });

      const result = await ApiService.analyzeImage(photo.uri, {
        enhance: false,
      });

      if (result && (result.success || result.data)) {
        const analysisData = result.data || result.analysis || result;
        setLastAnalysis(analysisData);
      }
    } catch (error) {
      console.error('Error in real-time scan:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }, [isAnalyzing]);

  const startScanning = useCallback(() => {
    setIsScanning(true);
    scanIntervalRef.current = setInterval(() => {
      captureAndAnalyze();
    }, SCAN_INTERVAL);
  }, [captureAndAnalyze]);

  const stopScanning = useCallback(() => {
    setIsScanning(false);
    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
      scanIntervalRef.current = null;
    }
  }, []);

  const captureFullAnalysis = useCallback(async (): Promise<string | null> => {
    if (!cameraRef.current) return null;

    try {
      stopScanning();
      setIsAnalyzing(true);

      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
      });

      return photo.uri;
    } catch (error) {
      console.error('Error capturing full analysis:', error);
      return null;
    } finally {
      setIsAnalyzing(false);
    }
  }, [stopScanning]);

  return {
    hasPermission,
    isScanning,
    lastAnalysis,
    isAnalyzing,
    cameraRef,
    startScanning,
    stopScanning,
    captureFullAnalysis,
    requestPermissions,
  };
};

