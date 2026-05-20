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
import { useCamera } from '../hooks/useCamera';

type CameraScreenNavigationProp = StackNavigationProp<RootStackParamList, 'MainTabs'>;

const CameraScreen: React.FC = () => {
  const navigation = useNavigation<CameraScreenNavigationProp>();
  const {
    hasPermission,
    cameraType,
    isRecording,
    isProcessing,
    cameraRef,
    toggleCameraType,
    takePicture,
    pickImage,
    recordVideo,
    requestPermissions,
  } = useCamera();

  const handleTakePicture = async () => {
    const uri = await takePicture();
    if (uri) {
      navigation.navigate('Analysis', { imageUri: uri });
    }
  };

  const handlePickImage = async () => {
    const uri = await pickImage();
    if (uri) {
      navigation.navigate('Analysis', { imageUri: uri });
    }
  };

  const handleRecordVideo = async () => {
    const uri = await recordVideo();
    if (uri) {
      navigation.navigate('Analysis', { videoUri: uri });
    }
  };

  if (hasPermission === null) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#6366f1" />
        <Text style={styles.loadingText}>Solicitando permisos...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.permissionContainer}>
        <Ionicons name="camera-outline" size={64} color="#6366f1" />
        <Text style={styles.permissionTitle}>Permiso de Cámara Requerido</Text>
        <Text style={styles.permissionText}>
          Necesitamos acceso a tu cámara para analizar tu piel
        </Text>
        <TouchableOpacity style={styles.permissionButton} onPress={requestPermissions}>
          <Text style={styles.permissionButtonText}>Solicitar Permiso</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={styles.camera}
        type={cameraType}
        ratio="16:9"
      >
        <View style={styles.overlay}>
          {/* Top Controls */}
          <View style={styles.topControls}>
            <TouchableOpacity
              style={styles.controlButton}
              onPress={toggleCameraType}
            >
              <View style={styles.iconContainer}>
                <Ionicons name="camera-reverse" size={28} color="#fff" />
              </View>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.controlButton}
              onPress={handlePickImage}
            >
              <View style={styles.iconContainer}>
                <Ionicons name="images" size={28} color="#fff" />
              </View>
            </TouchableOpacity>
          </View>

          {/* Bottom Controls */}
          <View style={styles.bottomControls}>
            <TouchableOpacity
              style={styles.controlButton}
              onPress={handlePickImage}
            >
              <View style={styles.iconButton}>
                <Ionicons name="image-outline" size={32} color="#fff" />
              </View>
              <Text style={styles.controlText}>Galería</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.captureButton, isProcessing && styles.captureButtonDisabled]}
              onPress={handleTakePicture}
              disabled={isProcessing}
              activeOpacity={0.8}
            >
              {isProcessing ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <View style={styles.captureButtonInner} />
              )}
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.controlButton}
              onPress={handleRecordVideo}
            >
              <View style={[styles.iconButton, isRecording && styles.recordingButton]}>
                <Ionicons
                  name={isRecording ? 'stop-circle' : 'videocam-outline'}
                  size={32}
                  color={isRecording ? '#ef4444' : '#fff'}
                />
              </View>
              <Text style={[styles.controlText, isRecording && styles.recordingText]}>
                {isRecording ? 'Grabando' : 'Video'}
              </Text>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  loadingText: {
    color: '#fff',
    marginTop: 16,
    fontSize: 16,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
    padding: 40,
  },
  permissionTitle: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 24,
    marginBottom: 12,
    textAlign: 'center',
  },
  permissionText: {
    color: '#ccc',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 24,
  },
  permissionButton: {
    backgroundColor: '#6366f1',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
  },
  permissionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'transparent',
    justifyContent: 'space-between',
  },
  topControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
    paddingTop: 60,
  },
  bottomControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    padding: 30,
    paddingBottom: 50,
  },
  controlButton: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  iconButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  controlText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
    marginTop: 4,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#6366f1',
    shadowColor: '#6366f1',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  captureButtonDisabled: {
    opacity: 0.6,
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#6366f1',
  },
  recordingButton: {
    backgroundColor: 'rgba(239, 68, 68, 0.3)',
    borderWidth: 2,
    borderColor: '#ef4444',
  },
  recordingText: {
    color: '#ef4444',
    fontWeight: '600',
  },
});

export default CameraScreen;

