/**
 * Manual Generator Form
 * =====================
 * Form component for generating manuals
 */

import { useState } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { useRouter } from 'expo-router';
import { useMutation } from '@tanstack/react-query';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';
import { manualService } from '@/services/api/manual-service';
import { CATEGORIES } from '@/constants/categories';

interface ManualGeneratorFormProps {
  initialMode?: string;
  initialCategory?: string;
}

export function ManualGeneratorForm({ initialMode = 'text', initialCategory }: ManualGeneratorFormProps) {
  const router = useRouter();
  const { state } = useApp();
  const colors = state.colors;

  const [mode, setMode] = useState<'text' | 'image' | 'gallery'>(initialMode as any);
  const [problemDescription, setProblemDescription] = useState('');
  const [selectedCategory, setSelectedCategory] = useState(initialCategory || 'general');
  const [selectedImages, setSelectedImages] = useState<string[]>([]);

  const generateMutation = useMutation({
    mutationFn: async () => {
      if (mode === 'text') {
        return manualService.generateFromText({
          problem_description: problemDescription,
          category: selectedCategory,
          include_safety: true,
          include_tools: true,
          include_materials: true,
        });
      } else if (mode === 'image' && selectedImages.length > 0) {
        if (selectedImages.length === 1) {
          return manualService.generateFromImage(
            selectedImages[0],
            problemDescription,
            selectedCategory
          );
        } else {
          return manualService.generateFromMultipleImages(
            selectedImages,
            problemDescription,
            selectedCategory
          );
        }
      } else {
        throw new Error('Por favor completa todos los campos requeridos');
      }
    },
    onSuccess: (data) => {
      if (data.success && data.manual) {
        // Navigate to manual detail
        router.push(`/manual/${Date.now()}` as any);
      } else {
        Alert.alert('Error', data.error || 'No se pudo generar el manual');
      }
    },
    onError: (error: Error) => {
      Alert.alert('Error', error.message || 'Ocurrió un error al generar el manual');
    },
  });

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permisos', 'Se necesita acceso a la galería');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: mode === 'gallery',
      quality: 0.8,
    });

    if (!result.canceled) {
      const uris = result.assets.map((asset) => asset.uri);
      if (mode === 'gallery') {
        setSelectedImages([...selectedImages, ...uris].slice(0, 5));
      } else {
        setSelectedImages(uris.slice(0, 1));
      }
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permisos', 'Se necesita acceso a la cámara');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      quality: 0.8,
    });

    if (!result.canceled) {
      setSelectedImages([result.assets[0].uri]);
    }
  };

  const handleGenerate = () => {
    if (mode === 'text' && !problemDescription.trim()) {
      Alert.alert('Error', 'Por favor describe el problema');
      return;
    }

    if ((mode === 'image' || mode === 'gallery') && selectedImages.length === 0) {
      Alert.alert('Error', 'Por favor selecciona al menos una imagen');
      return;
    }

    generateMutation.mutate();
  };

  return (
    <View style={styles.container}>
      {/* Mode Selector */}
      <View style={styles.modeSelector}>
        <TouchableOpacity
          style={[
            styles.modeButton,
            mode === 'text' && { backgroundColor: colors.tint },
            { borderColor: colors.border },
          ]}
          onPress={() => setMode('text')}
        >
          <Ionicons
            name="text"
            size={20}
            color={mode === 'text' ? '#FFFFFF' : colors.text}
          />
          <Text
            style={[
              styles.modeButtonText,
              { color: mode === 'text' ? '#FFFFFF' : colors.text },
            ]}
          >
            Texto
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.modeButton,
            mode === 'image' && { backgroundColor: colors.tint },
            { borderColor: colors.border },
          ]}
          onPress={() => setMode('image')}
        >
          <Ionicons
            name="camera"
            size={20}
            color={mode === 'image' ? '#FFFFFF' : colors.text}
          />
          <Text
            style={[
              styles.modeButtonText,
              { color: mode === 'image' ? '#FFFFFF' : colors.text },
            ]}
          >
            Foto
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.modeButton,
            mode === 'gallery' && { backgroundColor: colors.tint },
            { borderColor: colors.border },
          ]}
          onPress={() => setMode('gallery')}
        >
          <Ionicons
            name="images"
            size={20}
            color={mode === 'gallery' ? '#FFFFFF' : colors.text}
          />
          <Text
            style={[
              styles.modeButtonText,
              { color: mode === 'gallery' ? '#FFFFFF' : colors.text },
            ]}
          >
            Galería
          </Text>
        </TouchableOpacity>
      </View>

      {/* Problem Description */}
      <View style={[styles.inputContainer, { backgroundColor: colors.card }]}>
        <Text style={[styles.label, { color: colors.text }]}>Descripción del Problema</Text>
        <TextInput
          style={[styles.textInput, { color: colors.text, borderColor: colors.border }]}
          placeholder="Describe el problema que necesitas resolver..."
          placeholderTextColor={colors.textSecondary}
          multiline
          numberOfLines={6}
          value={problemDescription}
          onChangeText={setProblemDescription}
          textAlignVertical="top"
        />
      </View>

      {/* Image Selection */}
      {(mode === 'image' || mode === 'gallery') && (
        <View style={[styles.inputContainer, { backgroundColor: colors.card }]}>
          <Text style={[styles.label, { color: colors.text }]}>
            {mode === 'gallery' ? 'Imágenes (máx. 5)' : 'Imagen'}
          </Text>
          <View style={styles.imageButtons}>
            <TouchableOpacity
              style={[styles.imageButton, { backgroundColor: colors.tint }]}
              onPress={takePhoto}
            >
              <Ionicons name="camera" size={20} color="#FFFFFF" />
              <Text style={styles.imageButtonText}>Tomar Foto</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.imageButton, { backgroundColor: colors.secondary }]}
              onPress={pickImage}
            >
              <Ionicons name="images" size={20} color="#FFFFFF" />
              <Text style={styles.imageButtonText}>Galería</Text>
            </TouchableOpacity>
          </View>
          {selectedImages.length > 0 && (
            <Text style={[styles.imageCount, { color: colors.textSecondary }]}>
              {selectedImages.length} imagen{selectedImages.length > 1 ? 'es' : ''} seleccionada
              {selectedImages.length > 1 ? 's' : ''}
            </Text>
          )}
        </View>
      )}

      {/* Category Selector */}
      <View style={[styles.inputContainer, { backgroundColor: colors.card }]}>
        <Text style={[styles.label, { color: colors.text }]}>Categoría</Text>
        <View style={styles.categoryGrid}>
          {Object.values(CATEGORIES).map((category) => (
            <TouchableOpacity
              key={category.id}
              style={[
                styles.categoryButton,
                selectedCategory === category.id && { backgroundColor: colors.tint },
                { borderColor: colors.border },
              ]}
              onPress={() => setSelectedCategory(category.id)}
            >
              <Ionicons
                name={category.icon as any}
                size={20}
                color={selectedCategory === category.id ? '#FFFFFF' : category.color}
              />
              <Text
                style={[
                  styles.categoryButtonText,
                  {
                    color: selectedCategory === category.id ? '#FFFFFF' : colors.text,
                  },
                ]}
              >
                {category.displayName}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Generate Button */}
      <TouchableOpacity
        style={[styles.generateButton, { backgroundColor: colors.tint }]}
        onPress={handleGenerate}
        disabled={generateMutation.isPending}
      >
        {generateMutation.isPending ? (
          <ActivityIndicator size="small" color="#FFFFFF" />
        ) : (
          <>
            <Ionicons name="sparkles" size={20} color="#FFFFFF" />
            <Text style={styles.generateButtonText}>Generar Manual</Text>
          </>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: 20,
  },
  modeSelector: {
    flexDirection: 'row',
    gap: 12,
  },
  modeButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    gap: 6,
  },
  modeButtonText: {
    fontSize: 14,
    fontWeight: '500',
  },
  inputContainer: {
    padding: 16,
    borderRadius: 12,
    gap: 12,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
  },
  textInput: {
    minHeight: 120,
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    fontSize: 16,
  },
  imageButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  imageButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
    borderRadius: 8,
    gap: 6,
  },
  imageButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '500',
  },
  imageCount: {
    fontSize: 12,
    textAlign: 'center',
  },
  categoryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  categoryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    gap: 6,
  },
  categoryButtonText: {
    fontSize: 12,
    fontWeight: '500',
  },
  generateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    gap: 8,
  },
  generateButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});

