import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { 
  AccessibleText, 
  AccessibleButton, 
  AccessibleView, 
  useAccessibility, 
  useResponsiveAccessibility,
  accessibilityStyles 
} from '../../utils/accessibility';

// Types for the component
interface PostGenerationRequest {
  topic: string;
  keyPoints: string[];
  targetAudience: string;
  industry: string;
  tone: 'professional' | 'casual' | 'friendly';
  postType: 'announcement' | 'educational' | 'update' | 'insight';
  keywords?: string[];
  additionalContext?: string;
}

interface GeneratedPost {
  id: string;
  content: string;
  optimizationScore: number;
  suggestions: string[];
  generationTime: number;
}

// Main component with responsive design and accessibility
export function LinkedInPostGenerator() {
  const { width, height } = useWindowDimensions();
  const isTablet = width > 768;
  const isLandscape = width > height;
  const { getScaledFontSize, isScreenReaderActive } = useAccessibility();
  const { getResponsiveFontSize } = useResponsiveAccessibility();

  // State management
  const [isLoading, setIsLoading] = useState(false);
  const [generatedPost, setGeneratedPost] = useState<GeneratedPost | null>(null);
  const [formData, setFormData] = useState<PostGenerationRequest>({
    topic: '',
    keyPoints: [''],
    targetAudience: '',
    industry: '',
    tone: 'professional',
    postType: 'announcement',
    keywords: [],
    additionalContext: '',
  });

  // Pure function for form validation
  function validateForm(): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!formData.topic.trim()) errors.push('Topic is required');
    if (!formData.targetAudience.trim()) errors.push('Target audience is required');
    if (!formData.industry.trim()) errors.push('Industry is required');
    if (formData.keyPoints.length === 0 || !formData.keyPoints[0].trim()) {
      errors.push('At least one key point is required');
    }

    return { isValid: errors.length === 0, errors };
  }

  // Pure function for generating post
  const generatePost = useCallback(async () => {
    const validation = validateForm();
    if (!validation.isValid) {
      Alert.alert('Validation Error', validation.errors.join('\n'));
      return;
    }

    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockGeneratedPost: GeneratedPost = {
        id: `post_${Date.now()}`,
        content: `🚀 ${formData.topic}\n\n${formData.keyPoints.map(point => `• ${point}`).join('\n')}\n\n#${formData.industry} #Innovation #Growth`,
        optimizationScore: 0.85,
        suggestions: ['Add more hashtags', 'Include a call-to-action'],
        generationTime: 1.2,
      };

      setGeneratedPost(mockGeneratedPost);
    } catch (error) {
      Alert.alert('Error', 'Failed to generate post. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [formData]);

  // Pure function for updating key points
  function updateKeyPoint(index: number, value: string) {
    const newKeyPoints = [...formData.keyPoints];
    newKeyPoints[index] = value;
    setFormData(prev => ({ ...prev, keyPoints: newKeyPoints }));
  }

  // Pure function for adding key point
  function addKeyPoint() {
    setFormData(prev => ({
      ...prev,
      keyPoints: [...prev.keyPoints, ''],
    }));
  }

  // Pure function for removing key point
  function removeKeyPoint(index: number) {
    if (formData.keyPoints.length > 1) {
      const newKeyPoints = formData.keyPoints.filter((_, i) => i !== index);
      setFormData(prev => ({ ...prev, keyPoints: newKeyPoints }));
    }
  }

  // Responsive styles
  const styles = {
    container: {
      flex: 1,
      backgroundColor: '#f5f5f5',
    },
    safeArea: {
      flex: 1,
    },
    scrollView: {
      flex: 1,
      paddingHorizontal: isTablet ? 40 : 20,
    },
    header: {
      alignItems: 'center',
      paddingVertical: 20,
    },
    title: {
      fontSize: isTablet ? 32 : 24,
      fontWeight: 'bold',
      color: '#1a1a1a',
      textAlign: 'center',
    },
    subtitle: {
      fontSize: isTablet ? 18 : 16,
      color: '#666',
      textAlign: 'center',
      marginTop: 8,
    },
    formContainer: {
      backgroundColor: '#fff',
      borderRadius: 12,
      padding: isTablet ? 32 : 20,
      marginVertical: 20,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 8,
      elevation: 4,
    },
    inputGroup: {
      marginBottom: 20,
    },
    label: {
      fontSize: 16,
      fontWeight: '600',
      color: '#333',
      marginBottom: 8,
    },
    textInput: {
      borderWidth: 1,
      borderColor: '#ddd',
      borderRadius: 8,
      paddingHorizontal: 16,
      paddingVertical: 12,
      fontSize: 16,
      backgroundColor: '#fff',
    },
    textArea: {
      borderWidth: 1,
      borderColor: '#ddd',
      borderRadius: 8,
      paddingHorizontal: 16,
      paddingVertical: 12,
      fontSize: 16,
      backgroundColor: '#fff',
      minHeight: 100,
      textAlignVertical: 'top',
    },
    keyPointContainer: {
      flexDirection: 'row',
      alignItems: 'center',
      marginBottom: 12,
    },
    keyPointInput: {
      flex: 1,
      borderWidth: 1,
      borderColor: '#ddd',
      borderRadius: 8,
      paddingHorizontal: 16,
      paddingVertical: 12,
      fontSize: 16,
      backgroundColor: '#fff',
      marginRight: 8,
    },
    removeButton: {
      padding: 8,
      backgroundColor: '#ff4444',
      borderRadius: 6,
    },
    addButton: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
      paddingVertical: 12,
      paddingHorizontal: 16,
      backgroundColor: '#007AFF',
      borderRadius: 8,
      marginTop: 8,
    },
    addButtonText: {
      color: '#fff',
      fontSize: 16,
      fontWeight: '600',
      marginLeft: 8,
    },
    generateButton: {
      backgroundColor: '#007AFF',
      borderRadius: 12,
      paddingVertical: 16,
      paddingHorizontal: 32,
      alignItems: 'center',
      marginTop: 20,
    },
    generateButtonText: {
      color: '#fff',
      fontSize: 18,
      fontWeight: 'bold',
    },
    resultContainer: {
      backgroundColor: '#fff',
      borderRadius: 12,
      padding: isTablet ? 32 : 20,
      marginTop: 20,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 8,
      elevation: 4,
    },
    resultTitle: {
      fontSize: 20,
      fontWeight: 'bold',
      color: '#333',
      marginBottom: 16,
    },
    resultContent: {
      fontSize: 16,
      lineHeight: 24,
      color: '#333',
      marginBottom: 16,
    },
    metricsContainer: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginTop: 16,
      paddingTop: 16,
      borderTopWidth: 1,
      borderTopColor: '#eee',
    },
    metric: {
      alignItems: 'center',
    },
    metricValue: {
      fontSize: 18,
      fontWeight: 'bold',
      color: '#007AFF',
    },
    metricLabel: {
      fontSize: 12,
      color: '#666',
      marginTop: 4,
    },
    suggestionsContainer: {
      marginTop: 16,
    },
    suggestionItem: {
      flexDirection: 'row',
      alignItems: 'center',
      marginBottom: 8,
    },
    suggestionIcon: {
      marginRight: 8,
    },
    suggestionText: {
      fontSize: 14,
      color: '#666',
    },
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          style={styles.scrollView}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={{ paddingBottom: 40 }}
        >
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.title}>LinkedIn Post Generator</Text>
            <Text style={styles.subtitle}>
              Create engaging posts with AI-powered optimization
            </Text>
          </View>

          {/* Form */}
          <View style={styles.formContainer}>
            {/* Topic */}
            <View style={styles.inputGroup}>
              <Text style={styles.label}>Topic *</Text>
              <TextInput
                style={styles.textInput}
                value={formData.topic}
                onChangeText={text => setFormData(prev => ({ ...prev, topic: text }))}
                placeholder="Enter your post topic"
                multiline
              />
            </View>

            {/* Key Points */}
            <View style={styles.inputGroup}>
              <Text style={styles.label}>Key Points *</Text>
              {formData.keyPoints.map((point, index) => (
                <View key={index} style={styles.keyPointContainer}>
                  <TextInput
                    style={styles.keyPointInput}
                    value={point}
                    onChangeText={text => updateKeyPoint(index, text)}
                    placeholder={`Key point ${index + 1}`}
                  />
                  {formData.keyPoints.length > 1 && (
                    <TouchableOpacity
                      style={styles.removeButton}
                      onPress={() => removeKeyPoint(index)}
                    >
                      <Ionicons name="close" size={20} color="#fff" />
                    </TouchableOpacity>
                  )}
                </View>
              ))}
              <TouchableOpacity style={styles.addButton} onPress={addKeyPoint}>
                <Ionicons name="add" size={20} color="#fff" />
                <Text style={styles.addButtonText}>Add Key Point</Text>
              </TouchableOpacity>
            </View>

            {/* Target Audience */}
            <View style={styles.inputGroup}>
              <Text style={styles.label}>Target Audience *</Text>
              <TextInput
                style={styles.textInput}
                value={formData.targetAudience}
                onChangeText={text => setFormData(prev => ({ ...prev, targetAudience: text }))}
                placeholder="e.g., tech professionals, marketers"
              />
            </View>

            {/* Industry */}
            <View style={styles.inputGroup}>
              <Text style={styles.label}>Industry *</Text>
              <TextInput
                style={styles.textInput}
                value={formData.industry}
                onChangeText={text => setFormData(prev => ({ ...prev, industry: text }))}
                placeholder="e.g., technology, marketing"
              />
            </View>

            {/* Tone Selection */}
            <View style={styles.inputGroup}>
              <Text style={styles.label}>Tone</Text>
              <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8 }}>
                {(['professional', 'casual', 'friendly'] as const).map(tone => (
                  <TouchableOpacity
                    key={tone}
                    style={{
                      paddingHorizontal: 16,
                      paddingVertical: 8,
                      borderRadius: 20,
                      backgroundColor: formData.tone === tone ? '#007AFF' : '#f0f0f0',
                    }}
                    onPress={() => setFormData(prev => ({ ...prev, tone }))}
                  >
                    <Text
                      style={{
                        color: formData.tone === tone ? '#fff' : '#333',
                        fontSize: 14,
                        fontWeight: '500',
                      }}
                    >
                      {tone.charAt(0).toUpperCase() + tone.slice(1)}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Post Type Selection */}
            <View style={styles.inputGroup}>
              <Text style={styles.label}>Post Type</Text>
              <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8 }}>
                {(['announcement', 'educational', 'update', 'insight'] as const).map(type => (
                  <TouchableOpacity
                    key={type}
                    style={{
                      paddingHorizontal: 16,
                      paddingVertical: 8,
                      borderRadius: 20,
                      backgroundColor: formData.postType === type ? '#007AFF' : '#f0f0f0',
                    }}
                    onPress={() => setFormData(prev => ({ ...prev, postType: type }))}
                  >
                    <Text
                      style={{
                        color: formData.postType === type ? '#fff' : '#333',
                        fontSize: 14,
                        fontWeight: '500',
                      }}
                    >
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Additional Context */}
            <View style={styles.inputGroup}>
              <Text style={styles.label}>Additional Context</Text>
              <TextInput
                style={styles.textArea}
                value={formData.additionalContext}
                onChangeText={text => setFormData(prev => ({ ...prev, additionalContext: text }))}
                placeholder="Any additional context or requirements..."
                multiline
              />
            </View>

            {/* Generate Button */}
            <TouchableOpacity
              style={[styles.generateButton, isLoading && { opacity: 0.7 }]}
              onPress={generatePost}
              disabled={isLoading}
            >
              {isLoading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.generateButtonText}>Generate Post</Text>
              )}
            </TouchableOpacity>
          </View>

          {/* Generated Result */}
          {generatedPost && (
            <View style={styles.resultContainer}>
              <Text style={styles.resultTitle}>Generated Post</Text>
              <Text style={styles.resultContent}>{generatedPost.content}</Text>

              {/* Metrics */}
              <View style={styles.metricsContainer}>
                <View style={styles.metric}>
                  <Text style={styles.metricValue}>
                    {(generatedPost.optimizationScore * 100).toFixed(0)}%
                  </Text>
                  <Text style={styles.metricLabel}>Optimization Score</Text>
                </View>
                <View style={styles.metric}>
                  <Text style={styles.metricValue}>{generatedPost.generationTime}s</Text>
                  <Text style={styles.metricLabel}>Generation Time</Text>
                </View>
              </View>

              {/* Suggestions */}
              {generatedPost.suggestions.length > 0 && (
                <View style={styles.suggestionsContainer}>
                  <Text style={styles.label}>Optimization Suggestions</Text>
                  {generatedPost.suggestions.map((suggestion, index) => (
                    <View key={index} style={styles.suggestionItem}>
                      <Ionicons
                        name="bulb-outline"
                        size={16}
                        color="#007AFF"
                        style={styles.suggestionIcon}
                      />
                      <Text style={styles.suggestionText}>{suggestion}</Text>
                    </View>
                  ))}
                </View>
              )}
            </View>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

// Export the component
export default LinkedInPostGenerator; 