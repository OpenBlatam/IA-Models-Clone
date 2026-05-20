import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useGenerateVideo } from '@/hooks/use-video-generation';
import { useRecommendations } from '@/hooks/use-analytics';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Loading } from '@/components/ui/loading';
import type { VideoGenerationRequest, VideoStyle, AudioVoice, SubtitleStyle } from '@/types/api';
import { VideoStyle as VideoStyleMap, AudioVoice as AudioVoiceMap, SubtitleStyle as SubtitleStyleMap } from '@/types/api';

export default function VideoGenerationScreen() {
  const router = useRouter();
  const generateVideo = useGenerateVideo();
  const [script, setScript] = useState('');
  const [videoStyle, setVideoStyle] = useState<VideoStyle>(VideoStyleMap.REALISTIC);
  const [audioVoice, setAudioVoice] = useState<AudioVoice>(AudioVoiceMap.NEUTRAL);
  const [subtitleStyle, setSubtitleStyle] = useState<SubtitleStyle>(SubtitleStyleMap.MODERN);
  const [subtitleEnabled, setSubtitleEnabled] = useState(true);

  const { data: recommendations } = useRecommendations(script, 'es', undefined, 'general', script.length > 10);

  const handleGenerate = async () => {
    if (script.length < 10) {
      Alert.alert('Error', 'Script must be at least 10 characters long');
      return;
    }

    try {
      const request: VideoGenerationRequest = {
        script: {
          text: script,
          language: 'es',
        },
        video_config: {
          style: videoStyle,
          resolution: '1920x1080',
          fps: 30,
        },
        audio_config: {
          voice: audioVoice,
          speed: 1.0,
        },
        subtitle_config: {
          enabled: subtitleEnabled,
          style: subtitleStyle,
        },
        output_format: 'mp4',
        output_quality: 'high',
      };

      const response = await generateVideo.mutateAsync(request);
      router.push(`/video-detail?videoId=${response.video_id}`);
    } catch (error: any) {
      Alert.alert('Error', error.detail || 'Failed to generate video');
    }
  };

  if (generateVideo.isPending) {
    return <Loading message="Generating video..." />;
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
        >
          <Text style={styles.title}>Generate Video</Text>

          <Input
            label="Script"
            value={script}
            onChangeText={setScript}
            placeholder="Enter your script here..."
            multiline
            numberOfLines={8}
            style={styles.scriptInput}
          />

          {recommendations && (
            <View style={styles.recommendations}>
              <Text style={styles.recommendationsTitle}>AI Recommendations</Text>
              <Text style={styles.recommendationsText}>{recommendations.reasoning}</Text>
            </View>
          )}

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Video Style</Text>
            <View style={styles.optionsRow}>
              {Object.values(VideoStyleMap).map((style) => (
                <Button
                  key={style}
                  title={style}
                  onPress={() => setVideoStyle(style)}
                  variant={videoStyle === style ? 'primary' : 'outline'}
                  size="small"
                  style={styles.optionButton}
                />
              ))}
            </View>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Audio Voice</Text>
            <View style={styles.optionsRow}>
              {Object.values(AudioVoiceMap).map((voice) => (
                <Button
                  key={voice}
                  title={voice}
                  onPress={() => setAudioVoice(voice)}
                  variant={audioVoice === voice ? 'primary' : 'outline'}
                  size="small"
                  style={styles.optionButton}
                />
              ))}
            </View>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Subtitles</Text>
            <Button
              title={subtitleEnabled ? 'Enabled' : 'Disabled'}
              onPress={() => setSubtitleEnabled(!subtitleEnabled)}
              variant={subtitleEnabled ? 'primary' : 'outline'}
              size="small"
              style={styles.toggleButton}
            />
            {subtitleEnabled && (
              <View style={styles.optionsRow}>
                {Object.values(SubtitleStyleMap).map((style) => (
                  <Button
                    key={style}
                    title={style}
                    onPress={() => setSubtitleStyle(style)}
                    variant={subtitleStyle === style ? 'primary' : 'outline'}
                    size="small"
                    style={styles.optionButton}
                  />
                ))}
              </View>
            )}
          </View>

          <Button
            title="Generate Video"
            onPress={handleGenerate}
            variant="primary"
            size="large"
            style={styles.generateButton}
            disabled={script.length < 10}
          />
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  keyboardView: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 24,
    color: '#000000',
  },
  scriptInput: {
    minHeight: 120,
    textAlignVertical: 'top',
    marginBottom: 16,
  },
  recommendations: {
    backgroundColor: '#E3F2FD',
    borderRadius: 8,
    padding: 16,
    marginBottom: 24,
  },
  recommendationsTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: '#000000',
  },
  recommendationsText: {
    fontSize: 14,
    color: '#666666',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    color: '#000000',
  },
  optionsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  optionButton: {
    marginRight: 8,
    marginBottom: 8,
  },
  toggleButton: {
    marginBottom: 12,
    alignSelf: 'flex-start',
  },
  generateButton: {
    marginTop: 8,
  },
});


