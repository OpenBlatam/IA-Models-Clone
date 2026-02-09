import { useMutation, useQueryClient } from '@tanstack/react-query';
import { videoService } from '@/services/video-service';
import { Alert } from 'react-native';

export function useShareVideo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      videoId,
      options,
    }: {
      videoId: string;
      options: {
        shared_with_email?: string;
        shared_with_id?: string;
        permission?: string;
        is_public?: boolean;
        expires_at?: string;
      };
    }) => videoService.shareVideo(videoId, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['video'] });
      Alert.alert('Success', 'Video shared successfully');
    },
    onError: (error: any) => {
      Alert.alert('Error', error.detail || 'Failed to share video');
    },
  });
}

export function useAddWatermark() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      videoId,
      options,
    }: {
      videoId: string;
      options: {
        watermark_text?: string;
        watermark_image?: string;
        position?: string;
        opacity?: number;
        size?: number;
      };
    }) => videoService.addWatermark(videoId, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['video'] });
      Alert.alert('Success', 'Watermark added successfully');
    },
    onError: (error: any) => {
      Alert.alert('Error', error.detail || 'Failed to add watermark');
    },
  });
}

export function useTranscribeVideo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      videoId,
      language,
    }: {
      videoId: string;
      language?: string;
    }) => videoService.transcribeVideo(videoId, language),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['video'] });
    },
    onError: (error: any) => {
      Alert.alert('Error', error.detail || 'Failed to transcribe video');
    },
  });
}

export function useAddKenBurnsEffect() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      videoId,
      options,
    }: {
      videoId: string;
      options: {
        zoom?: number;
        pan_x?: number;
        pan_y?: number;
      };
    }) => videoService.addKenBurnsEffect(videoId, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['video'] });
      Alert.alert('Success', 'Ken Burns effect added successfully');
    },
    onError: (error: any) => {
      Alert.alert('Error', error.detail || 'Failed to add effect');
    },
  });
}

export function useExportVideo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      videoId,
      platforms,
    }: {
      videoId: string;
      platforms: string[];
    }) => videoService.exportVideo(videoId, platforms),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['video'] });
      Alert.alert('Success', 'Export started successfully');
    },
    onError: (error: any) => {
      Alert.alert('Error', error.detail || 'Failed to export video');
    },
  });
}

export function useRegisterWebhook() {
  return useMutation({
    mutationFn: ({
      videoId,
      webhookUrl,
    }: {
      videoId: string;
      webhookUrl: string;
    }) => videoService.registerWebhook(videoId, webhookUrl),
    onSuccess: () => {
      Alert.alert('Success', 'Webhook registered successfully');
    },
    onError: (error: any) => {
      Alert.alert('Error', error.detail || 'Failed to register webhook');
    },
  });
}


