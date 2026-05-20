import { useState, useRef, useEffect, useCallback } from 'react';

interface UseVideoPlayerOptions {
  videoUrl: string;
  autoPlay?: boolean;
  volume?: number;
  onTimeUpdate?: (currentTime: number) => void;
  onEnded?: () => void;
  onProgress?: (progress: number) => void;
  onError?: (error: string) => void;
  onVolumeChange?: (volume: number) => void;
}

interface VideoPlayerState {
  isPlaying: boolean;
  isMuted: boolean;
  currentVolume: number;
  currentTime: number;
  duration: number;
  isFullscreen: boolean;
  showControls: boolean;
  isLoading: boolean;
  error: string | null;
  playbackRate: number;
  isDragging: boolean;
  previewTime: number | null;
}

export function useVideoPlayer({
  videoUrl,
  autoPlay = false,
  volume = 1,
  onTimeUpdate,
  onEnded,
  onProgress,
  onError,
  onVolumeChange,
}: UseVideoPlayerOptions) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const controlsTimeoutRef = useRef<NodeJS.Timeout>();
  const playPromiseRef = useRef<Promise<void> | null>(null);
  const isPlayRequestedRef = useRef(false);
  const progressBarRef = useRef<HTMLDivElement>(null);

  const [state, setState] = useState<VideoPlayerState>({
    isPlaying: false,
    isMuted: false,
    currentVolume: volume,
    currentTime: 0,
    duration: 0,
    isFullscreen: false,
    showControls: true,
    isLoading: true,
    error: null,
    playbackRate: 1,
    isDragging: false,
    previewTime: null,
  });

  const updateState = useCallback((updates: Partial<VideoPlayerState>) => {
    setState(prev => ({ ...prev, ...updates }));
  }, []);

  const handleLoadStart = useCallback(() => {
    updateState({ isLoading: true });
  }, [updateState]);

  const handleLoadedData = useCallback(() => {
    updateState({ isLoading: false });
  }, [updateState]);

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    
    const currentTime = video.currentTime;
    updateState({ currentTime });
    onTimeUpdate?.(currentTime);
    
    if (state.duration > 0) {
      const progress = (currentTime / state.duration) * 100;
      onProgress?.(progress);
    }
  }, [onTimeUpdate, onProgress, state.duration, updateState]);

  const handleDurationChange = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    updateState({ duration: video.duration });
  }, [updateState]);

  const handleEnded = useCallback(() => {
    updateState({ isPlaying: false });
    isPlayRequestedRef.current = false;
    onEnded?.();
  }, [onEnded, updateState]);

  const handleError = useCallback((e: Event) => {
    const videoElement = e.target as HTMLVideoElement;
    const error = videoElement.error;
    
    let errorMessage = "Error al cargar el video. Por favor, intenta de nuevo.";
    
    if (error) {
      switch (error.code) {
        case error.MEDIA_ERR_ABORTED:
          errorMessage = "La carga del video fue cancelada.";
          break;
        case error.MEDIA_ERR_NETWORK:
          errorMessage = "Error de red al cargar el video.";
          break;
        case error.MEDIA_ERR_DECODE:
          errorMessage = "Error al decodificar el video.";
          break;
        case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
          errorMessage = "Formato de video no soportado.";
          break;
      }
    }
    
    updateState({ error: errorMessage, isLoading: false });
    onError?.(errorMessage);
  }, [onError, updateState]);

  const handleCanPlay = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return;
    
    updateState({ isLoading: false });
    if (autoPlay && !isPlayRequestedRef.current) {
      try {
        isPlayRequestedRef.current = true;
        await video.play();
        updateState({ isPlaying: true });
      } catch (error) {
        updateState({ 
          error: "Error al iniciar la reproducción. Por favor, intenta de nuevo.",
          isPlaying: false 
        });
      } finally {
        isPlayRequestedRef.current = false;
      }
    }
  }, [autoPlay, updateState]);

  const handleWaiting = useCallback(() => {
    updateState({ isLoading: true });
  }, [updateState]);

  const handlePlaying = useCallback(() => {
    updateState({ isLoading: false });
  }, [updateState]);

  const togglePlayPause = useCallback(async () => {
    const video = videoRef.current;
    if (!video || isPlayRequestedRef.current) return;

    try {
      isPlayRequestedRef.current = true;
      if (state.isPlaying) {
        video.pause();
        updateState({ isPlaying: false });
      } else {
        await video.play();
        updateState({ isPlaying: true });
      }
    } catch (error) {
      updateState({ isPlaying: false });
    } finally {
      isPlayRequestedRef.current = false;
    }
  }, [state.isPlaying, updateState]);

  const toggleMute = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    const newMuted = !state.isMuted;
    video.muted = newMuted;
    updateState({ isMuted: newMuted });
  }, [state.isMuted, updateState]);

  const changeVolume = useCallback((newVolume: number) => {
    const video = videoRef.current;
    if (!video) return;

    const clampedVolume = Math.max(0, Math.min(1, newVolume));
    video.volume = clampedVolume;
    updateState({ currentVolume: clampedVolume, isMuted: clampedVolume === 0 });
    onVolumeChange?.(clampedVolume);
  }, [onVolumeChange, updateState]);

  const seekTo = useCallback((time: number) => {
    const video = videoRef.current;
    if (!video) return;

    video.currentTime = Math.max(0, Math.min(state.duration, time));
  }, [state.duration]);

  const changePlaybackRate = useCallback((rate: number) => {
    const video = videoRef.current;
    if (!video) return;

    video.playbackRate = rate;
    updateState({ playbackRate: rate });
  }, [updateState]);

  const toggleFullscreen = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return;

    try {
      if (!state.isFullscreen) {
        if (video.requestFullscreen) {
          await video.requestFullscreen();
        }
      } else {
        if (document.exitFullscreen) {
          await document.exitFullscreen();
        }
      }
    } catch (error) {
      console.error("Fullscreen error:", error);
    }
  }, [state.isFullscreen]);

  useEffect(() => {
    if (videoRef.current) {
      updateState({ isLoading: true, error: null });
      videoRef.current.load();
    }
  }, [videoUrl, updateState]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    video.addEventListener("loadstart", handleLoadStart);
    video.addEventListener("loadeddata", handleLoadedData);
    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("durationchange", handleDurationChange);
    video.addEventListener("ended", handleEnded);
    video.addEventListener("error", handleError);
    video.addEventListener("canplay", handleCanPlay);
    video.addEventListener("waiting", handleWaiting);
    video.addEventListener("playing", handlePlaying);

    return () => {
      video.removeEventListener("loadstart", handleLoadStart);
      video.removeEventListener("loadeddata", handleLoadedData);
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("durationchange", handleDurationChange);
      video.removeEventListener("ended", handleEnded);
      video.removeEventListener("error", handleError);
      video.removeEventListener("canplay", handleCanPlay);
      video.removeEventListener("waiting", handleWaiting);
      video.removeEventListener("playing", handlePlaying);
    };
  }, [
    handleLoadStart,
    handleLoadedData,
    handleTimeUpdate,
    handleDurationChange,
    handleEnded,
    handleError,
    handleCanPlay,
    handleWaiting,
    handlePlaying,
  ]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" || e.key === " ") {
        e.preventDefault();
        togglePlayPause();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [togglePlayPause]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      updateState({ isFullscreen: !!document.fullscreenElement });
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, [updateState]);

  return {
    videoRef,
    progressBarRef,
    state,
    actions: {
      togglePlayPause,
      toggleMute,
      changeVolume,
      seekTo,
      changePlaybackRate,
      toggleFullscreen,
      updateState,
    },
  };
}
