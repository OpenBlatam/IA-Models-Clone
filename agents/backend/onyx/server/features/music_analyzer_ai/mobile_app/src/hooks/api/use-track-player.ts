import { useState, useCallback, useRef, useEffect } from 'react';
import { Audio } from 'expo-av';
import type { Track } from '../../types/api';

export interface PlaybackState {
  isPlaying: boolean;
  isPaused: boolean;
  isStopped: boolean;
  currentTime: number;
  duration: number;
  volume: number;
}

export function useTrackPlayer() {
  const [currentTrack, setCurrentTrack] = useState<Track | null>(null);
  const [playbackState, setPlaybackState] = useState<PlaybackState>({
    isPlaying: false,
    isPaused: false,
    isStopped: true,
    currentTime: 0,
    duration: 0,
    volume: 1.0,
  });
  const soundRef = useRef<Audio.Sound | null>(null);

  useEffect(() => {
    return () => {
      if (soundRef.current) {
        soundRef.current.unloadAsync();
      }
    };
  }, []);

  const loadTrack = useCallback(async (track: Track) => {
    if (!track.preview_url) {
      throw new Error('Track has no preview URL');
    }

    if (soundRef.current) {
      await soundRef.current.unloadAsync();
    }

    const { sound } = await Audio.Sound.createAsync(
      { uri: track.preview_url },
      { shouldPlay: false }
    );

    soundRef.current = sound;

    const status = await sound.getStatusAsync();
    if (status.isLoaded) {
      setCurrentTrack(track);
      setPlaybackState((prev) => ({
        ...prev,
        duration: status.durationMillis || 0,
        isStopped: true,
        isPlaying: false,
        isPaused: false,
      }));
    }
  }, []);

  const play = useCallback(async () => {
    if (!soundRef.current) return;

    await soundRef.current.playAsync();
    setPlaybackState((prev) => ({
      ...prev,
      isPlaying: true,
      isPaused: false,
      isStopped: false,
    }));
  }, []);

  const pause = useCallback(async () => {
    if (!soundRef.current) return;

    await soundRef.current.pauseAsync();
    setPlaybackState((prev) => ({
      ...prev,
      isPlaying: false,
      isPaused: true,
      isStopped: false,
    }));
  }, []);

  const stop = useCallback(async () => {
    if (!soundRef.current) return;

    await soundRef.current.stopAsync();
    setPlaybackState((prev) => ({
      ...prev,
      isPlaying: false,
      isPaused: false,
      isStopped: true,
      currentTime: 0,
    }));
  }, []);

  const setVolume = useCallback(async (volume: number) => {
    if (!soundRef.current) return;

    await soundRef.current.setVolumeAsync(Math.max(0, Math.min(1, volume)));
    setPlaybackState((prev) => ({ ...prev, volume }));
  }, []);

  const seekTo = useCallback(async (positionMillis: number) => {
    if (!soundRef.current) return;

    await soundRef.current.setPositionAsync(positionMillis);
    setPlaybackState((prev) => ({ ...prev, currentTime: positionMillis }));
  }, []);

  return {
    currentTrack,
    playbackState,
    loadTrack,
    play,
    pause,
    stop,
    setVolume,
    seekTo,
  };
}


