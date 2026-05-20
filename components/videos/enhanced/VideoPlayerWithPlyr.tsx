"use client";

import React, { useRef, useEffect, useState, memo } from "react";
import Plyr from "plyr-react";
import "plyr-react/plyr.css";
import { motion, AnimatePresence } from "framer-motion";
import { VideoPlayerProps } from "@/lib/types/video";
import { useVideoAnalytics } from "@/lib/hooks/useVideoAnalytics";
import { toast } from "react-hot-toast";

interface VideoPlayerWithPlyrProps extends VideoPlayerProps {
  videoId?: string;
  userId?: string;
  enableAnalytics?: boolean;
  plyrOptions?: any;
}

const VideoPlayerWithPlyr = memo(function VideoPlayerWithPlyr({
  videoUrl,
  videoId,
  userId,
  autoPlay = false,
  volume = 1,
  enableAnalytics = true,
  onTimeUpdate,
  onEnded,
  onProgress,
  onError,
  onVolumeChange,
  plyrOptions = {},
  className,
}: VideoPlayerWithPlyrProps) {
  const plyrRef = useRef<any>(null);
  const [isReady, setIsReady] = useState(false);
  
  const { trackWatchTime, trackVideoEvent } = useVideoAnalytics({
    videoId: videoId || '',
    userId,
  });

  const defaultPlyrOptions = {
    controls: [
      'play-large',
      'play',
      'progress',
      'current-time',
      'duration',
      'mute',
      'volume',
      'settings',
      'fullscreen'
    ],
    settings: ['quality', 'speed'],
    quality: {
      default: 720,
      options: [1080, 720, 480, 360]
    },
    speed: {
      selected: 1,
      options: [0.5, 0.75, 1, 1.25, 1.5, 2]
    },
    autoplay: autoPlay,
    volume: volume,
    clickToPlay: true,
    keyboard: { focused: true, global: false },
    tooltips: { controls: true, seek: true },
    captions: { active: false, update: false, language: 'auto' },
    fullscreen: { enabled: true, fallback: true, iosNative: false },
    storage: { enabled: true, key: 'plyr' },
    ...plyrOptions
  };

  useEffect(() => {
    if (plyrRef.current && plyrRef.current.plyr) {
      const player = plyrRef.current.plyr;
      
      const handleReady = () => {
        setIsReady(true);
        if (enableAnalytics && videoId) {
          trackVideoEvent('video_ready', { duration: player.duration });
        }
      };

      const handlePlay = () => {
        if (enableAnalytics && videoId) {
          trackVideoEvent('video_play', { currentTime: player.currentTime });
        }
      };

      const handlePause = () => {
        if (enableAnalytics && videoId) {
          trackVideoEvent('video_pause', { currentTime: player.currentTime });
        }
      };

      const handleTimeUpdate = () => {
        const currentTime = player.currentTime;
        onTimeUpdate?.(currentTime);
        
        if (enableAnalytics && videoId) {
          trackWatchTime(currentTime, !player.paused);
        }
      };

      const handleProgress = () => {
        if (player.duration > 0) {
          const progress = (player.currentTime / player.duration) * 100;
          onProgress?.(progress);
        }
      };

      const handleEnded = () => {
        onEnded?.();
        if (enableAnalytics && videoId) {
          trackVideoEvent('video_completed', { duration: player.duration });
        }
      };

      const handleError = (event: any) => {
        const errorMessage = "Error al reproducir el video";
        onError?.(errorMessage);
        toast.error(errorMessage);
        
        if (enableAnalytics && videoId) {
          trackVideoEvent('video_error', { error: event.detail });
        }
      };

      const handleVolumeChange = () => {
        const newVolume = player.volume;
        onVolumeChange?.(newVolume);
        
        if (enableAnalytics && videoId) {
          trackVideoEvent('volume_change', { volume: newVolume });
        }
      };

      const handleSeeked = () => {
        if (enableAnalytics && videoId) {
          trackVideoEvent('video_seek', { currentTime: player.currentTime });
        }
      };

      const handleRateChange = () => {
        if (enableAnalytics && videoId) {
          trackVideoEvent('playback_rate_change', { rate: player.speed });
        }
      };

      const handleQualityChange = () => {
        if (enableAnalytics && videoId) {
          trackVideoEvent('quality_change', { quality: player.quality });
        }
      };

      player.on('ready', handleReady);
      player.on('play', handlePlay);
      player.on('pause', handlePause);
      player.on('timeupdate', handleTimeUpdate);
      player.on('progress', handleProgress);
      player.on('ended', handleEnded);
      player.on('error', handleError);
      player.on('volumechange', handleVolumeChange);
      player.on('seeked', handleSeeked);
      player.on('ratechange', handleRateChange);
      player.on('qualitychange', handleQualityChange);

      return () => {
        player.off('ready', handleReady);
        player.off('play', handlePlay);
        player.off('pause', handlePause);
        player.off('timeupdate', handleTimeUpdate);
        player.off('progress', handleProgress);
        player.off('ended', handleEnded);
        player.off('error', handleError);
        player.off('volumechange', handleVolumeChange);
        player.off('seeked', handleSeeked);
        player.off('ratechange', handleRateChange);
        player.off('qualitychange', handleQualityChange);
      };
    }
  }, [isReady, enableAnalytics, videoId, onTimeUpdate, onProgress, onEnded, onError, onVolumeChange, trackWatchTime, trackVideoEvent]);

  return (
    <div className={`relative w-full ${className || ''}`}>
      <AnimatePresence>
        {!isReady && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/50 flex items-center justify-center z-10"
          >
            <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin" />
          </motion.div>
        )}
      </AnimatePresence>
      
      <Plyr
        ref={plyrRef}
        source={{
          type: 'video',
          sources: [
            {
              src: videoUrl,
              type: 'video/mp4',
            },
          ],
        }}
        options={defaultPlyrOptions}
      />
    </div>
  );
});

export default VideoPlayerWithPlyr;
