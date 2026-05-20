"use client";

import React, { memo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Maximize,
  Minimize,
  SkipBack,
  SkipForward,
  Loader2,
  AlertCircle,
  RotateCw,
  Settings,
} from "lucide-react";
import { useVideoPlayer } from "@/lib/hooks/useVideoPlayer";
import { useVideoControls } from "@/lib/hooks/useVideoControls";
import { useVideoAnalytics } from "@/lib/hooks/useVideoAnalytics";
import { VideoPlayerProps } from "@/lib/types/video";
import { toast } from "react-hot-toast";

interface VideoPlayerEnhancedProps extends VideoPlayerProps {
  videoId?: string;
  userId?: string;
  enableAnalytics?: boolean;
  enableKeyboardShortcuts?: boolean;
  showQualitySelector?: boolean;
  showPlaybackRateSelector?: boolean;
  customControls?: React.ReactNode;
}

const VideoPlayerEnhanced = memo(function VideoPlayerEnhanced({
  videoUrl,
  videoId,
  userId,
  autoPlay = false,
  volume = 1,
  enableAnalytics = true,
  enableKeyboardShortcuts = true,
  showQualitySelector = true,
  showPlaybackRateSelector = true,
  onTimeUpdate,
  onEnded,
  onProgress,
  onError,
  onVolumeChange,
  customControls,
  className,
}: VideoPlayerEnhancedProps) {
  const { videoRef, progressBarRef, state, actions } = useVideoPlayer({
    videoUrl,
    autoPlay,
    volume,
    onTimeUpdate: useCallback((time: number) => {
      onTimeUpdate?.(time);
      if (enableAnalytics && videoId) {
        trackWatchTime(time, state.isPlaying);
      }
    }, [onTimeUpdate, enableAnalytics, videoId]),
    onEnded: useCallback(() => {
      onEnded?.();
      if (enableAnalytics && videoId) {
        trackVideoEvent('video_completed', { duration: state.duration });
      }
    }, [onEnded, enableAnalytics, videoId]),
    onProgress,
    onError: useCallback((error: string) => {
      onError?.(error);
      toast.error(error);
      if (enableAnalytics && videoId) {
        trackVideoEvent('video_error', { error });
      }
    }, [onError, enableAnalytics, videoId]),
    onVolumeChange,
  });

  const { showControls, isDragging, previewTime, actions: controlActions } = useVideoControls();

  const { trackWatchTime, trackVideoEvent } = useVideoAnalytics({
    videoId: videoId || '',
    userId,
  });

  const formatTime = useCallback((time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  }, []);

  const handleProgressClick = useCallback((e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * state.duration;
    actions.seekTo(newTime);
    
    if (enableAnalytics && videoId) {
      trackVideoEvent('video_seek', { from: state.currentTime, to: newTime });
    }
  }, [state.duration, state.currentTime, actions, enableAnalytics, videoId, trackVideoEvent]);

  const handlePlayPause = useCallback(() => {
    actions.togglePlayPause();
    if (enableAnalytics && videoId) {
      trackVideoEvent(state.isPlaying ? 'video_pause' : 'video_play', {
        currentTime: state.currentTime,
      });
    }
  }, [actions, enableAnalytics, videoId, state.isPlaying, state.currentTime, trackVideoEvent]);

  const handleVolumeChange = useCallback((newVolume: number) => {
    actions.changeVolume(newVolume);
    if (enableAnalytics && videoId) {
      trackVideoEvent('volume_change', { volume: newVolume });
    }
  }, [actions, enableAnalytics, videoId, trackVideoEvent]);

  const handlePlaybackRateChange = useCallback((rate: number) => {
    actions.changePlaybackRate(rate);
    if (enableAnalytics && videoId) {
      trackVideoEvent('playback_rate_change', { rate });
    }
  }, [actions, enableAnalytics, videoId, trackVideoEvent]);

  const handleFullscreen = useCallback(() => {
    actions.toggleFullscreen();
    if (enableAnalytics && videoId) {
      trackVideoEvent('fullscreen_toggle', { isFullscreen: !state.isFullscreen });
    }
  }, [actions, enableAnalytics, videoId, state.isFullscreen, trackVideoEvent]);

  if (state.error) {
    return (
      <div className="relative w-full aspect-video bg-black rounded-lg flex items-center justify-center">
        <div className="text-center text-white">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-500" />
          <p className="text-lg font-medium mb-2">Error de reproducción</p>
          <p className="text-sm text-gray-300 mb-4">{state.error}</p>
          <Button
            onClick={() => {
              actions.updateState({ error: null, isLoading: true });
              videoRef.current?.load();
            }}
            variant="outline"
            className="text-white border-white hover:bg-white hover:text-black"
          >
            <RotateCw className="w-4 h-4 mr-2" />
            Reintentar
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div 
      className={cn("relative w-full aspect-video bg-black rounded-lg overflow-hidden group", className)}
      onMouseMove={controlActions.handleMouseMove}
      onMouseLeave={controlActions.handleMouseLeave}
    >
      <video
        ref={videoRef}
        src={videoUrl}
        className="w-full h-full object-cover"
        playsInline
        preload="metadata"
      />

      {/* Loading Overlay */}
      <AnimatePresence>
        {state.isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/50 flex items-center justify-center"
          >
            <Loader2 className="w-12 h-12 text-white animate-spin" />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Controls Overlay */}
      <AnimatePresence>
        {showControls && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/40"
          >
            {/* Center Play/Pause Button */}
            <div className="absolute inset-0 flex items-center justify-center">
              <Button
                onClick={handlePlayPause}
                variant="ghost"
                size="lg"
                className="w-16 h-16 rounded-full bg-black/50 hover:bg-black/70 text-white border-2 border-white/20"
              >
                {state.isPlaying ? (
                  <Pause className="w-8 h-8" />
                ) : (
                  <Play className="w-8 h-8 ml-1" />
                )}
              </Button>
            </div>

            {/* Bottom Controls */}
            <div className="absolute bottom-0 left-0 right-0 p-4">
              {/* Progress Bar */}
              <div className="mb-4">
                <div
                  ref={progressBarRef}
                  className="relative w-full h-2 bg-white/20 rounded-full cursor-pointer"
                  onMouseDown={controlActions.startDragging}
                  onMouseUp={controlActions.stopDragging}
                  onClick={handleProgressClick}
                >
                  <div
                    className="absolute top-0 left-0 h-full bg-primary rounded-full transition-all duration-150"
                    style={{ width: `${state.duration > 0 ? (state.currentTime / state.duration) * 100 : 0}%` }}
                  />
                  {previewTime !== null && (
                    <div
                      className="absolute top-0 h-full w-1 bg-white rounded-full"
                      style={{ left: `${(previewTime / state.duration) * 100}%` }}
                    />
                  )}
                </div>
              </div>

              {/* Control Buttons */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Button
                    onClick={handlePlayPause}
                    variant="ghost"
                    size="sm"
                    className="text-white hover:bg-white/20"
                  >
                    {state.isPlaying ? (
                      <Pause className="w-5 h-5" />
                    ) : (
                      <Play className="w-5 h-5" />
                    )}
                  </Button>

                  <Button
                    onClick={() => actions.seekTo(Math.max(0, state.currentTime - 10))}
                    variant="ghost"
                    size="sm"
                    className="text-white hover:bg-white/20"
                  >
                    <SkipBack className="w-5 h-5" />
                  </Button>

                  <Button
                    onClick={() => actions.seekTo(Math.min(state.duration, state.currentTime + 10))}
                    variant="ghost"
                    size="sm"
                    className="text-white hover:bg-white/20"
                  >
                    <SkipForward className="w-5 h-5" />
                  </Button>

                  <div className="flex items-center gap-2">
                    <Button
                      onClick={actions.toggleMute}
                      variant="ghost"
                      size="sm"
                      className="text-white hover:bg-white/20"
                    >
                      {state.isMuted || state.currentVolume === 0 ? (
                        <VolumeX className="w-5 h-5" />
                      ) : (
                        <Volume2 className="w-5 h-5" />
                      )}
                    </Button>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={state.currentVolume}
                      onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
                      className="w-20 h-1 bg-white/20 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  <span className="text-white text-sm">
                    {formatTime(state.currentTime)} / {formatTime(state.duration)}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  {showPlaybackRateSelector && (
                    <select
                      value={state.playbackRate}
                      onChange={(e) => handlePlaybackRateChange(parseFloat(e.target.value))}
                      className="bg-black/50 text-white text-sm rounded px-2 py-1 border border-white/20"
                    >
                      <option value={0.5}>0.5x</option>
                      <option value={0.75}>0.75x</option>
                      <option value={1}>1x</option>
                      <option value={1.25}>1.25x</option>
                      <option value={1.5}>1.5x</option>
                      <option value={2}>2x</option>
                    </select>
                  )}

                  {customControls}

                  <Button
                    onClick={handleFullscreen}
                    variant="ghost"
                    size="sm"
                    className="text-white hover:bg-white/20"
                  >
                    {state.isFullscreen ? (
                      <Minimize className="w-5 h-5" />
                    ) : (
                      <Maximize className="w-5 h-5" />
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

export default VideoPlayerEnhanced;
