"use client";

import React, { useRef, useEffect, useState, memo } from "react";
import Hls from "hls.js";
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
  Settings,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { VideoPlayerProps } from "@/lib/types/video";
import { useVideoPlayer } from "@/lib/hooks/useVideoPlayer";
import { useVideoControls } from "@/lib/hooks/useVideoControls";
import { useVideoAnalytics } from "@/lib/hooks/useVideoAnalytics";
import { toast } from "react-hot-toast";

interface VideoPlayerWithHLSProps extends VideoPlayerProps {
  videoId?: string;
  userId?: string;
  enableAnalytics?: boolean;
  hlsConfig?: any;
  enableAdaptiveStreaming?: boolean;
}

const VideoPlayerWithHLS = memo(function VideoPlayerWithHLS({
  videoUrl,
  videoId,
  userId,
  autoPlay = false,
  volume = 1,
  enableAnalytics = true,
  enableAdaptiveStreaming = true,
  onTimeUpdate,
  onEnded,
  onProgress,
  onError,
  onVolumeChange,
  hlsConfig = {},
  className,
}: VideoPlayerWithHLSProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const [isHLSSupported, setIsHLSSupported] = useState(false);
  const [currentQuality, setCurrentQuality] = useState<number>(-1);
  const [availableQualities, setAvailableQualities] = useState<any[]>([]);

  const { state, actions } = useVideoPlayer({
    videoUrl,
    autoPlay,
    volume,
    onTimeUpdate,
    onEnded,
    onProgress,
    onError,
    onVolumeChange,
  });

  const { showControls, actions: controlActions } = useVideoControls();

  const { trackWatchTime, trackVideoEvent } = useVideoAnalytics({
    videoId: videoId || '',
    userId,
  });

  const defaultHLSConfig = {
    enableWorker: true,
    lowLatencyMode: true,
    backBufferLength: 90,
    maxBufferLength: 30,
    maxMaxBufferLength: 600,
    maxBufferSize: 60 * 1000 * 1000,
    maxBufferHole: 0.5,
    highBufferWatchdogPeriod: 2,
    nudgeOffset: 0.1,
    nudgeMaxRetry: 3,
    maxFragLookUpTolerance: 0.25,
    liveSyncDurationCount: 3,
    liveMaxLatencyDurationCount: Infinity,
    liveDurationInfinity: false,
    enableSoftwareAES: true,
    manifestLoadingTimeOut: 10000,
    manifestLoadingMaxRetry: 1,
    manifestLoadingRetryDelay: 1000,
    levelLoadingTimeOut: 10000,
    levelLoadingMaxRetry: 4,
    levelLoadingRetryDelay: 1000,
    fragLoadingTimeOut: 20000,
    fragLoadingMaxRetry: 6,
    fragLoadingRetryDelay: 1000,
    startFragPrefetch: false,
    testBandwidth: true,
    progressive: false,
    ...hlsConfig,
  };

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const isHLS = videoUrl.includes('.m3u8');
    setIsHLSSupported(isHLS && Hls.isSupported());

    if (isHLS && Hls.isSupported()) {
      const hls = new Hls(defaultHLSConfig);
      hlsRef.current = hls;

      hls.loadSource(videoUrl);
      hls.attachMedia(video);

      hls.on(Hls.Events.MANIFEST_PARSED, (event, data) => {
        if (enableAdaptiveStreaming) {
          const qualities = data.levels.map((level: any, index: number) => ({
            index,
            height: level.height,
            width: level.width,
            bitrate: level.bitrate,
            name: `${level.height}p`,
          }));
          
          qualities.unshift({ index: -1, name: 'Auto', height: 0, width: 0, bitrate: 0 });
          setAvailableQualities(qualities);
          setCurrentQuality(-1);
        }

        if (enableAnalytics && videoId) {
          trackVideoEvent('hls_manifest_parsed', {
            levels: data.levels.length,
          });
        }
      });

      hls.on(Hls.Events.LEVEL_SWITCHED, (event, data) => {
        setCurrentQuality(data.level);
        
        if (enableAnalytics && videoId) {
          trackVideoEvent('quality_change', {
            level: data.level,
            height: hls.levels[data.level]?.height,
          });
        }
      });

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          switch (data.type) {
            case Hls.ErrorTypes.NETWORK_ERROR:
              hls.startLoad();
              break;
            case Hls.ErrorTypes.MEDIA_ERROR:
              hls.recoverMediaError();
              break;
            default:
              const errorMessage = "Error crítico en la transmisión HLS";
              onError?.(errorMessage);
              toast.error(errorMessage);
              break;
          }
        }

        if (enableAnalytics && videoId) {
          trackVideoEvent('hls_error', {
            type: data.type,
            details: data.details,
            fatal: data.fatal,
          });
        }
      });

      hls.on(Hls.Events.BUFFER_APPENDED, () => {
        if (enableAnalytics && videoId) {
          trackVideoEvent('buffer_health', {
            buffered: video.buffered.length > 0 ? video.buffered.end(0) - video.currentTime : 0,
          });
        }
      });

      return () => {
        if (hlsRef.current) {
          hlsRef.current.destroy();
          hlsRef.current = null;
        }
      };
    } else if (isHLS && !Hls.isSupported() && video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = videoUrl;
    } else {
      video.src = videoUrl;
    }
  }, [videoUrl, enableAnalytics, videoId, enableAdaptiveStreaming, onError, trackVideoEvent]);

  const handleQualityChange = (qualityIndex: number) => {
    if (hlsRef.current) {
      hlsRef.current.currentLevel = qualityIndex;
      setCurrentQuality(qualityIndex);
      
      if (enableAnalytics && videoId) {
        trackVideoEvent('manual_quality_change', { level: qualityIndex });
      }
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

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
              if (hlsRef.current) {
                hlsRef.current.startLoad();
              } else {
                videoRef.current?.load();
              }
            }}
            variant="outline"
            className="text-white border-white hover:bg-white hover:text-black"
          >
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
                onClick={actions.togglePlayPause}
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
                  className="relative w-full h-2 bg-white/20 rounded-full cursor-pointer"
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const clickX = e.clientX - rect.left;
                    const percentage = clickX / rect.width;
                    const newTime = percentage * state.duration;
                    actions.seekTo(newTime);
                  }}
                >
                  <div
                    className="absolute top-0 left-0 h-full bg-primary rounded-full transition-all duration-150"
                    style={{ width: `${state.duration > 0 ? (state.currentTime / state.duration) * 100 : 0}%` }}
                  />
                </div>
              </div>

              {/* Control Buttons */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Button
                    onClick={actions.togglePlayPause}
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
                      onChange={(e) => actions.changeVolume(parseFloat(e.target.value))}
                      className="w-20 h-1 bg-white/20 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  <span className="text-white text-sm">
                    {formatTime(state.currentTime)} / {formatTime(state.duration)}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  {/* Quality Selector for HLS */}
                  {isHLSSupported && availableQualities.length > 0 && (
                    <select
                      value={currentQuality}
                      onChange={(e) => handleQualityChange(parseInt(e.target.value))}
                      className="bg-black/50 text-white text-sm rounded px-2 py-1 border border-white/20"
                    >
                      {availableQualities.map((quality) => (
                        <option key={quality.index} value={quality.index}>
                          {quality.name}
                        </option>
                      ))}
                    </select>
                  )}

                  {/* Playback Rate */}
                  <select
                    value={state.playbackRate}
                    onChange={(e) => actions.changePlaybackRate(parseFloat(e.target.value))}
                    className="bg-black/50 text-white text-sm rounded px-2 py-1 border border-white/20"
                  >
                    <option value={0.5}>0.5x</option>
                    <option value={0.75}>0.75x</option>
                    <option value={1}>1x</option>
                    <option value={1.25}>1.25x</option>
                    <option value={1.5}>1.5x</option>
                    <option value={2}>2x</option>
                  </select>

                  <Button
                    onClick={actions.toggleFullscreen}
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

export default VideoPlayerWithHLS;
