"use client";
import React, { useRef, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Maximize2,
  Minimize,
  RotateCcw,
  RotateCw,
  ChevronsLeft,
  ChevronsRight
} from "lucide-react";

interface VideoPlayerCoreProps {
  videoUrl: string;
  autoPlay?: boolean;
  onTimeUpdate?: (currentTime: number) => void;
  onEnded?: () => void;
  onProgress?: (progress: number) => void;
  onError?: (error: string) => void;
  volume?: number;
  onVolumeChange?: (volume: number) => void;
}

const VideoPlayerCore: React.FC<VideoPlayerCoreProps> = ({
  videoUrl,
  autoPlay = true,
  onTimeUpdate,
  onEnded,
  onProgress,
  onError,
  volume = 1,
  onVolumeChange,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentVolume, setCurrentVolume] = useState(volume);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const controlsTimeoutRef = useRef<NodeJS.Timeout>();
  const playPromiseRef = useRef<Promise<void> | null>(null);
  const isPlayRequestedRef = useRef(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [previewTime, setPreviewTime] = useState<number | null>(null);
  const progressBarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (videoRef.current) {
      setIsLoading(true);
      setError(null);
      videoRef.current.load();
    }
  }, [videoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadStart = () => {
      setIsLoading(true);
    };

    const handleLoadedData = () => {
      setIsLoading(false);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      onTimeUpdate?.(video.currentTime);
      onProgress?.((video.currentTime / video.duration) * 100);
    };

    const handleDurationChange = () => {
      setDuration(video.duration);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      isPlayRequestedRef.current = false;
      onEnded?.();
    };

    const handleError = (e: Event) => {
      const videoElement = e.target as HTMLVideoElement;
      const error = videoElement.error;
      

      
      let errorMessage = "Error al cargar el video. Por favor, intenta de nuevo.";
      
      if (error) {
        switch (error.code) {
          case MediaError.MEDIA_ERR_ABORTED:
            errorMessage = "La reproducción del video fue interrumpida.";
            break;
          case MediaError.MEDIA_ERR_NETWORK:
            errorMessage = "Error de red al cargar el video. Verifica tu conexión a internet.";
            break;
          case MediaError.MEDIA_ERR_DECODE:
            errorMessage = "El video no se puede reproducir porque está dañado o en un formato no soportado.";
            break;
          case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
            errorMessage = "El formato del video no es soportado por tu navegador.";
            break;
          default:
            errorMessage = `Error desconocido (código: ${error.code}). Por favor, intenta de nuevo.`;
        }
      }
      
      setError(errorMessage);
      onError?.(errorMessage);
      setIsLoading(false);
      setIsPlaying(false);
      isPlayRequestedRef.current = false;
    };

    const handleCanPlay = async () => {
      setIsLoading(false);
      if (autoPlay && !isPlayRequestedRef.current) {
        try {
          isPlayRequestedRef.current = true;
          await video.play();
          setIsPlaying(true);
        } catch (error) {

          setError("Error al iniciar la reproducción. Por favor, intenta de nuevo.");
          setIsPlaying(false);
        } finally {
          isPlayRequestedRef.current = false;
        }
      }
    };

    const handleWaiting = () => {
      setIsLoading(true);
    };

    const handlePlaying = () => {
      setIsLoading(false);
    };

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
  }, [autoPlay, onTimeUpdate, onEnded, onProgress, onError]);

  // Add spacebar play/pause support
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" || e.key === " ") {
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPlaying]);

  const togglePlay = async () => {
    if (!videoRef.current) return;

    try {
      if (isPlayRequestedRef.current) {
        return; // Evitar múltiples solicitudes simultáneas
      }

      isPlayRequestedRef.current = true;

      if (isPlaying) {
        await videoRef.current.pause();
        setIsPlaying(false);
      } else {
        if (playPromiseRef.current) {
          await playPromiseRef.current;
        }
        playPromiseRef.current = videoRef.current.play();
        await playPromiseRef.current;
        setIsPlaying(true);
      }
    } catch (error) {
      setIsPlaying(false);
    } finally {
      isPlayRequestedRef.current = false;
      playPromiseRef.current = null;
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleVolumeChange = (value: number[]) => {
    const newVolume = value[0];
    if (videoRef.current) {
      videoRef.current.volume = newVolume;
      setCurrentVolume(newVolume);
      setIsMuted(newVolume === 0);
      onVolumeChange?.(newVolume);
    }
  };

  const handleSeek = (value: number[]) => {
    const newTime = value[0];
    if (videoRef.current) {
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  const handleProgressBarMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current || !progressBarRef.current) return;
    
    const progressBar = progressBarRef.current;
    const rect = progressBar.getBoundingClientRect();
    const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const newTime = position * duration;
    
    setPreviewTime(newTime);
  };

  const handleProgressBarMouseLeave = () => {
    setPreviewTime(null);
  };

  const handleProgressBarMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current || !progressBarRef.current) return;
    
    setIsDragging(true);
    const progressBar = progressBarRef.current;
    const rect = progressBar.getBoundingClientRect();
    const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const newTime = position * duration;
    
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
    setPreviewTime(null);
  };

  useEffect(() => {
    const handleMouseUp = () => {
      setIsDragging(false);
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !videoRef.current || !progressBarRef.current) return;
      
      const progressBar = progressBarRef.current;
      const rect = progressBar.getBoundingClientRect();
      const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const newTime = position * duration;
      
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mousemove', handleMouseMove);

    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, [isDragging, duration]);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      videoRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const formatTime = (time: number) => {
    const hours = Math.floor(time / 3600);
    const minutes = Math.floor((time % 3600) / 60);
    const seconds = Math.floor(time % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
    }
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  const handleMouseMove = () => {
    setShowControls(true);
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current);
    }
    controlsTimeoutRef.current = setTimeout(() => {
      if (isPlaying) {
        setShowControls(false);
      }
    }, 3000);
  };

  const skip = (seconds: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + seconds));
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const changePlaybackRate = () => {
    const rates = [1, 1.25, 1.5, 2];
    const idx = rates.indexOf(playbackRate);
    const next = rates[(idx + 1) % rates.length];
    setPlaybackRate(next);
    if (videoRef.current) videoRef.current.playbackRate = next;
  };

  return (
    <div 
      className={cn(
        "relative w-full h-full bg-black group",
        isFullscreen && "fixed inset-0 z-50"
      )}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setShowControls(false)}
    >
      {/* Loading State */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
          <div className="w-16 h-16 border-4 border-[#ff0000] border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 text-white p-4">
          <div className="text-[#ff0000] mb-2">
            <svg
              className="w-12 h-12"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <p className="text-lg font-semibold mb-2">{error}</p>
          <Button
            variant="secondary"
            onClick={() => {
              setError(null);
              setIsLoading(true);
              if (videoRef.current) {
                videoRef.current.load();
              }
            }}
          >
            Reintentar
          </Button>
        </div>
      )}

      {/* Video Element */}
      <video
        ref={videoRef}
        className="w-full h-full"
        playsInline
        preload="auto"
        crossOrigin="anonymous"
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      >
        {videoUrl && (
          <source 
            src={videoUrl} 
            type="video/mp4"
          />
        )}
        Tu navegador no soporta el elemento de video.
      </video>

      {/* Overlay Controls */}
      <AnimatePresence>
        {showControls && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent pointer-events-none"
          />
        )}
      </AnimatePresence>

      {/* Video Controls */}
      <AnimatePresence>
        {showControls && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 flex flex-col justify-end p-4 z-10"
          >
            {/* Progress Bar */}
            <div 
              ref={progressBarRef}
              className="absolute bottom-16 left-0 right-0 px-4 cursor-pointer group/progress"
              onMouseMove={handleProgressBarMouseMove}
              onMouseLeave={handleProgressBarMouseLeave}
              onMouseDown={handleProgressBarMouseDown}
            >
              <div className="relative h-1 bg-white/20 rounded-full group-hover/progress:h-2 transition-all duration-200">
                {/* Buffered Progress */}
                <div
                  className="absolute h-full bg-white/40 rounded-full"
                  style={{ width: `${(videoRef.current?.buffered.length ? videoRef.current.buffered.end(videoRef.current.buffered.length - 1) / duration : 0) * 100}%` }}
                />
                {/* Played Progress */}
                <div
                  className="absolute h-full bg-[#ff0000] rounded-full"
                  style={{ width: `${(currentTime / duration) * 100}%` }}
                />
                {/* Preview Progress */}
                {previewTime !== null && (
                  <div
                    className="absolute h-full bg-[#ff0000]/50 rounded-full"
                    style={{ width: `${(previewTime / duration) * 100}%` }}
                  />
                )}
                {/* Progress Handle */}
                <div
                  className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-[#ff0000] rounded-full opacity-0 group-hover/progress:opacity-100 transition-opacity duration-200 shadow-lg cursor-pointer"
                  style={{ 
                    left: `${((previewTime ?? currentTime) / duration) * 100}%`, 
                    transform: 'translate(-50%, -50%)'
                  }}
                />
                {/* Time Preview Tooltip */}
                {previewTime !== null && (
                  <div className="absolute bottom-full left-0 transform -translate-x-1/2 mb-2 px-2 py-1 bg-black/90 rounded text-white text-xs whitespace-nowrap shadow-lg">
                    {formatTime(previewTime)}
                  </div>
                )}
              </div>
            </div>

            {/* Controls Row */}
            <div className="flex items-center justify-between w-full bg-gradient-to-t from-black/90 to-black/70 rounded-lg px-4 py-2 shadow-lg backdrop-blur-md">
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="icon" onClick={() => skip(-15)} className="text-white hover:text-[#ff0000] transition-colors">
                  <RotateCcw className="h-6 w-6" />
                  <span className="text-xs absolute ml-1">15</span>
                </Button>
                <Button variant="ghost" size="icon" onClick={togglePlay} className="text-white hover:text-[#ff0000] transition-colors">
                  {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6" />}
                </Button>
                <Button variant="ghost" size="icon" onClick={() => skip(15)} className="text-white hover:text-[#ff0000] transition-colors">
                  <RotateCw className="h-6 w-6" />
                  <span className="text-xs absolute ml-1">15</span>
                </Button>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="icon" onClick={toggleMute} className="text-white hover:text-[#ff0000] transition-colors">
                    {isMuted ? <VolumeX className="h-6 w-6" /> : <Volume2 className="h-6 w-6" />}
                  </Button>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={currentVolume}
                    onChange={(e) => handleVolumeChange([parseFloat(e.target.value)])}
                    className="w-24 h-1 bg-white/30 rounded-full appearance-none cursor-pointer hover:h-2 transition-all duration-200"
                    style={{
                      background: `linear-gradient(to right, #ff0000 0%, #ff0000 ${currentVolume * 100}%, rgba(255, 255, 255, 0.3) ${currentVolume * 100}%, rgba(255, 255, 255, 0.3) 100%)`
                    }}
                  />
                </div>
                <div className="flex items-center gap-1 text-white text-sm font-mono min-w-[120px]">
                  <span>{formatTime(currentTime)}</span>
                  <span className="text-white/60">/</span>
                  <span className="text-white/60">{formatTime(duration)}</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="icon" onClick={changePlaybackRate} className="text-white hover:text-[#ff0000] transition-colors text-sm font-medium">
                  {playbackRate}x
                </Button>
                <Button variant="ghost" size="icon" onClick={toggleFullscreen} className="text-white hover:text-[#ff0000] transition-colors">
                  {isFullscreen ? <Minimize className="h-6 w-6" /> : <Maximize2 className="h-6 w-6" />}
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default VideoPlayerCore;
