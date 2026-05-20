import React from 'react';
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { 
  Play, 
  Pause, 
  Volume2, 
  VolumeX, 
  Maximize2, 
  Minimize,
  ChevronLeft,
  ChevronRight
} from "lucide-react";
import { cn } from "@/lib/utils";

interface VideoControlsProps {
  isPlaying: boolean;
  isMuted: boolean;
  volume: number;
  currentTime: number;
  duration: number;
  isFullscreen: boolean;
  onPlayPause: () => void;
  onMute: () => void;
  onVolumeChange: (value: number[]) => void;
  onSeek: (value: number[]) => void;
  onFullscreen: () => void;
  onPrevious: () => void;
  onNext: () => void;
  showControls: boolean;
  className?: string;
}

export const VideoControls: React.FC<VideoControlsProps> = ({
  isPlaying,
  isMuted,
  volume,
  currentTime,
  duration,
  isFullscreen,
  onPlayPause,
  onMute,
  onVolumeChange,
  onSeek,
  onFullscreen,
  onPrevious,
  onNext,
  showControls,
  className
}) => {
  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  return (
    <div className={cn(
      "absolute inset-0 flex flex-col justify-end p-4 z-10 transition-opacity duration-300",
      !showControls && "opacity-0",
      className
    )}>
      {/* Progress Bar */}
      <div className="w-full h-1 bg-gray-600 rounded-full mb-4">
        <div 
          className="h-full bg-primary rounded-full"
          style={{ width: `${(currentTime / duration) * 100}%` }}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={onPlayPause}
            className="text-white hover:text-primary"
          >
            {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6" />}
          </Button>

          <Button
            variant="ghost"
            size="icon"
            onClick={onMute}
            className="text-white hover:text-primary"
          >
            {isMuted ? <VolumeX className="h-6 w-6" /> : <Volume2 className="h-6 w-6" />}
          </Button>

          <div className="flex items-center space-x-2">
            <span className="text-white text-sm">{formatTime(currentTime)}</span>
            <span className="text-white text-sm">/</span>
            <span className="text-white text-sm">{formatTime(duration)}</span>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={onPrevious}
            className="text-white hover:text-primary"
          >
            <ChevronLeft className="h-6 w-6" />
          </Button>

          <Button
            variant="ghost"
            size="icon"
            onClick={onNext}
            className="text-white hover:text-primary"
          >
            <ChevronRight className="h-6 w-6" />
          </Button>

          <Button
            variant="ghost"
            size="icon"
            onClick={onFullscreen}
            className="text-white hover:text-primary"
          >
            {isFullscreen ? <Minimize className="h-6 w-6" /> : <Maximize2 className="h-6 w-6" />}
          </Button>
        </div>
      </div>
    </div>
  );
}; 