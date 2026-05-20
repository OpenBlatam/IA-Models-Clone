'use client';

import { useState, useRef, VideoHTMLAttributes } from 'react';
import { Play, Pause, Volume2, VolumeX } from 'lucide-react';
import { Button } from './Button';
import { cn } from '@/lib/utils';

interface VideoProps extends Omit<VideoHTMLAttributes<HTMLVideoElement>, 'controls'> {
  showControls?: boolean;
  autoplay?: boolean;
  className?: string;
}

export const Video = ({
  src,
  showControls = true,
  autoplay = false,
  className,
  ...props
}: VideoProps) => {
  const [isPlaying, setIsPlaying] = useState(autoplay);
  const [isMuted, setIsMuted] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  return (
    <div className={cn('relative', className)}>
      <video
        ref={videoRef}
        src={src}
        className="w-full h-full object-cover"
        autoPlay={autoplay}
        muted={isMuted}
        {...props}
      />
      {showControls && (
        <div className="absolute bottom-4 left-4 flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={handlePlayPause}
            aria-label={isPlaying ? 'Pausar' : 'Reproducir'}
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={handleMute}
            aria-label={isMuted ? 'Activar sonido' : 'Silenciar'}
          >
            {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
          </Button>
        </div>
      )}
    </div>
  );
};

