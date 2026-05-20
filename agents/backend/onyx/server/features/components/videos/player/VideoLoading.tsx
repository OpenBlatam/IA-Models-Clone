import React from 'react';
import { cn } from "@/lib/utils";

interface VideoLoadingProps {
  className?: string;
}

export const VideoLoading: React.FC<VideoLoadingProps> = ({
  className
}) => {
  return (
    <div className={cn(
      "absolute inset-0 flex items-center justify-center bg-zinc-900",
      className
    )}>
      <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin" />
    </div>
  );
}; 