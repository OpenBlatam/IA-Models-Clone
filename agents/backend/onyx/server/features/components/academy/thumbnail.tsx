"use client";

import { motion } from "framer-motion";
import { Play } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThumbnailProps {
  src: string;
  alt: string;
  duration: string;
  className?: string;
}

export function Thumbnail({ src, alt, duration, className }: ThumbnailProps) {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className={cn(
        "relative aspect-video rounded-xl overflow-hidden bg-muted group",
        className
      )}
    >
      <img
        src={src}
        alt={alt}
        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
      />
      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.2 }}
          className="bg-white/90 p-3 rounded-full"
        >
          <Play className="w-6 h-6 text-black" />
        </motion.div>
      </div>
      <div className="absolute bottom-2 right-2 bg-black/80 text-white text-xs px-2 py-1 rounded">
        {duration}
      </div>
    </motion.div>
  );
}    