"use client";

import { motion } from "framer-motion";
import { Award, Clock, ListVideo } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";

interface VideoHeaderProps {
  title: string;
  instructor: string;
  progress: number;
  experience: number;
  currentClass: number;
  totalClasses: number;
  duration: string;
  onToggleContent: () => void;
  isContentVisible: boolean;
}

export default function VideoHeader({
  title,
  instructor,
  progress,
  experience,
  currentClass,
  totalClasses,
  duration,
  onToggleContent,
  isContentVisible,
}: VideoHeaderProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-foreground">{title}</h1>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <Award className="w-4 h-4" />
              {experience} XP
            </span>
            <span>•</span>
            <span className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              {duration}
            </span>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={onToggleContent}
          className="text-muted-foreground hover:text-foreground"
        >
          <ListVideo className="w-4 h-4 mr-2" />
          {isContentVisible ? "Ocultar Contenido" : "Ver Contenido"}
        </Button>
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">
            Progreso: {progress}%
          </span>
          <span className="text-muted-foreground">
            Clase {currentClass} de {totalClasses}
          </span>
        </div>
        <Progress value={progress} className="h-2 bg-muted" />
      </div>
    </div>
  );
}    