"use client";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface CourseMetaProps {
  level: string;
  category: string;
  totalClasses: number;
  experience: number;
  progress?: number;
  className?: string;
}

export function CourseMeta({
  level,
  category,
  totalClasses,
  experience,
  progress = 0,
  className,
}: CourseMetaProps) {
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center gap-2">
        <Badge variant="secondary" className="text-xs capitalize">
          {level}
        </Badge>
        <Badge variant="outline" className="text-xs">
          {category}
        </Badge>
      </div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span>{totalClasses} clases</span>
        <span>•</span>
        <span>{experience} XP</span>
      </div>
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">Progreso</span>
          <span className="font-medium">{progress}%</span>
        </div>
        <Progress value={progress} className="h-1" />
      </div>
    </div>
  );
} 