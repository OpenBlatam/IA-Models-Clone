"use client";
import { motion } from "framer-motion";
import { BookOpen, Users, Star, Award } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface VideoContentProps {
  title: string;
  description: string;
  instructor: string;
  students: number;
  rating: number;
  progress: number;
  experience: number;
}

export default function VideoContent({
  title,
  description,
  instructor,
  students,
  rating,
  progress,
  experience,
}: VideoContentProps) {
  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-4"
      >
        <h1 className="text-2xl font-bold text-foreground">{title}</h1>
        <p className="text-muted-foreground">{description}</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="flex items-center gap-4 text-sm text-muted-foreground"
      >
        <span className="flex items-center gap-1">
          <BookOpen className="w-4 h-4" />
          {instructor}
        </span>
        <span>•</span>
        <span className="flex items-center gap-1">
          <Users className="w-4 h-4" />
          {students} estudiantes
        </span>
        <span>•</span>
        <span className="flex items-center gap-1">
          <Star className="w-4 h-4 text-primary" />
          {rating.toFixed(1)}
        </span>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="space-y-2"
      >
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">
            Progreso del curso
          </span>
          <span className="text-muted-foreground">
            {progress}%
          </span>
        </div>
        <Progress value={progress} className="h-2 bg-muted" />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="flex items-center gap-2 text-sm text-primary"
      >
        <Award className="w-4 h-4" />
        <span>Ganarás {experience} XP al completar este curso</span>
      </motion.div>
    </div>
  );
}    