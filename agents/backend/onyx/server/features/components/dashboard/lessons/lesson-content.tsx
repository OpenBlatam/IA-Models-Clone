'use client';

import { useState } from "react";
import { Lesson, Theme } from "./types";
import { themes } from "./themes";
import { LessonContentView } from "./lesson-content-view";
import { ExercisesView } from "./exercises-view";
import { useToast } from "@/components/ui/use-toast";
import { useRouter } from "next/navigation";

interface LessonContentProps {
  lesson: Lesson;
  onComplete: () => void;
}

export function LessonContent({ lesson, onComplete }: LessonContentProps) {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const [showExercises, setShowExercises] = useState(false);
  const { toast } = useToast();
  const router = useRouter();

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  const handleStartExercises = () => {
    setShowExercises(true);
  };

  const handleFinishExercises = async (score: number) => {
    try {
      const response = await fetch('/api/lessons/complete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lessonId: lesson.id,
          score: score,
          experienceReward: lesson.experienceReward
        }),
      });

      if (!response.ok) {
        throw new Error('Error al completar la lección');
      }

      const data = await response.json();

      toast({
        title: "¡Lección Completada! 🎉",
        description: `Has ganado ${lesson.experienceReward} puntos de experiencia. ${data.levelUp ? '¡Has subido de nivel!' : ''}`,
        duration: 5000,
      });

      router.refresh();
      onComplete();
    } catch (error) {
      toast({
        title: "Error",
        description: "Hubo un problema al completar la lección. Por favor, intenta de nuevo.",
        variant: "destructive",
      });
    }
  };

  if (showExercises) {
    return (
      <ExercisesView
        exercises={lesson.exercises}
        currentTheme={themes[theme]}
        theme={theme}
        onComplete={onComplete}
        toggleTheme={toggleTheme}
        onFinish={handleFinishExercises}
      />
    );
  }

  return (
    <LessonContentView
      lesson={lesson}
      currentContent={lesson.content ? {
        title: lesson.content.sections?.[0]?.title || lesson.title,
        description: lesson.content.sections?.[0]?.content || lesson.description,
        sections: lesson.content.sections?.map(section => ({
          title: section.title,
          description: section.content,
          icon: 'globe' as const,
          example: section.content
        })) || []
      } : { 
        title: lesson.title, 
        description: lesson.description, 
        sections: [] 
      }}
      currentTheme={themes[theme]}
      theme={theme}
      onComplete={onComplete}
      toggleTheme={toggleTheme}
      onStartExercises={handleStartExercises}
    />
  );
}      