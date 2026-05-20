'use client';

import { LessonContent } from "./lessons/lesson-content";
import { Lesson } from "./lessons/types";

interface LessonContentWrapperProps {
  lesson: Lesson;
  onComplete: () => void;
}

export function LessonContentWrapper({ lesson, onComplete }: LessonContentWrapperProps) {
  return <LessonContent lesson={lesson} onComplete={onComplete} />;
} 