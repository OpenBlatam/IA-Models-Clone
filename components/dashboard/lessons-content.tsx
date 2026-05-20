'use client';

import { useState, useCallback, memo } from 'react';
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/shared/icons";
import { cn } from "@/lib/utils";
import { LessonContentWrapper } from "./lesson-content";
import { Lesson } from "./lessons/types";

interface LessonsContentProps {
  lessons: Lesson[];
  userLevel: number;
}

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};

// Memoized Lesson Card component with enhanced design
const LessonCard = memo(({ 
  lesson, 
  userLevel, 
  onClick 
}: { 
  lesson: Lesson; 
  userLevel: number; 
  onClick: (lesson: Lesson) => void;
}) => {
  const isLocked = lesson.requiredLevel > userLevel;
  
  return (
    <motion.div variants={item}>
      <Card 
        className={cn(
          "relative overflow-hidden cursor-pointer transition-all duration-300",
          "bg-gray-800/30 backdrop-blur-sm border border-gray-700/50",
          "hover:border-purple-500/50 hover:shadow-lg hover:shadow-purple-500/10",
          "group",
          isLocked && "opacity-50 cursor-not-allowed"
        )}
        onClick={() => !isLocked && onClick(lesson)}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-blue-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative z-10">
          <CardTitle className="text-sm font-medium text-gray-200">
            {lesson.title}
          </CardTitle>
          {isLocked ? (
            <Icons.lock className="h-4 w-4 text-gray-500" />
          ) : (
            <Icons.bookOpen className="h-4 w-4 text-purple-500" />
          )}
        </CardHeader>
        <CardContent className="relative z-10">
          <p className="text-xs text-gray-400 mb-4">
            {lesson.description}
          </p>
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <Icons.target className="h-3 w-3 text-purple-500" />
              <span className="text-gray-300">Nivel {lesson.requiredLevel}</span>
            </div>
            <div className="flex items-center gap-2">
              <Icons.star className="h-3 w-3 text-yellow-500" />
              <span className="text-gray-300">{lesson.experienceReward} XP</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
});

LessonCard.displayName = 'LessonCard';

export function LessonsContent({ lessons, userLevel }: LessonsContentProps) {
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null);

  const handleLessonClick = useCallback((lesson: Lesson) => {
    if (lesson.requiredLevel <= userLevel) {
      setSelectedLesson(lesson);
    }
  }, [userLevel]);

  const handleLessonComplete = useCallback(() => {
    setSelectedLesson(null);
  }, []);

  if (selectedLesson) {
    return (
      <div className="p-8 pt-6">
        <LessonContentWrapper lesson={selectedLesson} onComplete={handleLessonComplete} />
      </div>
    );
  }

  return (
    <motion.div 
      className="space-y-4"
      variants={container}
      initial="hidden"
      animate="show"
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {lessons.map((lesson) => (
          <LessonCard
            key={lesson.id}
            lesson={lesson}
            userLevel={userLevel}
            onClick={handleLessonClick}
          />
        ))}
      </div>
    </motion.div>
  );
}    