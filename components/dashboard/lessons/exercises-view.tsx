'use client';

import { useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ArrowLeft, ArrowRight, Moon, Sun, CheckCircle2, XCircle, Trophy } from "lucide-react";
import { Exercise, Theme } from "./types";
import { fadeInUp, staggerContainer } from "./animations";

interface ExercisesViewProps {
  exercises: Exercise[];
  currentTheme: Theme;
  theme: 'light' | 'dark';
  onComplete: () => void;
  toggleTheme: () => void;
  onFinish: (score: number) => void;
}

export function ExercisesView({
  exercises,
  currentTheme,
  theme,
  onComplete,
  toggleTheme,
  onFinish
}: ExercisesViewProps) {
  const [currentExerciseIndex, setCurrentExerciseIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [score, setScore] = useState(0);
  const [showResults, setShowResults] = useState(false);

  const currentExercise = exercises[currentExerciseIndex];

  const handleAnswerSelect = (answer: string) => {
    setSelectedAnswer(answer);
  };

  const handleNext = () => {
    if (selectedAnswer === currentExercise.correctAnswer) {
      setScore(score + currentExercise.points);
    }

    if (currentExerciseIndex < exercises.length - 1) {
      setCurrentExerciseIndex(currentExerciseIndex + 1);
      setSelectedAnswer(null);
    } else {
      setShowResults(true);
    }
  };

  const handleFinish = () => {
    onFinish(score);
  };

  if (showResults) {
    return (
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="max-w-4xl mx-auto relative"
      >
        <div className="flex justify-start gap-2 mb-6">
          <Button
            variant="ghost"
            size="sm"
            onClick={onComplete}
            className={cn(
              "rounded-full backdrop-blur-sm border",
              theme === 'light' ? "bg-white/80 border-zinc-200/50" : "bg-zinc-800/80 border-zinc-700/50",
              "hover:scale-105 transition-all duration-300"
            )}
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Volver a lecciones
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className={cn(
              "rounded-full backdrop-blur-sm border",
              theme === 'light' ? "bg-white/80 border-zinc-200/50" : "bg-zinc-800/80 border-zinc-700/50",
              "hover:scale-110 transition-all duration-300"
            )}
          >
            {theme === 'light' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
          </Button>
        </div>
        <Card className={cn(
          "bg-gradient-to-br",
          currentTheme.background,
          "backdrop-blur-sm",
          currentTheme.border,
          currentTheme.shadow,
          "rounded-3xl overflow-hidden"
        )}>
          <CardHeader className="space-y-6 pb-8">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="flex items-center justify-center space-x-6"
            >
              <motion.div 
                className="relative"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                <motion.div 
                  className={cn(
                    "absolute inset-0 bg-gradient-to-r",
                    currentTheme.accent.primary,
                    "rounded-full blur-xl opacity-50"
                  )}
                  animate={{ 
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 0.7, 0.5]
                  }}
                  transition={{ 
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
                <div className={cn(
                  "relative p-4 rounded-full bg-gradient-to-r",
                  currentTheme.accent.primary,
                  "shadow-lg"
                )}>
                  <Trophy className="w-8 h-8 text-white" />
                </div>
              </motion.div>
              <div className="space-y-3 text-center">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <CardTitle className={cn(
                    "text-4xl font-bold bg-gradient-to-r",
                    currentTheme.text.accent,
                    "bg-clip-text text-transparent tracking-tight"
                  )}>
                    ¡Felicitaciones!
                  </CardTitle>
                </motion.div>
                <motion.p 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  className={cn("text-lg", currentTheme.text.secondary, "font-light tracking-wide")}
                >
                  Has completado los ejercicios con una puntuación de {score} puntos
                </motion.p>
              </div>
            </motion.div>
          </CardHeader>
          <CardContent className="space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <Button
                onClick={handleFinish}
                className={cn(
                  "w-full bg-gradient-to-r",
                  currentTheme.button.primary,
                  "text-white py-8 text-xl font-medium tracking-wide rounded-2xl",
                  "transition-all duration-500 transform hover:scale-[1.02]",
                  currentTheme.shadow,
                  currentTheme.hoverShadow,
                  "group"
                )}
              >
                <motion.span
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.8 }}
                  className="flex items-center justify-center"
                >
                  Finalizar Lección
                </motion.span>
              </Button>
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="max-w-4xl mx-auto relative px-4 sm:px-8"
    >
      <div className="flex justify-start gap-2 mb-4 sm:mb-6">
        <Button
          variant="ghost"
          size="sm"
          onClick={onComplete}
          className={cn(
            "rounded-full backdrop-blur-sm border text-sm sm:text-base",
            theme === 'light' ? "bg-white/80 border-zinc-200/50" : "bg-zinc-800/80 border-zinc-700/50",
            "hover:scale-105 transition-all duration-300"
          )}
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Volver
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          className={cn(
            "rounded-full backdrop-blur-sm border",
            theme === 'light' ? "bg-white/80 border-zinc-200/50" : "bg-zinc-800/80 border-zinc-700/50",
            "hover:scale-110 transition-all duration-300"
          )}
        >
          {theme === 'light' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
        </Button>
      </div>
      <Card className={cn(
        "bg-gradient-to-br",
        currentTheme.background,
        "backdrop-blur-sm",
        currentTheme.border,
        currentTheme.shadow,
        "rounded-2xl sm:rounded-3xl overflow-hidden"
      )}>
        <CardHeader className="space-y-4 sm:space-y-6 pb-4 sm:pb-8">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h2 className={cn(
                "text-xl sm:text-2xl font-bold bg-gradient-to-r",
                currentTheme.text.accent,
                "bg-clip-text text-transparent tracking-tight"
              )}>
                Ejercicio {currentExerciseIndex + 1} de {exercises.length}
              </h2>
              <p className={cn("text-sm sm:text-base", currentTheme.text.secondary)}>
                {exercises[currentExerciseIndex].question}
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6 sm:space-y-8">
          <div className="grid gap-4">
            {exercises[currentExerciseIndex].options.map((option, index) => (
              <Button
                key={index}
                variant="outline"
                onClick={() => handleAnswerSelect(option)}
                className={cn(
                  "w-full justify-start text-left p-4 sm:p-6 rounded-xl sm:rounded-2xl",
                  "border-2 transition-all duration-300",
                  currentTheme.border,
                  "hover:border-purple-500/50 hover:shadow-lg hover:shadow-purple-500/10",
                  "group"
                )}
              >
                <div className="flex items-center space-x-4">
                  <div className={cn(
                    "flex items-center justify-center w-8 h-8 sm:w-10 sm:h-10 rounded-full",
                    "bg-gradient-to-r",
                    currentTheme.accent.primary,
                    "text-white font-medium text-sm sm:text-base"
                  )}>
                    {String.fromCharCode(65 + index)}
                  </div>
                  <span className="text-sm sm:text-base">{option}</span>
                </div>
              </Button>
            ))}
          </div>
          <div className="flex justify-between items-center pt-4">
            <Button
              variant="ghost"
              onClick={() => setCurrentExerciseIndex(Math.max(currentExerciseIndex - 1, 0))}
              disabled={currentExerciseIndex === 0}
              className={cn(
                "rounded-full backdrop-blur-sm border text-sm sm:text-base",
                theme === 'light' ? "bg-white/80 border-zinc-200/50" : "bg-zinc-800/80 border-zinc-700/50",
                "hover:scale-105 transition-all duration-300",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Anterior
            </Button>
            <Button
              variant="ghost"
              onClick={handleNext}
              disabled={currentExerciseIndex === exercises.length - 1}
              className={cn(
                "rounded-full backdrop-blur-sm border text-sm sm:text-base",
                theme === 'light' ? "bg-white/80 border-zinc-200/50" : "bg-zinc-800/80 border-zinc-700/50",
                "hover:scale-105 transition-all duration-300",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              Siguiente
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}    