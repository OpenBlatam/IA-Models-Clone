'use client';

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ArrowLeft, Moon, Sun, ChevronRight, Globe, Share, BarChart, Target, Brain, Rocket, Sparkles } from "lucide-react";
import { Lesson, MarketingSection, Theme } from "./types";
import { fadeInUp, staggerContainer } from "./animations";

interface LessonContentViewProps {
  lesson: Lesson;
  currentContent: {
    title: string;
    description: string;
    sections: MarketingSection[];
  };
  currentTheme: Theme;
  theme: 'light' | 'dark';
  onComplete: () => void;
  toggleTheme: () => void;
  onStartExercises: () => void;
}

export function LessonContentView({
  lesson,
  currentContent,
  currentTheme,
  theme,
  onComplete,
  toggleTheme,
  onStartExercises
}: LessonContentViewProps) {
  if (!currentContent) {
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
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="flex flex-col sm:flex-row items-center sm:items-start space-y-4 sm:space-y-0 sm:space-x-6"
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
                  "relative p-3 sm:p-4 rounded-full bg-gradient-to-r",
                  currentTheme.accent.primary,
                  "shadow-lg"
                )}>
                  {lesson.id === "marketing" ? (
                    <Globe className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                  ) : lesson.id === "market-segmentation" ? (
                    <Target className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                  ) : lesson.id === "brand-positioning" ? (
                    <Target className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                  ) : lesson.id === "web-analytics" ? (
                    <BarChart className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                  ) : (
                    <Target className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                  )}
                </div>
              </motion.div>
              <div className="space-y-2 sm:space-y-3 text-center sm:text-left">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <CardTitle className={cn(
                    "text-2xl sm:text-4xl font-bold bg-gradient-to-r",
                    currentTheme.text.accent,
                    "bg-clip-text text-transparent tracking-tight"
                  )}>
                    {lesson.title}
                  </CardTitle>
                </motion.div>
                <motion.p 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  className={cn("text-base sm:text-lg", currentTheme.text.secondary, "font-light tracking-wide")}
                >
                  {lesson.description}
                </motion.p>
              </div>
            </motion.div>
          </CardHeader>
          <CardContent className="space-y-6 sm:space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <Button
                onClick={onStartExercises}
                className={cn(
                  "w-full bg-gradient-to-r",
                  currentTheme.button.primary,
                  "text-white py-6 sm:py-8 text-lg sm:text-xl font-medium tracking-wide rounded-xl sm:rounded-2xl",
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
                  Comenzar Ejercicios
                  <motion.div
                    animate={{ x: [0, 5, 0] }}
                    transition={{ duration: 1, repeat: Infinity }}
                    className="ml-2"
                  >
                    <ChevronRight className="h-5 w-5 sm:h-6 sm:w-6 group-hover:translate-x-1 transition-transform" />
                  </motion.div>
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
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col sm:flex-row items-center sm:items-start space-y-4 sm:space-y-0 sm:space-x-6"
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
                "relative p-3 sm:p-4 rounded-full bg-gradient-to-r",
                currentTheme.accent.primary,
                "shadow-lg"
              )}>
                {lesson.id === "marketing" ? (
                  <Globe className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                ) : lesson.id === "market-segmentation" ? (
                  <Target className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                ) : lesson.id === "brand-positioning" ? (
                  <Target className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                ) : lesson.id === "web-analytics" ? (
                  <BarChart className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                ) : (
                  <Target className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                )}
              </div>
            </motion.div>
            <div className="space-y-2 sm:space-y-3 text-center sm:text-left">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                <CardTitle className={cn(
                  "text-2xl sm:text-4xl font-bold bg-gradient-to-r",
                  currentTheme.text.accent,
                  "bg-clip-text text-transparent tracking-tight"
                )}>
                  {currentContent.title}
                </CardTitle>
              </motion.div>
              <motion.p 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
                className={cn("text-base sm:text-lg", currentTheme.text.secondary, "font-light tracking-wide")}
              >
                {currentContent.description}
              </motion.p>
            </div>
          </motion.div>
        </CardHeader>
        <CardContent className="space-y-6 sm:space-y-8">
          <motion.div
            variants={staggerContainer}
            initial="initial"
            animate="animate"
            className="space-y-6 sm:space-y-8"
          >
            {currentContent.sections.map((section, index) => (
              <motion.div
                key={index}
                variants={fadeInUp}
                whileHover={{ scale: 1.02 }}
                className={cn(
                  "group p-4 sm:p-8 rounded-xl sm:rounded-2xl",
                  currentTheme.card,
                  "backdrop-blur-sm",
                  currentTheme.border,
                  currentTheme.shadow,
                  currentTheme.hoverShadow,
                  "transition-all duration-500"
                )}
              >
                <div className="flex flex-col sm:flex-row items-center sm:items-start space-y-4 sm:space-y-0 sm:space-x-6">
                  <motion.div 
                    className="relative"
                    whileHover={{ rotate: 360 }}
                    transition={{ duration: 0.5 }}
                  >
                    <motion.div 
                      className={cn(
                        "absolute inset-0 bg-gradient-to-r",
                        currentTheme.accent.primary,
                        "rounded-full blur-lg opacity-50"
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
                      "relative p-3 sm:p-4 rounded-full bg-gradient-to-r",
                      currentTheme.accent.primary,
                      "shadow-lg"
                    )}>
                      {section.icon === "globe" && <Globe className="w-6 h-6 sm:w-8 sm:h-8 text-white" />}
                      {section.icon === "share" && <Share className="w-6 h-6 sm:w-8 sm:h-8 text-white" />}
                      {section.icon === "barChart" && <BarChart className="w-6 h-6 sm:w-8 sm:h-8 text-white" />}
                      {section.icon === "target" && <Target className="w-6 h-6 sm:w-8 sm:h-8 text-white" />}
                      {section.icon === "brain" && <Brain className="w-6 h-6 sm:w-8 sm:h-8 text-white" />}
                      {section.icon === "rocket" && <Rocket className="w-6 h-6 sm:w-8 sm:h-8 text-white" />}
                    </div>
                  </motion.div>
                  <div className="flex-1 space-y-3 sm:space-y-4 text-center sm:text-left">
                    <motion.h3 
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className={cn(
                        "text-xl sm:text-2xl font-bold bg-gradient-to-r",
                        currentTheme.text.accent,
                        "bg-clip-text text-transparent tracking-tight"
                      )}
                    >
                      {section.title}
                    </motion.h3>
                    <motion.p 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: index * 0.1 + 0.2 }}
                      className={cn("text-base sm:text-lg leading-relaxed font-light tracking-wide", currentTheme.text.secondary)}
                    >
                      {section.description}
                    </motion.p>
                    <motion.div 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 + 0.4 }}
                      className={cn(
                        "bg-gradient-to-r",
                        currentTheme.accent.secondary,
                        "p-4 sm:p-6 rounded-lg sm:rounded-xl border border-amber-100/50 shadow-inner"
                      )}
                    >
                      <div className="flex items-center justify-center sm:justify-start space-x-2 text-amber-600 mb-2">
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        >
                          <Sparkles className="w-4 h-4 sm:w-5 sm:h-5" />
                        </motion.div>
                        <span className="font-medium tracking-wide text-sm sm:text-base">Ejemplo Práctico</span>
                      </div>
                      <p className={cn(currentTheme.text.secondary, "tracking-wide text-sm sm:text-base")}>
                        {section.example}
                      </p>
                    </motion.div>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <Button
              onClick={onStartExercises}
              className={cn(
                "w-full bg-gradient-to-r",
                currentTheme.button.primary,
                "text-white py-6 sm:py-8 text-lg sm:text-xl font-medium tracking-wide rounded-xl sm:rounded-2xl",
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
                Comenzar Ejercicios
                <motion.div
                  animate={{ x: [0, 5, 0] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  className="ml-2"
                >
                  <ChevronRight className="h-5 w-5 sm:h-6 sm:w-6 group-hover:translate-x-1 transition-transform" />
                </motion.div>
              </motion.span>
            </Button>
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  );
}    