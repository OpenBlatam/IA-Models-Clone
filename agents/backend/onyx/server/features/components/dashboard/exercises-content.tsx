'use client';

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/shared/icons";
import { cn } from "@/lib/utils";

interface Exercise {
  id: string;
  type: string;
  question: string;
  points: number;
  lesson: {
    title: string;
    difficulty: string;
  };
}

interface ExercisesContentProps {
  exercises: Exercise[];
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

export function ExercisesContent({ exercises, userLevel }: ExercisesContentProps) {
  return (
    <motion.div 
      className="space-y-4 p-8 pt-6"
      variants={container}
      initial="hidden"
      animate="show"
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {exercises.map((exercise) => {
          const isLocked = exercise.lesson.difficulty === "Avanzado" && userLevel < 3;
          
          return (
            <motion.div key={exercise.id} variants={item}>
              <Card className={cn(
                "relative overflow-hidden",
                isLocked && "opacity-50"
              )}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    {exercise.question}
                  </CardTitle>
                  {isLocked ? (
                    <Icons.lock className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <Icons.target className="h-4 w-4 text-primary" />
                  )}
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground mb-4">
                    <Icons.bookOpen className="h-3 w-3" />
                    <span>{exercise.lesson.title}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <Icons.target className="h-3 w-3 text-muted-foreground" />
                      <span>{exercise.type}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Icons.star className="h-3 w-3 text-yellow-500" />
                      <span>{exercise.points} XP</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      <motion.div variants={item} className="mt-8">
        <Card>
          <CardHeader>
            <CardTitle>Consejos para Ejercicios</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <Icons.target className="h-5 w-5 text-primary" />
                <div>
                  <h4 className="text-sm font-medium">Practica Regularmente</h4>
                  <p className="text-xs text-muted-foreground">
                    Completa al menos un ejercicio al día para mantener tu progreso.
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <Icons.trophy className="h-5 w-5 text-yellow-500" />
                <div>
                  <h4 className="text-sm font-medium">Gana XP</h4>
                  <p className="text-xs text-muted-foreground">
                    Cada ejercicio completado te da puntos de experiencia.
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <Icons.star className="h-5 w-5 text-blue-500" />
                <div>
                  <h4 className="text-sm font-medium">Desbloquea Contenido</h4>
                  <p className="text-xs text-muted-foreground">
                    Sube de nivel para acceder a ejercicios más avanzados.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}    