'use client';

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/shared/icons";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  unlockedAt: Date | null;
  progress?: number;
}

interface AchievementsContentProps {
  achievements: Achievement[];
  userLevel: number;
  totalExperience: number;
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

const getIconComponent = (iconName: string) => {
  switch (iconName) {
    case 'trophy':
      return Icons.trophy;
    case 'star':
      return Icons.star;
    case 'target':
      return Icons.target;
    case 'flame':
      return Icons.flame;
    case 'sparkles':
      return Icons.sparkles;
    default:
      return Icons.trophy;
  }
};

export function AchievementsContent({ 
  achievements, 
  userLevel,
  totalExperience 
}: AchievementsContentProps) {
  const unlockedCount = achievements.filter(a => a.unlockedAt).length;
  const progress = (unlockedCount / achievements.length) * 100;

  return (
    <motion.div 
      className="space-y-4 p-8 pt-6"
      variants={container}
      initial="hidden"
      animate="show"
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {achievements.map((achievement) => {
          const IconComponent = getIconComponent(achievement.icon);
          const isUnlocked = !!achievement.unlockedAt;
          
          return (
            <motion.div key={achievement.id} variants={item}>
              <Card className={cn(
                "relative overflow-hidden",
                !isUnlocked && "opacity-50"
              )}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    {achievement.title}
                  </CardTitle>
                  <IconComponent className={cn(
                    "h-4 w-4",
                    isUnlocked ? "text-yellow-500" : "text-muted-foreground"
                  )} />
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground mb-4">
                    {achievement.description}
                  </p>
                  {achievement.progress !== undefined && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span>Progreso</span>
                        <span>{achievement.progress}%</span>
                      </div>
                      <Progress value={achievement.progress} className="h-1" />
                    </div>
                  )}
                  {isUnlocked && (
                    <div className="mt-2 text-xs text-muted-foreground">
                      Desbloqueado el {new Date(achievement.unlockedAt!).toLocaleDateString()}
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      <motion.div variants={item} className="mt-8">
        <Card>
          <CardHeader>
            <CardTitle>Resumen de Logros</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Icons.trophy className="h-5 w-5 text-yellow-500" />
                  <span className="text-sm font-medium">Logros Desbloqueados</span>
                </div>
                <span className="text-sm font-medium">
                  {unlockedCount} / {achievements.length}
                </span>
              </div>
              <Progress value={progress} className="h-2" />
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Icons.star className="h-5 w-5 text-blue-500" />
                  <span className="text-sm font-medium">Nivel Actual</span>
                </div>
                <span className="text-sm font-medium">Nivel {userLevel}</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Icons.flame className="h-5 w-5 text-orange-500" />
                  <span className="text-sm font-medium">Experiencia Total</span>
                </div>
                <span className="text-sm font-medium">{totalExperience} XP</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}    