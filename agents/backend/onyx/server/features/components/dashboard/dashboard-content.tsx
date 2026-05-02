'use client';

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Icons } from "@/components/shared/icons";
import { motion } from "framer-motion";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { useState } from "react";

interface GameStats {
  currentStreak: number;
  totalXP: number;
  level: number;
  lessonsCompleted: number;
  dailyGoal: number;
  dailyProgress: number;
  weeklyProgress: Array<{
    day: string;
    xp: number;
  }>;
}

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  unlockedAt: string | null;
}

interface DashboardContentProps {
  gameStats: GameStats;
  achievements: Achievement[];
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

const chartConfig = {
  xp: {
    label: "XP Earned",
    theme: {
      light: "hsl(var(--primary))",
      dark: "hsl(var(--primary))",
    },
  },
  goal: {
    label: "Daily Goal",
    theme: {
      light: "hsl(var(--muted))",
      dark: "hsl(var(--muted))",
    },
  },
};

export function DashboardContent({ gameStats, achievements }: DashboardContentProps) {
  const [globalScore, setGlobalScore] = useState(0);
  const [feedback, setFeedback] = useState<string | null>(null);

  const handleAnswer = (answer: string, correctAnswer: string) => {
    if (answer === correctAnswer) {
      setGlobalScore(globalScore + 25);
      setFeedback("¡Correcto! Has ganado 25 XP.");
    } else {
      setFeedback(`Incorrecto. La respuesta correcta es: ${correctAnswer}`);
    }
  };

  const chartData = gameStats.weeklyProgress.map(day => ({
    ...day,
    goal: gameStats.dailyGoal,
  }));

  return (
    <motion.div 
      className="space-y-4 p-8 pt-6"
      variants={container}
      initial="hidden"
      animate="show"
    >
      {/* Stats Cards */}
      <motion.div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <motion.div variants={item}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Current Streak</CardTitle>
              <Icons.flame className="h-4 w-4 text-orange-500" />
            </CardHeader>
            <CardContent>
              <motion.div 
                className="text-2xl font-bold"
                initial={{ scale: 0.5 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200, damping: 10 }}
              >
                {gameStats.currentStreak} days
              </motion.div>
              <p className="text-xs text-muted-foreground">
                Keep it up! 🔥
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total XP</CardTitle>
              <Icons.star className="h-4 w-4 text-yellow-500" />
            </CardHeader>
            <CardContent>
              <motion.div 
                className="text-2xl font-bold"
                initial={{ scale: 0.5 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200, damping: 10 }}
              >
                {gameStats.totalXP} XP
              </motion.div>
              <p className="text-xs text-muted-foreground">
                Level {gameStats.level}
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Lessons Completed</CardTitle>
              <Icons.trophy className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <motion.div 
                className="text-2xl font-bold"
                initial={{ scale: 0.5 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200, damping: 10 }}
              >
                {gameStats.lessonsCompleted}
              </motion.div>
              <p className="text-xs text-muted-foreground">
                Great progress!
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Daily Goal</CardTitle>
              <Icons.target className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <motion.div 
                className="text-2xl font-bold"
                initial={{ scale: 0.5 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200, damping: 10 }}
              >
                {gameStats.dailyProgress}/{gameStats.dailyGoal} XP
              </motion.div>
              <Progress 
                value={(gameStats.dailyProgress / gameStats.dailyGoal) * 100} 
                className="mt-2"
              />
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>

      {/* Weekly Progress Chart */}
      <motion.div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7" variants={item}>
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Weekly Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis 
                    dataKey="day" 
                    className="text-xs text-muted-foreground"
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    className="text-xs text-muted-foreground"
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) => `${value} XP`}
                  />
                  <ChartTooltip
                    content={
                      <ChartTooltipContent
                        labelFormatter={(label) => `${label}`}
                        formatter={(value, name) => [
                          `${value} XP`,
                          name === "xp" ? "XP Earned" : "Daily Goal",
                        ]}
                      />
                    }
                  />
                  <Bar
                    dataKey="xp"
                    className="fill-primary"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="goal"
                    className="fill-muted"
                    radius={[4, 4, 0, 0]}
                    opacity={0.3}
                  />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>

        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>Recent Achievements</CardTitle>
          </CardHeader>
          <CardContent>
            <motion.div 
              className="space-y-4"
              variants={container}
              initial="hidden"
              animate="show"
            >
              {achievements.slice(0, 3).map((achievement, index) => {
                let IconComponent = Icons.trophy;
                if (achievement.icon === 'star') IconComponent = Icons.star;
                if (achievement.icon === 'flame') IconComponent = Icons.flame;
                if (achievement.icon === 'target') IconComponent = Icons.target;
                
                return (
                  <motion.div 
                    key={achievement.id} 
                    className="flex items-center"
                    variants={item}
                    initial={{ x: -20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <IconComponent className="mr-2 h-4 w-4 text-yellow-500" />
                    <div>
                      <p className="text-sm font-medium">{achievement.title}</p>
                      <p className="text-xs text-muted-foreground">{achievement.description}</p>
                    </div>
                  </motion.div>
                );
              })}
              {(!achievements || achievements.length === 0) && (
                <motion.p 
                  className="text-sm text-muted-foreground"
                  variants={item}
                >
                  No achievements yet. Keep learning!
                </motion.p>
              )}
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>

      <Card className="col-span-4">
        <CardHeader>
          <CardTitle>Experiencia Global</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mt-2 text-sm">Experiencia Global: {globalScore} XP</p>
        </CardContent>
      </Card>
    </motion.div>
  );
}    