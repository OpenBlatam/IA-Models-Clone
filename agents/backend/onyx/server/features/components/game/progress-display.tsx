"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Icons } from "@/components/shared/icons";
import { AchievementList } from "./achievement";

interface UserProgress {
  experience: number;
  level: number;
  streak: number;
  achievements: Array<{
    id: string;
    title: string;
    description: string;
    icon: string;
    unlockedAt: string | null;
  }>;
}

export function ProgressDisplay() {
  const [progress, setProgress] = useState<UserProgress | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProgress = async () => {
      try {
        const response = await fetch("/api/game/progress");
        if (!response.ok) throw new Error("Failed to fetch progress");
        const data = await response.json();
        setProgress(data);
      } catch (error) {
        console.error("Failed to fetch user progress data");
      } finally {
        setLoading(false);
      }
    };

    fetchProgress();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Icons.spinner className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (!progress) {
    return (
      <div className="text-center p-8">
        <p className="text-muted-foreground">No progress data available</p>
      </div>
    );
  }

  const experienceToNextLevel = progress.level * 1000 - progress.experience;
  const progressPercentage = ((progress.experience % 1000) / 1000) * 100;

  return (
    <div className="space-y-8 p-6">
      <div className="grid gap-6 md:grid-cols-3">
        {/* Level Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-lg border bg-card p-6 shadow-sm"
        >
          <div className="flex items-center gap-4">
            <div className="rounded-full bg-primary/10 p-3">
              <Icons.star className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Level {progress.level}</h3>
              <p className="text-sm text-muted-foreground">
                {experienceToNextLevel} XP to next level
              </p>
            </div>
          </div>
          <div className="mt-4 h-2 w-full rounded-full bg-secondary">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${progressPercentage}%` }}
              className="h-full rounded-full bg-primary"
            />
          </div>
        </motion.div>

        {/* Experience Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="rounded-lg border bg-card p-6 shadow-sm"
        >
          <div className="flex items-center gap-4">
            <div className="rounded-full bg-primary/10 p-3">
              <Icons.sparkles className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Experience</h3>
              <p className="text-sm text-muted-foreground">
                {progress.experience} XP total
              </p>
            </div>
          </div>
        </motion.div>

        {/* Streak Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="rounded-lg border bg-card p-6 shadow-sm"
        >
          <div className="flex items-center gap-4">
            <div className="rounded-full bg-primary/10 p-3">
              <Icons.flame className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Current Streak</h3>
              <p className="text-sm text-muted-foreground">
                {progress.streak} days
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Achievements Section */}
      <div className="mt-8">
        <h2 className="mb-4 text-2xl font-bold">Achievements</h2>
        <AchievementList achievements={progress.achievements} />
      </div>
    </div>
  );
}              