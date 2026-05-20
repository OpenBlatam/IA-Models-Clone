import { motion } from "framer-motion";
import { Icons } from "@/components/shared/icons";
import { cn } from "@/lib/utils";

export interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  unlockedAt: string | null;
}

interface AchievementProps {
  achievement: Achievement;
  className?: string;
}

export function Achievement({ achievement, className }: AchievementProps) {
  const isUnlocked = achievement.unlockedAt !== null;
  const Icon = Icons[achievement.icon as keyof typeof Icons] || Icons.trophy;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        "p-4 rounded-lg border transition-colors",
        isUnlocked
          ? "bg-white border-green-200"
          : "bg-gray-50 border-gray-200",
        className
      )}
    >
      <div className="flex items-start space-x-4">
        <div
          className={cn(
            "p-2 rounded-full",
            isUnlocked ? "bg-green-100" : "bg-gray-200"
          )}
        >
          {isUnlocked ? (
            <Icon className="w-6 h-6 text-green-600" />
          ) : (
            <Icons.lock className="w-6 h-6 text-gray-400" />
          )}
        </div>
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900">{achievement.title}</h4>
          <p className="text-sm text-gray-500">{achievement.description}</p>
          {isUnlocked && achievement.unlockedAt && (
            <p className="mt-1 text-xs text-gray-400">
              Desbloqueado el{" "}
              {new Date(achievement.unlockedAt).toLocaleDateString()}
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
}

interface AchievementListProps {
  achievements: Achievement[];
  className?: string;
}

export function AchievementList({ achievements, className }: AchievementListProps) {
  return (
    <div className={cn("space-y-4", className)}>
      <h3 className="text-lg font-semibold text-gray-900">Logros</h3>
      <div className="grid gap-4 sm:grid-cols-2">
        {achievements.map((achievement) => (
          <Achievement
            key={achievement.id}
            achievement={achievement}
          />
        ))}
      </div>
    </div>
  );
}    