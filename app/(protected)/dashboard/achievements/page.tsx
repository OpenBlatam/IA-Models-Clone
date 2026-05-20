import { getCurrentUser } from "@/lib/session";
import { constructMetadata } from "@/lib/utils";
import { DashboardHeader } from "@/components/dashboard/header";
import { AchievementsContent } from "@/components/dashboard/achievements-content";
import { prisma } from "@/lib/db";

export const metadata = constructMetadata({
  title: "Logros | Dashboard",
  description: "Revisa tus logros y progreso en la plataforma.",
});

export default async function AchievementsPage() {
  const user = await getCurrentUser();

  if (!user) {
    return null;
  }

  const userProgress = await prisma.userProgress.findUnique({
    where: {
      userId: user.id,
    },
    include: {
      achievements: true,
    },
  });

  // Define default achievements if none exist
  const defaultAchievements = [
    {
      id: "first-lesson",
      title: "Primera Lección",
      description: "Completa tu primera lección en la plataforma.",
      icon: "star",
      progress: (userProgress?.level ?? 0) > 0 ? 100 : 0,
    },
    {
      id: "streak-3",
      title: "Racha de 3 Días",
      description: "Mantén una racha de actividad de 3 días consecutivos.",
      icon: "flame",
      progress: Math.min((userProgress?.streak || 0) / 3 * 100, 100),
    },
    {
      id: "level-5",
      title: "Experto en Crecimiento",
      description: "Alcanza el nivel 5 en la plataforma.",
      icon: "trophy",
      progress: Math.min((userProgress?.level || 0) / 5 * 100, 100),
    },
    {
      id: "complete-10",
      title: "Aprendiz Dedicado",
      description: "Completa 10 ejercicios en la plataforma.",
      icon: "target",
      progress: 0, // This would need to be calculated based on completed exercises
    },
    {
      id: "perfect-score",
      title: "Puntuación Perfecta",
      description: "Obtén una puntuación perfecta en cualquier ejercicio.",
      icon: "sparkles",
      progress: 0, // This would need to be calculated based on exercise scores
    },
  ];

  // Merge default achievements with user's unlocked achievements
  const achievements = defaultAchievements.map(achievement => {
    const unlockedAchievement = userProgress?.achievements.find(
      a => a.id === achievement.id
    );
    return {
      ...achievement,
      unlockedAt: unlockedAchievement?.unlockedAt || null,
    };
  });

  return (
    <div className="flex flex-col gap-8 pb-8">
      <DashboardHeader
        heading="Logros"
        text="Revisa tus logros y progreso en la plataforma."
      />
      <AchievementsContent 
        achievements={achievements}
        userLevel={userProgress?.level || 1}
        totalExperience={userProgress?.experience || 0}
      />
    </div>
  );
}   