import { getCurrentUser } from "@/lib/session";
import { constructMetadata } from "@/lib/utils";
import { DashboardHeader } from "@/components/dashboard/header";
import { prisma } from "@/lib/db";
import { DashboardContent } from "@/components/dashboard/dashboard-content";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui/skeleton";

export const metadata = constructMetadata({
  title: "Dashboard – Learning Progress",
  description: "Track your learning progress and achievements.",
});

async function getWeeklyProgress(userId: string) {
  const today = new Date();
  const startOfWeek = new Date(today);
  startOfWeek.setDate(today.getDate() - today.getDay());

  const weeklyProgress = await prisma.userProgress.findMany({
    where: {
      userId,
      lastActive: {
        gte: startOfWeek,
      },
    },
    orderBy: {
      lastActive: 'asc',
    },
    select: {
      experience: true,
      lastActive: true,
    },
  });

  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  return days.map(day => {
    const progress = weeklyProgress.find(p => 
      new Date(p.lastActive).toLocaleDateString('en-US', { weekday: 'short' }) === day
    );
    return {
      day,
      xp: progress ? progress.experience : 0,
    };
  });
}

async function UserProgress({ userId }: { userId: string }) {
  const [userProgress, weeklyProgress] = await Promise.all([
    prisma.userProgress.findUnique({
      where: { userId },
      include: {
        achievements: {
          select: {
            id: true,
            title: true,
            description: true,
            icon: true,
            unlockedAt: true,
          },
        },
      },
    }),
    getWeeklyProgress(userId),
  ]);

  const gameStats = {
    currentStreak: userProgress?.streak || 0,
    totalXP: userProgress?.experience || 0,
    level: userProgress?.level || 1,
    lessonsCompleted: userProgress?.achievements?.length || 0,
    dailyGoal: 50,
    dailyProgress: weeklyProgress[new Date().getDay()].xp,
    weeklyProgress,
  };

  return (
    <DashboardContent 
      gameStats={gameStats}
      achievements={userProgress?.achievements?.map(achievement => ({
        ...achievement,
        unlockedAt: achievement.unlockedAt?.toISOString() || null
      })) || []}
    />
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-8">
      <div className="space-y-4">
        <Skeleton className="h-8 w-[250px]" />
        <Skeleton className="h-4 w-[300px]" />
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-[125px]" />
        ))}
      </div>
    </div>
  );
}

export default async function DashboardPage() {
  const user = await getCurrentUser();
  if (!user?.id) return null;

  return (
    <>
      <DashboardHeader
        heading="Learning Dashboard"
        text={`Welcome back, ${user?.name}! Track your progress and achievements.`}
      />
      <Suspense fallback={<DashboardSkeleton />}>
        <UserProgress userId={user.id} />
      </Suspense>
    </>
  );
}
