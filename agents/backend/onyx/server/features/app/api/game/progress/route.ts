import { NextResponse } from "next/server";
import { auth } from "@/auth";
import { prisma } from "@/lib/db";

export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    const userProgress = await prisma.userProgress.findUnique({
      where: { userId: session.user.id },
      include: {
        achievements: true,
      },
    });

    if (!userProgress) {
      return NextResponse.json({
        experience: 0,
        level: 1,
        streak: 0,
        achievements: [],
      });
    }

    return NextResponse.json(userProgress);
  } catch (error) {
    console.error("[PROGRESS_GET]", error);
    return new NextResponse("Internal error", { status: 500 });
  }
}

export async function POST(req: Request) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    const body = await req.json();
    const { exerciseId, points } = body;

    if (!exerciseId || typeof points !== "number") {
      return new NextResponse("Invalid request", { status: 400 });
    }

    // Get current user progress
    const userProgress = await prisma.userProgress.findUnique({
      where: { userId: session.user.id },
    });

    if (!userProgress) {
      // Create new user progress if it doesn't exist
      const newProgress = await prisma.userProgress.create({
        data: {
          userId: session.user.id,
          experience: points,
          level: 1,
          streak: 1,
          lastActive: new Date(),
        },
      });
      return NextResponse.json(newProgress);
    }

    // Calculate new experience and level
    const newExperience = userProgress.experience + points;
    const newLevel = Math.floor(newExperience / 1000) + 1;

    // Update streak
    const lastActive = new Date(userProgress.lastActive);
    const today = new Date();
    const isConsecutiveDay =
      lastActive.getDate() === today.getDate() - 1 &&
      lastActive.getMonth() === today.getMonth() &&
      lastActive.getFullYear() === today.getFullYear();

    const newStreak = isConsecutiveDay ? userProgress.streak + 1 : 1;

    // Update user progress
    const updatedProgress = await prisma.userProgress.update({
      where: { userId: session.user.id },
      data: {
        experience: newExperience,
        level: newLevel,
        streak: newStreak,
        lastActive: new Date(),
      },
    });

    // Check for achievements
    const achievements = await checkAchievements(session.user.id, updatedProgress);

    return NextResponse.json({
      progress: updatedProgress,
      achievements,
    });
  } catch (error) {
    console.error("[PROGRESS_UPDATE]", error);
    return new NextResponse("Internal error", { status: 500 });
  }
}

async function checkAchievements(userId: string, progress: any) {
  const newAchievements: any[] = [];

  // Level achievements
  if (progress.level >= 5) {
    const level5Achievement = await prisma.achievement.findFirst({
      where: {
        userProgressId: progress.id,
        title: "Level 5",
      },
    });

    if (!level5Achievement) {
      newAchievements.push(
        await prisma.achievement.create({
          data: {
            title: "Level 5",
            description: "¡Has alcanzado el nivel 5!",
            icon: "trophy",
            unlockedAt: new Date(),
            updatedAt: new Date(),
            userProgressId: progress.id,
          },
        })
      );
    }
  }

  // Streak achievements
  if (progress.streak >= 7) {
    const weekStreakAchievement = await prisma.achievement.findFirst({
      where: {
        userProgressId: progress.id,
        title: "7 Day Streak",
      },
    });

    if (!weekStreakAchievement) {
      newAchievements.push(
        await prisma.achievement.create({
          data: {
            title: "7 Day Streak",
            description: "¡Has mantenido una racha de 7 días!",
            icon: "trophy",
            unlockedAt: new Date(),
            updatedAt: new Date(),
            userProgressId: progress.id,
          },
        })
      );
    }
  }

  return newAchievements;
}       