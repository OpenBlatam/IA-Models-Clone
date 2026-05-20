import { prisma } from "@/lib/db";

export async function createNotification({
  userId,
  type,
  title,
  message,
}: {
  userId: string;
  type: string;
  title: string;
  message: string;
}) {
  try {
    const notification = await prisma.notification.create({
      data: {
        userId,
        type,
        title,
        message,
      },
    });
    return notification;
  } catch (error) {
    console.error("Error creating notification:", error);
    return null;
  }
}

export async function createLessonCompletedNotification({
  userId,
  lessonTitle,
}: {
  userId: string;
  lessonTitle: string;
}) {
  return createNotification({
    userId,
    type: "LESSON_COMPLETED",
    title: "¡Lección Completada!",
    message: `Has completado la lección "${lessonTitle}". ¡Buen trabajo!`,
  });
}

export async function createAchievementUnlockedNotification({
  userId,
  achievementTitle,
}: {
  userId: string;
  achievementTitle: string;
}) {
  return createNotification({
    userId,
    type: "ACHIEVEMENT_UNLOCKED",
    title: "¡Logro Desbloqueado!",
    message: `Has desbloqueado el logro "${achievementTitle}". ¡Felicidades!`,
  });
}

export async function createLevelUpNotification({
  userId,
  newLevel,
}: {
  userId: string;
  newLevel: number;
}) {
  return createNotification({
    userId,
    type: "LEVEL_UP",
    title: "¡Subiste de Nivel!",
    message: `¡Felicidades! Has alcanzado el nivel ${newLevel}.`,
  });
} 