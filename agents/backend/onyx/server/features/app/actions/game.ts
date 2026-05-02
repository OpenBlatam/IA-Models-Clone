'use server';

import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { revalidatePath } from "next/cache";
import { headers, cookies } from "next/headers";

export async function updateProgress(exerciseId: string, points: number) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user?.id) {
      throw new Error("Unauthorized");
    }

    // Get or create user progress
    let userProgress = await prisma.userProgress.findUnique({
      where: { userId: session.user.id },
      include: {
        Lesson: true
      }
    });

    if (!userProgress) {
      userProgress = await prisma.userProgress.create({
        data: {
          userId: session.user.id,
          experience: 0,
          level: 1,
          streak: 0,
          lastActive: new Date(),
        },
        include: {
          Lesson: true
        }
      });
    }

    // Get the lesson ID from the exercise
    const exercise = await prisma.exercise.findUnique({
      where: { id: exerciseId },
      select: { lessonId: true }
    });

    if (!exercise) {
      throw new Error("Exercise not found");
    }

    // Check if the lesson is already completed
    const isLessonCompleted = userProgress.Lesson.some(
      lesson => lesson.id === exercise.lessonId
    );

    // Update experience and level
    const newExperience = userProgress.experience + points;
    const newLevel = Math.floor(newExperience / 1000) + 1;

    // Update streak
    const today = new Date();
    const lastActive = new Date(userProgress.lastActive);
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
        lastActive: today,
        ...(isLessonCompleted ? {} : {
          Lesson: {
            connect: { id: exercise.lessonId }
          }
        })
      },
      include: {
        Lesson: true
      }
    });

    revalidatePath("/dashboard");
    return { 
      success: true,
      completedLessons: updatedProgress.Lesson.length
    };
  } catch (error) {
    console.error("[PROGRESS_UPDATE]", error);
    throw error;
  }
}   