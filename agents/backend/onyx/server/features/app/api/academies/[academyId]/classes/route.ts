import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { redis } from "@/lib/redis";
import { getCurrentUser } from "@/lib/session";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ academyId: string }> }
) {
  try {
    const resolvedParams = await params;
    const academyId = resolvedParams.academyId;
    const user = await getCurrentUser();
    const cacheKey = `academy:${academyId}:classes`;

    // Try Redis first
    const cached = await redis.get(cacheKey);
    if (cached) {
      return NextResponse.json(JSON.parse(cached));
    }

    // If not cached, fetch from DB
    const classes = await prisma.academyClass.findMany({
      where: { academyId },
      orderBy: { order: "asc" },
      include: {
        academy: {
          select: {
            s3Config: true,
          },
        },
      },
    });

    // If user is logged in, get their progress
    if (user) {
      const progress = await prisma.academyProgress.findUnique({
        where: {
          userId_academyId: {
            userId: user.id,
            academyId,
          },
        },
      });

      if (progress) {
        const classesWithProgress = classes.map((classItem) => ({
          ...classItem,
          isCompleted: progress.completedClasses.includes(classItem.id),
          progress: progress.completedClasses.includes(classItem.id) ? 100 : 0,
        }));

        // Cache the result
        await redis.set(cacheKey, JSON.stringify(classesWithProgress), "EX", 300);
        return NextResponse.json(classesWithProgress);
      }
    }

    // Cache the result
    await redis.set(cacheKey, JSON.stringify(classes), "EX", 300);
    return NextResponse.json(classes);
  } catch (error) {
    console.error("[ACADEMY_CLASSES_GET]", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}   