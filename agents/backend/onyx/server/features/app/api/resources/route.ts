import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redis } from "@/lib/redis";
import { prisma } from "@/lib/prisma";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const videoId = searchParams.get("videoId");
    const courseId = searchParams.get("courseId");

    if (!videoId || !courseId) {
      return NextResponse.json(
        { error: "Missing required parameters" },
        { status: 400 }
      );
    }

    // Try to get from cache first
    const cacheKey = `resources:${videoId}:${courseId}`;
    const cachedData = await redis.get(cacheKey);
    if (cachedData) {
      return NextResponse.json(JSON.parse(cachedData));
    }

    // If not in cache, fetch from database
    const resources = await prisma.resource.findMany({
      where: {
        videoId,
        video: {
          courseId,
        },
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    // Cache the results for 5 minutes
    await redis.set(cacheKey, JSON.stringify(resources), "EX", 300);

    return NextResponse.json(resources);
  } catch (error) {
    console.error("Error fetching resources:", error);
    return NextResponse.json(
      { error: "Error fetching resources" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }

    const body = await request.json();
    const { title, description, url, type, videoId, courseId } = body;

    if (!title || !url || !videoId || !courseId) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    // Verify that the video belongs to the specified course
    const video = await prisma.video.findFirst({
      where: {
        id: videoId,
        courseId,
      },
    });

    if (!video) {
      return NextResponse.json(
        { error: "Video not found or does not belong to the specified course" },
        { status: 404 }
      );
    }

    const resource = await prisma.resource.create({
      data: {
        title,
        description,
        url,
        type: type || "file",
        videoId,
      },
    });

    // Invalidate cache
    const cacheKey = `resources:${videoId}:${courseId}`;
    await redis.del(cacheKey);

    return NextResponse.json(resource);
  } catch (error) {
    console.error("Error creating resource:", error);
    return NextResponse.json(
      { error: "Error creating resource" },
      { status: 500 }
    );
  }
} 