import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { redis } from "@/lib/redis";
import { getCurrentUser } from "@/lib/session";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ videoId: string }> }
) {
  try {
    const resolvedParams = await params;
    const videoId = resolvedParams.videoId;
    const cacheKey = `video:${videoId}:comments`;

    // Try Redis first
    const cached = await redis.get(cacheKey);
    if (cached) {
      return NextResponse.json(JSON.parse(cached));
    }

    // If not cached, fetch from DB
    const comments = await prisma.comment.findMany({
      where: { videoId },
      orderBy: { createdAt: "desc" },
      include: { user: true },
    });

    // Cache the result
    await redis.set(cacheKey, JSON.stringify(comments), "EX", 300); // Cache for 5 minutes

    return NextResponse.json(comments);
  } catch (error) {
    console.error("[VIDEO_COMMENTS_GET]", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ videoId: string }> }
) {
  try {
    const user = await getCurrentUser();
    if (!user) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }

    const resolvedParams = await params;
    const videoId = resolvedParams.videoId;
    const { content } = await req.json();

    if (!content) {
      return NextResponse.json(
        { error: "Content is required" },
        { status: 400 }
      );
    }

    const comment = await prisma.comment.create({
      data: {
        content,
        videoId,
        userId: user.id,
      },
      include: {
        user: true,
      },
    });

    // Invalidate cache
    await redis.del(`video:${videoId}:comments`);

    return NextResponse.json(comment);
  } catch (error) {
    console.error("[VIDEO_COMMENTS_POST]", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}   