import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const videoId = searchParams.get("videoId");
    const courseId = searchParams.get("courseId");

    if (!videoId || !courseId) {
      return NextResponse.json(
        { error: "Se requieren videoId y courseId" },
        { status: 400 }
      );
    }

    const questions = await prisma.question.findMany({
      where: {
        videoId,
        courseId,
      },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            image: true,
          },
        },
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return NextResponse.json(questions);
  } catch (error) {
    console.error("Error al obtener preguntas:", error);
    return NextResponse.json(
      { error: "Error al obtener preguntas" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json(
        { error: "No autorizado" },
        { status: 401 }
      );
    }

    const { content, videoId, courseId } = await request.json();

    if (!content || !videoId || !courseId) {
      return NextResponse.json(
        { error: "Se requieren content, videoId y courseId" },
        { status: 400 }
      );
    }

    const question = await prisma.question.create({
      data: {
        content,
        videoId,
        courseId,
        userId: session.user.id,
        status: "pending",
      },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            image: true,
          },
        },
      },
    });

    // Iniciar el proceso de respuesta en segundo plano
    processQuestion(question.id, content, videoId, courseId).catch(console.error);

    return NextResponse.json(question);
  } catch (error) {
    console.error("Error al crear pregunta:", error);
    return NextResponse.json(
      { error: "Error al crear pregunta" },
      { status: 500 }
    );
  }
}

async function processQuestion(
  questionId: string,
  content: string,
  videoId: string,
  courseId: string
) {
  try {
    // Obtener respuesta de DeepSeek
    const deepseekResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/deepseek`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        question: content,
        context: `Video ID: ${videoId}, Course ID: ${courseId}`,
      }),
    });

    if (!deepseekResponse.ok) {
      throw new Error("Error al obtener respuesta de DeepSeek");
    }

    const { answer } = await deepseekResponse.json();

    // Actualizar la pregunta con la respuesta
    await prisma.question.update({
      where: { id: questionId },
      data: {
        answer,
        status: "answered",
      },
    });
  } catch (error) {
    console.error("Error al procesar pregunta:", error);
    await prisma.question.update({
      where: { id: questionId },
      data: {
        status: "error",
      },
    });
  }
} 