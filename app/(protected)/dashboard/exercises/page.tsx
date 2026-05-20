import { getCurrentUser } from "@/lib/session";
import { constructMetadata } from "@/lib/utils";
import { DashboardHeader } from "@/components/dashboard/header";
import { ExercisesContent } from "@/components/dashboard/exercises-content";
import { prisma } from "@/lib/db";

export const metadata = constructMetadata({
  title: "Ejercicios | Dashboard",
  description: "Practica y mejora tus habilidades con ejercicios interactivos.",
});

const aiMarketingExercises = [
  {
    id: "ai-marketing-ex-1",
    type: "Caso Práctico",
    question: "Analiza una campaña de marketing real utilizando herramientas de IA para identificar patrones y oportunidades.",
    points: 50,
    lessonId: "ai-marketing-1",
    correctAnswer: "Análisis de datos y patrones de comportamiento",
    options: [
      "Análisis de datos y patrones de comportamiento",
      "Diseño de logos",
      "Fotografía de producto",
      "Redacción de emails"
    ]
  },
  {
    id: "ai-marketing-ex-2",
    type: "Práctica",
    question: "Crea contenido optimizado para diferentes plataformas usando herramientas de IA.",
    points: 75,
    lessonId: "ai-marketing-1",
    correctAnswer: "Generación de contenido adaptado",
    options: [
      "Generación de contenido adaptado",
      "Diseño de packaging",
      "Gestión de inventario",
      "Atención al cliente"
    ]
  },
  {
    id: "ai-marketing-ex-3",
    type: "Proyecto",
    question: "Utiliza análisis predictivo para optimizar el ROI de una campaña de marketing.",
    points: 100,
    lessonId: "ai-marketing-1",
    correctAnswer: "Optimización de presupuesto y segmentación",
    options: [
      "Optimización de presupuesto y segmentación",
      "Diseño de folletos",
      "Organización de eventos",
      "Gestión de redes sociales"
    ]
  },
  {
    id: "ai-marketing-ex-4",
    type: "Caso Práctico",
    question: "Implementa técnicas de IA para segmentar y personalizar mensajes para diferentes audiencias.",
    points: 85,
    lessonId: "ai-marketing-1",
    correctAnswer: "Personalización de mensajes por segmento",
    options: [
      "Personalización de mensajes por segmento",
      "Diseño de banners",
      "Fotografía de producto",
      "Redacción de blogs"
    ]
  }
];

export default async function ExercisesPage() {
  const user = await getCurrentUser();

  if (!user) {
    return null;
  }

  const userProgress = await prisma.userProgress.findUnique({
    where: {
      userId: user.id,
    },
  });

  // Create AI Marketing lesson if it doesn't exist
  const aiMarketingLesson = await prisma.lesson.upsert({
    where: { id: "ai-marketing-1" },
    update: {},
    create: {
      id: "ai-marketing-1",
      title: "Marketing con IA",
      description: "Aprende a utilizar la inteligencia artificial para mejorar tus estrategias de marketing.",
      difficulty: "Intermedio",
      category: "Marketing",
      requiredLevel: 2,
      experienceReward: 150
    }
  });

  // Create exercises if they don't exist
  for (const exercise of aiMarketingExercises) {
    await prisma.exercise.upsert({
      where: { id: exercise.id },
      update: {},
      create: {
        id: exercise.id,
        type: exercise.type,
        question: exercise.question,
        correctAnswer: exercise.correctAnswer,
        options: exercise.options,
        points: exercise.points,
        lessonId: exercise.lessonId
      }
    });
  }

  const exercises = await prisma.exercise.findMany({
    include: {
      lesson: {
        select: {
          title: true,
          difficulty: true,
        },
      },
    },
  });

  return (
    <div className="flex flex-col gap-8 pb-8">
      <DashboardHeader
        heading="Ejercicios"
        text="Practica y mejora tus habilidades con ejercicios interactivos."
      />
      <ExercisesContent 
        exercises={exercises} 
        userLevel={userProgress?.level || 1} 
      />
    </div>
  );
} 