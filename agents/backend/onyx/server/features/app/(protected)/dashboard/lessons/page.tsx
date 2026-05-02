import { authOptions } from "@/auth";
import { getServerSession } from "next-auth";
import { prisma } from "@/lib/db";
import { DashboardHeader } from "@/components/dashboard/header";
import { LessonsContent } from "@/components/dashboard/lessons-content";
import { Suspense } from "react";
import { redirect } from "next/navigation";
import { Exercise, Lesson } from "@/components/dashboard/lessons/types";
import { Prisma } from "@prisma/client";

// Marketing lesson exercises
const marketingExercises: Exercise[] = [
  {
    id: "marketing-1",
    type: "multiple_choice",
    question: "¿Cuál es la principal ventaja del marketing digital sobre el marketing tradicional?",
    options: [
      "Mayor presupuesto requerido",
      "Mayor segmentación y medición de resultados",
      "Menor alcance de audiencia",
      "Menor interacción con los clientes"
    ],
    correctAnswer: "Mayor segmentación y medición de resultados",
    points: 20,
    lessonId: "marketing"
  },
  {
    id: "marketing-2",
    type: "multiple_choice",
    question: "¿Cuál de los siguientes NO es un canal de marketing digital?",
    options: [
      "Email marketing",
      "SEO",
      "Vallas publicitarias",
      "Redes sociales"
    ],
    correctAnswer: "Vallas publicitarias",
    points: 20,
    lessonId: "marketing"
  },
  {
    id: "marketing-3",
    type: "multiple_choice",
    question: "¿Qué significa CTR en marketing digital?",
    options: [
      "Click Through Rate",
      "Conversion Tracking Rate",
      "Customer Total Revenue",
      "Content Tracking Ratio"
    ],
    correctAnswer: "Click Through Rate",
    points: 20,
    lessonId: "marketing"
  }
];

// Brand Positioning lesson exercises
const brandPositioningExercises: Exercise[] = [
  {
    id: "brand-1",
    type: "multiple_choice",
    question: "¿Cuál es el objetivo principal del posicionamiento de marca?",
    options: [
      "Crear una percepción única en la mente del consumidor",
      "Aumentar las ventas inmediatamente",
      "Reducir los costos de marketing",
      "Mejorar el diseño del producto"
    ],
    correctAnswer: "Crear una percepción única en la mente del consumidor",
    points: 15,
    lessonId: "brand-positioning"
  },
  {
    id: "brand-2",
    type: "multiple_choice",
    question: "¿Qué elemento es fundamental para un buen posicionamiento de marca?",
    options: [
      "Diferenciación clara de la competencia",
      "Precios más bajos que la competencia",
      "Mayor presencia en redes sociales",
      "Más productos que la competencia"
    ],
    correctAnswer: "Diferenciación clara de la competencia",
    points: 15,
    lessonId: "brand-positioning"
  },
  {
    id: "brand-3",
    type: "multiple_choice",
    question: "¿Cómo se mantiene un posicionamiento de marca efectivo?",
    options: [
      "Consistencia en todos los puntos de contacto",
      "Cambiando frecuentemente la estrategia",
      "Copiando a la competencia",
      "Enfocándose solo en el precio"
    ],
    correctAnswer: "Consistencia en todos los puntos de contacto",
    points: 15,
    lessonId: "brand-positioning"
  },
  {
    id: "brand-4",
    type: "multiple_choice",
    question: "¿Qué es la propuesta de valor única (UVP)?",
    options: [
      "La razón por la que los clientes deberían elegir tu marca",
      "El precio más bajo del mercado",
      "La cantidad de productos ofrecidos",
      "El número de seguidores en redes sociales"
    ],
    correctAnswer: "La razón por la que los clientes deberían elegir tu marca",
    points: 15,
    lessonId: "brand-positioning"
  },
  {
    id: "brand-5",
    type: "multiple_choice",
    question: "¿Cuál es la mejor estrategia para posicionar una marca en un mercado saturado?",
    options: [
      "Encontrar un nicho específico y diferenciarse",
      "Bajar los precios para competir",
      "Copiar a la competencia líder",
      "Aumentar el presupuesto de publicidad"
    ],
    correctAnswer: "Encontrar un nicho específico y diferenciarse",
    points: 15,
    lessonId: "brand-positioning"
  }
];

// Market Segmentation lesson exercises
const marketSegmentationExercises: Exercise[] = [
  {
    id: "segmentation-1",
    type: "multiple_choice",
    question: "¿Cuál es el objetivo principal de la segmentación de mercado?",
    options: [
      "Identificar grupos de consumidores con necesidades similares",
      "Reducir los costos de marketing",
      "Aumentar los precios de los productos",
      "Simplificar la producción"
    ],
    correctAnswer: "Identificar grupos de consumidores con necesidades similares",
    points: 15,
    lessonId: "market-segmentation"
  },
  {
    id: "segmentation-2",
    type: "multiple_choice",
    question: "¿Qué tipo de segmentación se basa en el estilo de vida y valores de los consumidores?",
    options: [
      "Segmentación psicográfica",
      "Segmentación demográfica",
      "Segmentación geográfica",
      "Segmentación conductual"
    ],
    correctAnswer: "Segmentación psicográfica",
    points: 15,
    lessonId: "market-segmentation"
  },
  {
    id: "segmentation-3",
    type: "multiple_choice",
    question: "¿Cuál es una ventaja clave de la segmentación de mercado?",
    options: [
      "Permite personalizar mensajes y ofertas",
      "Reduce la competencia",
      "Aumenta los precios",
      "Simplifica la producción"
    ],
    correctAnswer: "Permite personalizar mensajes y ofertas",
    points: 15,
    lessonId: "market-segmentation"
  },
  {
    id: "segmentation-4",
    type: "multiple_choice",
    question: "¿Qué tipo de segmentación se enfoca en el comportamiento de compra?",
    options: [
      "Segmentación conductual",
      "Segmentación demográfica",
      "Segmentación geográfica",
      "Segmentación psicográfica"
    ],
    correctAnswer: "Segmentación conductual",
    points: 15,
    lessonId: "market-segmentation"
  },
  {
    id: "segmentation-5",
    type: "multiple_choice",
    question: "¿Cuál es el primer paso en el proceso de segmentación de mercado?",
    options: [
      "Identificar las variables de segmentación relevantes",
      "Lanzar una campaña de marketing",
      "Establecer precios",
      "Diseñar el producto"
    ],
    correctAnswer: "Identificar las variables de segmentación relevantes",
    points: 15,
    lessonId: "market-segmentation"
  }
];

// Web Analytics lesson exercises
const webAnalyticsExercises: Exercise[] = [
  {
    id: "analytics-1",
    type: "multiple_choice",
    question: "¿Qué es un usuario único en analítica web?",
    options: [
      "Un visitante individual contado una vez en un período específico",
      "Un usuario que nunca ha visitado el sitio",
      "Un usuario que siempre usa el mismo dispositivo",
      "Un usuario que completa una compra"
    ],
    correctAnswer: "Un visitante individual contado una vez en un período específico",
    points: 20,
    lessonId: "web-analytics"
  },
  {
    id: "analytics-2",
    type: "multiple_choice",
    question: "¿Qué mide la tasa de rebote?",
    options: [
      "El porcentaje de visitantes que abandonan el sitio sin interactuar",
      "La velocidad de carga del sitio",
      "El número de páginas visitadas por sesión",
      "El tiempo promedio en el sitio"
    ],
    correctAnswer: "El porcentaje de visitantes que abandonan el sitio sin interactuar",
    points: 20,
    lessonId: "web-analytics"
  },
  {
    id: "analytics-3",
    type: "multiple_choice",
    question: "¿Cuál es el propósito de un embudo de conversión?",
    options: [
      "Visualizar el recorrido del usuario hasta completar una acción deseada",
      "Medir la velocidad del sitio",
      "Contar el número de visitantes",
      "Analizar el código del sitio"
    ],
    correctAnswer: "Visualizar el recorrido del usuario hasta completar una acción deseada",
    points: 20,
    lessonId: "web-analytics"
  },
  {
    id: "analytics-4",
    type: "multiple_choice",
    question: "¿Qué es un evento en analítica web?",
    options: [
      "Una interacción específica del usuario con el sitio",
      "Un error en el sitio",
      "Una actualización del contenido",
      "Un cambio en el diseño"
    ],
    correctAnswer: "Una interacción específica del usuario con el sitio",
    points: 20,
    lessonId: "web-analytics"
  },
  {
    id: "analytics-5",
    type: "multiple_choice",
    question: "¿Qué mide el tiempo promedio en la página?",
    options: [
      "Cuánto tiempo pasan los usuarios en una página específica",
      "La velocidad de carga de la página",
      "El tiempo entre actualizaciones",
      "La duración de las sesiones"
    ],
    correctAnswer: "Cuánto tiempo pasan los usuarios en una página específica",
    points: 20,
    lessonId: "web-analytics"
  }
];

// Loading component with enhanced design
function LoadingState() {
  return (
    <div className="flex flex-col gap-8 p-8 min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <div className="space-y-4">
        <div className="h-8 w-48 bg-gray-800/50 animate-pulse rounded-lg backdrop-blur-sm" />
        <div className="h-4 w-64 bg-gray-800/50 animate-pulse rounded-lg backdrop-blur-sm" />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[1, 2, 3].map((i) => (
          <div 
            key={i} 
            className="bg-gray-800/30 rounded-xl p-6 animate-pulse backdrop-blur-sm border border-gray-700/50 hover:border-purple-500/50 transition-all duration-300"
          >
            <div className="h-6 w-3/4 bg-gray-700/50 rounded-lg mb-4" />
            <div className="h-4 w-1/2 bg-gray-700/50 rounded-lg mb-2" />
            <div className="h-4 w-2/3 bg-gray-700/50 rounded-lg" />
          </div>
        ))}
      </div>
    </div>
  );
}

// Error component
function ErrorState() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 p-8">
      <div className="bg-gray-800/30 rounded-xl p-8 backdrop-blur-sm border border-gray-700/50 max-w-md w-full">
        <h2 className="text-2xl font-bold text-white mb-4">Error de Autenticación</h2>
        <p className="text-gray-300 mb-6">
          No se pudo acceder a tu sesión. Por favor, inicia sesión nuevamente.
        </p>
        <a
          href="/login"
          className="block w-full text-center bg-purple-600 hover:bg-purple-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
        >
          Volver al inicio de sesión
        </a>
      </div>
    </div>
  );
}

// Separate component for data fetching with enhanced design
async function LessonsData() {
  try {
    const session = await getServerSession(authOptions);
    
    if (!session?.user?.id) {
      return redirect("/login");
    }

    const userId = session.user.id;

    // First, ensure the user exists
    let user = await prisma.user.findUnique({
      where: { id: userId }
    });

    // If user doesn't exist, create them
    if (!user) {
      user = await prisma.user.create({
        data: {
          id: userId,
          email: session.user.email || "",
          name: session.user.name || "",
          image: session.user.image || "",
          role: "USER",
        }
      });
    }

    // Get user progress with proper error handling
    let userProgress;
    try {
      userProgress = await prisma.userProgress.upsert({
        where: { userId },
        create: {
          userId,
          experience: 0,
          level: 1,
          streak: 0,
        },
        update: {},
      });
    } catch (error) {
      return redirect("/login");
    }

    // Get all lessons
    const lessons = await prisma.lesson.findMany({
      include: {
        exercises: true
      }
    });

    // Parse lesson content
    const lessonsWithContent: Lesson[] = lessons.map(lesson => {
      const lessonData: Lesson = {
        id: lesson.id,
        title: lesson.title,
        description: lesson.description,
        difficulty: lesson.difficulty,
        category: lesson.category,
        requiredLevel: lesson.requiredLevel,
        experienceReward: lesson.experienceReward,
        exercises: lesson.exercises,
        content: null
      };

      if ((lesson as any).content) {
        try {
          const parsedContent = (lesson as any).content as Prisma.JsonValue;
          if (typeof parsedContent === 'object' && parsedContent !== null) {
            lessonData.content = {
              sections: (parsedContent as any).sections || []
            };
          }
        } catch (error) {
          // Skip invalid lesson content
        }
      }

      return lessonData;
    });

    return (
      <div className="container mx-auto px-4 py-8">
        <DashboardHeader
          heading="Lecciones"
          text="Aprende sobre marketing digital y desarrollo de negocios"
        />
        <div className="mt-8">
          <LessonsContent 
            lessons={lessonsWithContent} 
            userLevel={userProgress?.level || 1} 
          />
        </div>
      </div>
    );
  } catch (error) {
    return redirect("/login");
  }
}

export default function LessonsPage() {
  return (
    <Suspense fallback={<LoadingState />}>
      <LessonsData />
    </Suspense>
  );
}  