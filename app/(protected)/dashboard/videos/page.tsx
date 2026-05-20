"use client";

import React, { useState, useRef, useEffect, Suspense } from "react";
import { cn } from "@/lib/utils";
import { Flag, Languages, PlayCircle, ChevronDown, ChevronUp, Download, Link2, X, Video as VideoIcon, UploadCloud, Camera, ArrowLeft } from "lucide-react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import VideoHeader from "@/components/videos/VideoHeader";
import VideoResumen from "@/components/videos/VideoResumen";
import { VideoResources } from "@/components/videos/VideoResources";
import VideoComments from "@/components/videos/VideoComments";
import { VideoQuestionBar } from "@/components/videos/VideoQuestionBar";
import { useSession } from "next-auth/react";
import VideoSidebar from "@/components/videos/VideoSidebar";
import CongratsModal from "@/components/videos/CongratsModal";
import dynamic from "next/dynamic";
import NextClassModal from "@/components/videos/NextClassModal";
import { ThemeProvider } from "next-themes";
import VideoLearningPaths from "@/components/videos/VideoLearningPaths";
import { VideoQuestions } from "@/components/videos/VideoQuestions";
import { useSearchParams, useRouter } from "next/navigation";
import ContinueLearningCarousel from "@/components/videos/ContinueLearningCarousel";
import { motion } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Clock, Users, Award, Play } from "lucide-react";
import { academies } from "@/lib/academies";
import VideoPlayer from "@/components/videos/VideoPlayer";
import { Academy } from "@/lib/types/academy";
import { AcademyCard } from "@/components/academy/academy-card";

interface Comment {
  id: string;
  user: {
    name: string;
    image?: string | null;
  };
  content: string;
  createdAt: string;
}

interface Resource {
  id: string;
  title: string;
  type: "file" | "reading";
  url: string;
  description?: string;
}

interface Video {
  id: string;
  title: string;
  description: string;
  videoUrl: string;
  thumbnail: string;
  duration: string;
  isLocked: boolean;
  isCompleted: boolean;
  resumen: {
    titulo: string;
    descripcion: string;
    puntos: string[];
  };
  comments: Comment[];
  files: Resource[];
  readings: Resource[];
}

const videos: Video[] = [
  {
    id: "1",
    title: "Introducción a la Inteligencia Artificial",
    description: "Clase 1 de 15 • Curso Gratis de Introducción a la Inteligencia Artificial",
    videoUrl: "https://www.w3schools.com/html/movie.mp4",
    thumbnail: "/images/thumbnails/ai-intro.jpg",
    duration: "15:00",
    isLocked: false,
    isCompleted: false,
    resumen: {
      titulo: "¿Qué es la Inteligencia Artificial?",
      descripcion: "La IA es la simulación de procesos de inteligencia humana por parte de máquinas, especialmente sistemas informáticos.",
      puntos: ["¿Por qué es importante?", "La IA está transformando industrias y creando nuevas oportunidades de innovación."],
    },
    comments: [
      {
        id: "1",
        user: { name: "Juan Pérez", image: null },
        content: "Excelente introducción al tema.",
        createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
      },
    ],
    files: [
      {
        id: "1",
        title: "Introducción a la IA",
        type: "file",
        url: "/files/intro-ia.pdf",
        description: "PDF con los conceptos básicos"
      }
    ],
    readings: [
      {
        id: "1",
        title: "¿Qué es la IA? - IBM",
        type: "reading",
        url: "https://www.ibm.com/topics/artificial-intelligence",
        description: "Artículo introductorio"
      }
    ],
  },
  {
    id: "2",
    title: "Historia y Evolución de la IA",
    description: "Clase 2 de 15 • Curso Gratis de Introducción a la Inteligencia Artificial",
    videoUrl: "https://www.w3schools.com/html/movie.mp4",
    thumbnail: "/images/thumbnails/ai-history.jpg",
    duration: "15:00",
    isLocked: false,
    isCompleted: false,
    resumen: {
      titulo: "¿Cómo ha evolucionado la IA?",
      descripcion: "Desde los primeros algoritmos hasta los sistemas modernos de aprendizaje profundo, la IA ha recorrido un largo camino.",
      puntos: ["¿Cuáles son los hitos más importantes?"],
    },
    comments: [
      {
        id: "2",
        user: { name: "María García", image: null },
        content: "Muy interesante ver cómo ha evolucionado todo.",
        createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
      },
    ],
    files: [
      {
        id: "2",
        title: "Línea de tiempo de la IA",
        type: "file",
        url: "/files/ia-timeline.pdf",
        description: "PDF con la evolución histórica"
      }
    ],
    readings: [
      {
        id: "2",
        title: "Historia de la IA - Stanford",
        type: "reading",
        url: "https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/General%20Essays/AIMag26-04-016.pdf",
        description: "Artículo académico"
      }
    ],
  },
  {
    id: "3",
    title: "Aplicaciones Prácticas de la IA",
    description: "Clase 3 de 15 • Curso Gratis de Introducción a la Inteligencia Artificial",
    videoUrl: "https://www.w3schools.com/html/movie.mp4",
    thumbnail: "/images/thumbnails/ai-applications.jpg",
    duration: "15:00",
    isLocked: false,
    isCompleted: false,
    resumen: {
      titulo: "¿Dónde encontramos la IA en nuestra vida diaria?",
      descripcion: "La IA está presente en asistentes virtuales, recomendaciones de contenido, diagnóstico médico y más.",
      puntos: ["¿Qué sectores se benefician más?"],
    },
    comments: [
      {
        id: "3",
        user: { name: "Carlos López", image: null },
        content: "Me sorprende ver cuánto usamos la IA sin darnos cuenta.",
        createdAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
      },
    ],
    files: [
      {
        id: "3",
        title: "Casos de uso de la IA",
        type: "file",
        url: "/files/ia-casos.pdf",
        description: "PDF con ejemplos prácticos"
      }
    ],
    readings: [
      {
        id: "3",
        title: "IA en la vida cotidiana - MIT",
        type: "reading",
        url: "https://www.technologyreview.com/artificial-intelligence",
        description: "Artículo de divulgación"
      }
    ],
  },
  {
    id: "4",
    title: "Ética y Responsabilidad en la IA",
    description: "Clase 4 de 15 • Curso Gratis de Introducción a la Inteligencia Artificial",
    videoUrl: "https://www.w3schools.com/html/movie.mp4",
    thumbnail: "/images/thumbnails/ai-ethics.jpg",
    duration: "15:00",
    isLocked: false,
    isCompleted: false,
    resumen: {
      titulo: "¿Por qué es importante la ética en la IA?",
      descripcion: "La IA debe desarrollarse y utilizarse de manera responsable, considerando sus impactos sociales y éticos.",
      puntos: ["¿Qué principios debemos seguir?"],
    },
    comments: [
      {
        id: "4",
        user: { name: "Laura Martínez", image: null },
        content: "Excelente enfoque en los aspectos éticos.",
        createdAt: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
      },
    ],
    files: [
      {
        id: "4",
        title: "Guía de ética en IA",
        type: "file",
        url: "/files/ia-etica.pdf",
        description: "PDF con principios éticos"
      }
    ],
    readings: [
      {
        id: "4",
        title: "Ética en IA - UNESCO",
        type: "reading",
        url: "https://www.unesco.org/es/artificial-intelligence/ethics",
        description: "Recomendaciones éticas"
      }
    ],
  },
  {
    id: "5",
    title: "Machine Learning: Conceptos Básicos",
    description: "Clase 5 de 15 • Curso Gratis de Introducción a la Inteligencia Artificial",
    videoUrl: "https://www.w3schools.com/html/movie.mp4",
    thumbnail: "/images/thumbnails/ai-ml.jpg",
    duration: "15:00",
    isLocked: false,
    isCompleted: false,
    resumen: {
      titulo: "¿Qué es el Machine Learning?",
      descripcion: "El Machine Learning es una rama de la IA que permite a las máquinas aprender de los datos sin ser programadas explícitamente.",
      puntos: ["¿Cómo funciona?"],
    },
    comments: [
      {
        id: "5",
        user: { name: "Pedro Sánchez", image: null },
        content: "Muy bien explicado para principiantes.",
        createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
      },
    ],
    files: [
      {
        id: "5",
        title: "Fundamentos de Machine Learning",
        type: "file",
        url: "/files/ml-fundamentos.pdf",
        description: "PDF con conceptos básicos"
      }
    ],
    readings: [
      {
        id: "5",
        title: "Machine Learning - Google",
        type: "reading",
        url: "https://developers.google.com/machine-learning/crash-course",
        description: "Curso introductorio"
      }
    ],
  },
  {
    id: "6",
    title: "IA y Medio Ambiente: Soluciones Sostenibles",
    description: "Clase 6 de 15 • Curso Gratis de Introducción a la Inteligencia Artificial",
    videoUrl: "https://www.w3schools.com/html/movie.mp4",
    thumbnail: "/images/thumbnails/ai-environment.jpg",
    duration: "15:00",
    isLocked: false,
    isCompleted: false,
    resumen: {
      titulo: "¿Cómo ayuda la IA a proteger el medio ambiente?",
      descripcion: "La IA se utiliza para monitorear ecosistemas, optimizar el uso de recursos y predecir desastres naturales.",
      puntos: ["¿Qué proyectos destacan?"],
    },
    comments: [
      {
        id: "6",
        user: { name: "Jorge Herrera", image: null },
        content: "Excelente ver cómo la tecnología puede ayudar al medio ambiente.",
        createdAt: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(),
      },
    ],
    files: [
      {
        id: "6",
        title: "IA para la sostenibilidad",
        type: "file",
        url: "/files/ia-sostenibilidad.pdf",
        description: "PDF con casos de estudio"
      }
    ],
    readings: [
      {
        id: "6",
        title: "IA y medio ambiente - ONU",
        type: "reading",
        url: "https://www.unep.org/es/noticias-y-reportajes/reportaje/la-inteligencia-artificial-al-servicio-del-medio-ambiente",
        description: "Artículo de investigación"
      }
    ],
  },
];

// Import VideoPlayer with no SSR
const VideoPlayerComponent = React.lazy(() => import("@/components/videos/VideoPlayer"));

const VideoPlayerFallback = () => (
  <div className="w-full max-w-5xl aspect-video rounded-2xl overflow-hidden bg-zinc-900 shadow-xl border border-zinc-800 animate-fade-in">
    <div className="w-full h-full flex items-center justify-center">
      <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin" />
    </div>
  </div>
);

interface VideosPageProps {
  searchParams: { [key: string]: string | string[] | undefined };
}

interface CourseProgress {
  id: string;
  title: string;
  instructor?: string;
  image: string;
  currentClass: number;
  totalClasses: number;
  progress: number;
  route: string;
  icon: React.ReactNode;
  videoUrl?: string;
}

export default function VideosPage() {
  const router = useRouter();
  const [selectedAcademy, setSelectedAcademy] = useState<Academy | null>(null);
  const [selectedClassId, setSelectedClassId] = useState<string | null>(null);

  const handleAcademySelect = (academy: Academy) => {
    setSelectedAcademy(academy);
    // Seleccionar la primera clase por defecto
    if (academy.classes && academy.classes.length > 0) {
      setSelectedClassId(academy.classes[0].id);
    }
  };

  const handleClassSelect = (classId: string) => {
    setSelectedClassId(classId);
  };

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <div className="container mx-auto py-8 space-y-8">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Academias de IA</h1>
          <p className="text-muted-foreground">
            Aprende a dominar las herramientas de IA más populares con nuestros cursos especializados.
          </p>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="all" className="space-y-6">
          <TabsList>
            <TabsTrigger value="all">Todas</TabsTrigger>
            <TabsTrigger value="beginner">Principiante</TabsTrigger>
            <TabsTrigger value="intermediate">Intermedio</TabsTrigger>
            <TabsTrigger value="advanced">Avanzado</TabsTrigger>
          </TabsList>

          <TabsContent value="all" className="space-y-6">
            {/* Featured Academy */}
            {selectedAcademy && selectedClassId ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <Suspense fallback={<VideoPlayerFallback />}>
                  <VideoPlayerComponent
                    courseId={selectedAcademy.id}
                    classId={selectedClassId}
                    onSelectClass={handleClassSelect}
                    autoPlay={true}
                  />
                </Suspense>
              </motion.div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {academies.map((academy) => (
                  <AcademyCard
                    key={academy.id}
                    academy={academy}
                    onSelect={handleAcademySelect}
                  />
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="beginner" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {academies
                .filter((academy) => academy.level === "beginner")
                .map((academy) => (
                  <AcademyCard
                    key={academy.id}
                    academy={academy}
                    onSelect={handleAcademySelect}
                  />
                ))}
            </div>
          </TabsContent>

          <TabsContent value="intermediate" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {academies
                .filter((academy) => academy.level === "intermediate")
                .map((academy) => (
                  <AcademyCard
                    key={academy.id}
                    academy={academy}
                    onSelect={handleAcademySelect}
                  />
                ))}
            </div>
          </TabsContent>

          <TabsContent value="advanced" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {academies
                .filter((academy) => academy.level === "advanced")
                .map((academy) => (
                  <AcademyCard
                    key={academy.id}
                    academy={academy}
                    onSelect={handleAcademySelect}
                  />
                ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </ThemeProvider>
  );
}        