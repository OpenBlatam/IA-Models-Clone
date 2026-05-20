"use client";

import { useState } from "react";
import { ChevronRight, BookOpen, Video, Trophy, Users, Clock, Star, Target, BarChart, Megaphone, Brain, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import Image from "next/image";

interface LearningPath {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  route: string;
  progress: number;
  totalClasses: number;
  completedClasses: number;
  category: string;
  level: "beginner" | "intermediate" | "advanced";
  duration: string;
  image?: string;
}

const learningPaths: LearningPath[] = [
  {
    id: "marketing-basico",
    title: "Marketing Digital Básico",
    description: "Aprende los fundamentos del marketing digital, redes sociales y estrategias básicas de crecimiento",
    icon: <Megaphone className="w-5 h-5" />,
    route: "/dashboard/videos?path=marketing-basico",
    progress: 0,
    totalClasses: 12,
    completedClasses: 0,
    category: "Marketing Digital",
    level: "beginner",
    duration: "12 horas",
    image: "/images/marketing-basico.jpg"
  },
  {
    id: "marketing-avanzado",
    title: "Marketing Digital Avanzado",
    description: "Domina estrategias avanzadas de marketing, análisis de datos y optimización de conversiones",
    icon: <TrendingUp className="w-5 h-5" />,
    route: "/dashboard/videos?path=marketing-avanzado",
    progress: 0,
    totalClasses: 20,
    completedClasses: 0,
    category: "Marketing Digital",
    level: "advanced",
    duration: "25 horas",
    image: "/images/marketing-avanzado.jpg"
  },
  {
    id: "ai-fundamentals",
    title: "Fundamentos de IA",
    description: "Introducción a la Inteligencia Artificial y sus aplicaciones en el mundo actual",
    icon: <Brain className="w-5 h-5" />,
    route: "/dashboard/videos?path=ai-fundamentals",
    progress: 0,
    totalClasses: 15,
    completedClasses: 0,
    category: "Inteligencia Artificial",
    level: "beginner",
    duration: "15 horas",
    image: "/images/ai-fundamentals.jpg"
  },
  {
    id: "ai-advanced",
    title: "IA Avanzada y Machine Learning",
    description: "Domina técnicas avanzadas de IA, machine learning y deep learning",
    icon: <Target className="w-5 h-5" />,
    route: "/dashboard/videos?path=ai-advanced",
    progress: 0,
    totalClasses: 25,
    completedClasses: 0,
    category: "Inteligencia Artificial",
    level: "advanced",
    duration: "35 horas",
    image: "/images/ai-advanced.jpg"
  },
  {
    id: "analytics",
    title: "Análisis de Datos y Métricas",
    description: "Aprende a analizar datos, crear dashboards y tomar decisiones basadas en datos",
    icon: <BarChart className="w-5 h-5" />,
    route: "/dashboard/videos?path=analytics",
    progress: 0,
    totalClasses: 18,
    completedClasses: 0,
    category: "Marketing Digital",
    level: "intermediate",
    duration: "20 horas",
    image: "/images/analytics.jpg"
  },
  {
    id: "ai-marketing",
    title: "IA en Marketing",
    description: "Aprende a utilizar la IA para optimizar tus estrategias de marketing y automatización",
    icon: <Brain className="w-5 h-5" />,
    route: "/dashboard/videos?path=ai-marketing",
    progress: 0,
    totalClasses: 15,
    completedClasses: 0,
    category: "Inteligencia Artificial",
    level: "intermediate",
    duration: "18 horas",
    image: "/images/ai-marketing.jpg"
  }
];

export default function LearningPathsPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedLevel, setSelectedLevel] = useState<string>("all");

  const categories = ["all", ...Array.from(new Set(learningPaths.map(path => path.category)))];
  const levels = ["all", "beginner", "intermediate", "advanced"];

  const filteredPaths = learningPaths.filter(path => {
    const categoryMatch = selectedCategory === "all" || path.category === selectedCategory;
    const levelMatch = selectedLevel === "all" || path.level === selectedLevel;
    return categoryMatch && levelMatch;
  });

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Rutas de Aprendizaje</h1>
        
        {/* Filtros */}
        <div className="flex flex-wrap gap-4 mb-8">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Categoría:</span>
            <select 
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="bg-card border border-border rounded-md px-3 py-1 text-sm"
            >
              {categories.map(category => (
                <option key={category} value={category}>
                  {category === "all" ? "Todas" : category}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Nivel:</span>
            <select 
              value={selectedLevel}
              onChange={(e) => setSelectedLevel(e.target.value)}
              className="bg-card border border-border rounded-md px-3 py-1 text-sm"
            >
              {levels.map(level => (
                <option key={level} value={level}>
                  {level === "all" ? "Todos" : level.charAt(0).toUpperCase() + level.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Grid de rutas de aprendizaje */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredPaths.map((path) => (
            <Link
              key={path.id}
              href={path.route}
              className={cn(
                "group relative overflow-hidden rounded-xl border border-border bg-card transition-all hover:border-primary/50",
                "flex flex-col"
              )}
            >
              {/* Imagen de fondo */}
              <div className="relative h-48 w-full">
                <div className="absolute inset-0 bg-gradient-to-b from-transparent to-background/80" />
                {path.image ? (
                  <Image
                    src={path.image}
                    alt={path.title}
                    fill
                    className="object-cover"
                  />
                ) : (
                  <div className="h-full w-full bg-primary/10 flex items-center justify-center">
                    {path.icon}
                  </div>
                )}
              </div>

              {/* Contenido */}
              <div className="flex-1 p-6">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-medium px-2 py-1 rounded-full bg-primary/10 text-primary">
                    {path.category}
                  </span>
                  <span className="text-xs text-muted-foreground flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {path.duration}
                  </span>
                </div>
                <h3 className="text-xl font-semibold mb-2 group-hover:text-primary transition-colors">
                  {path.title}
                </h3>
                <p className="text-sm text-muted-foreground mb-4">
                  {path.description}
                </p>

                {/* Barra de progreso */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">
                      {path.completedClasses} de {path.totalClasses} clases
                    </span>
                    <span className="font-medium text-primary">
                      {path.progress}%
                    </span>
                  </div>
                  <div className="h-2 bg-border rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary transition-all"
                      style={{ width: `${path.progress}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Botón de continuar */}
              <div className="p-4 border-t border-border">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">
                    {path.progress === 0 ? "Comenzar" : "Continuar"}
                  </span>
                  <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
} 