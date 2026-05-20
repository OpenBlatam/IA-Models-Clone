"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"
import {
  Brain,
  Gamepad2,
  Puzzle,
  BarChart2,
  Layers,
  Target,
  Zap,
  Trophy,
  Lightbulb,
  Rocket,
  ArrowRight,
  ShoppingCart,
  MessageSquare,
  Bot
} from "lucide-react"
import Link from "next/link"

const gameCategories = [
  {
    title: "Juegos de Conocimiento",
    description: "Aprende conceptos clave de marketing mientras te diviertes",
    games: [
      {
        title: "Marketing Trivia",
        description: "Pon a prueba tus conocimientos de marketing con preguntas desafiantes",
        icon: Brain,
        color: "text-blue-500",
        bgColor: "bg-blue-500/10",
        href: "/dashboard/games/trivia"
      },
      {
        title: "Juego de Memoria",
        description: "Encuentra pares de conceptos de marketing",
        icon: Puzzle,
        color: "text-purple-500",
        bgColor: "bg-purple-500/10",
        href: "/dashboard/games/memory"
      },
      {
        title: "Juego de Relaciones",
        description: "Relaciona conceptos con sus definiciones",
        icon: Layers,
        color: "text-green-500",
        bgColor: "bg-green-500/10",
        href: "/dashboard/games/match"
      }
    ]
  },
  {
    title: "Juegos de Habilidad",
    description: "Mejora tus habilidades prácticas de marketing",
    games: [
      {
        title: "Marketing Flappy",
        description: "Evita obstáculos mientras aprendes conceptos de marketing",
        icon: Gamepad2,
        color: "text-yellow-500",
        bgColor: "bg-yellow-500/10",
        href: "/dashboard/games/flappy"
      },
      {
        title: "Simulador de Marketing",
        description: "Toma decisiones estratégicas como CMO y ve su impacto",
        icon: BarChart2,
        color: "text-red-500",
        bgColor: "bg-red-500/10",
        href: "/dashboard/games/simulator"
      },
      {
        title: "Drag & Drop",
        description: "Organiza elementos de marketing en el orden correcto",
        icon: Target,
        color: "text-indigo-500",
        bgColor: "bg-indigo-500/10",
        href: "/dashboard/games/drag-drop"
      }
    ]
  },
  {
    title: "Herramientas Creativas",
    description: "Desarrolla tu creatividad y estrategia en marketing",
    games: [
      {
        title: "Constructor de Productos",
        description: "Crea y personaliza productos virtuales",
        icon: ShoppingCart,
        color: "text-pink-500",
        bgColor: "bg-pink-500/10",
        href: "/dashboard/games/product-builder"
      },
      {
        title: "Marketing Game",
        description: "Juego estratégico de marketing digital",
        icon: Lightbulb,
        color: "text-orange-500",
        bgColor: "bg-orange-500/10",
        href: "/dashboard/games/marketing-game"
      },
      {
        title: "Blatam Assistant",
        description: "Asistente IA para marketing",
        icon: Bot,
        color: "text-cyan-500",
        bgColor: "bg-cyan-500/10",
        href: "/dashboard/games/blatam-assistant"
      }
    ]
  }
]

export function GamesMenu() {
  return (
    <div className="container mx-auto p-6">
      <div className="mb-12 text-center">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
          Juegos de Marketing
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Aprende marketing de forma divertida con estos juegos interactivos diseñados para mejorar tus habilidades
        </p>
      </div>

      <div className="space-y-16">
        {gameCategories.map((category, categoryIndex) => (
          <div key={category.title} className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold mb-2">{category.title}</h2>
              <p className="text-muted-foreground">{category.description}</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {category.games.map((game, gameIndex) => (
                <Link key={game.title} href={game.href}>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: (categoryIndex * 3 + gameIndex) * 0.1 }}
                    whileHover={{ y: -5 }}
                  >
                    <Card className="group relative overflow-hidden p-6 hover:shadow-xl transition-all duration-300 border-2 hover:border-primary/50">
                      <div className="flex flex-col h-full">
                        <div className={`p-3 rounded-lg ${game.bgColor} w-fit mb-4`}>
                          <game.icon className={`h-6 w-6 ${game.color}`} />
                        </div>
                        <div className="flex-grow">
                          <h3 className="font-bold text-xl mb-2 group-hover:text-primary transition-colors">
                            {game.title}
                          </h3>
                          <p className="text-muted-foreground mb-4">
                            {game.description}
                          </p>
                        </div>
                        <Button 
                          variant="ghost" 
                          className="w-full group-hover:bg-primary/10 transition-colors"
                        >
                          Jugar ahora
                          <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                        </Button>
                      </div>
                    </Card>
                  </motion.div>
                </Link>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}    