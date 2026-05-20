"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { motion, AnimatePresence } from "framer-motion"
import * as Dialog from "@radix-ui/react-dialog"
import { toast } from "sonner"
import { create } from "zustand"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from "chart.js"
import { Line } from "react-chartjs-2"
import { Brain, TrendingUp, Users, DollarSign, BarChart2 } from "lucide-react"
import { generateGameContent } from "@/lib/ai-service"

// Registrar componentes de Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

// Tipos de decisiones
interface Decision {
  id: string
  title: string
  description: string
  options: DecisionOption[]
  category: "social" | "content" | "paid" | "email"
}

interface DecisionOption {
  id: string
  text: string
  impact: {
    revenue: number
    engagement: number
    reach: number
    cost: number
  }
}

// Estado del juego
interface GameState {
  quarter: number
  budget: number
  revenue: number
  engagement: number
  reach: number
  history: {
    quarter: number
    revenue: number
    engagement: number
    reach: number
    budget: number
  }[]
  decisions: Decision[]
  currentDecision: Decision | null
  gameOver: boolean
  score: number
}

// Store con Zustand
const useGameStore = create<GameState>((set) => ({
  quarter: 1,
  budget: 100000,
  revenue: 0,
  engagement: 0,
  reach: 0,
  history: [],
  decisions: [],
  currentDecision: null,
  gameOver: false,
  score: 0
}))

export function MarketingSimulator() {
  const {
    quarter,
    budget,
    revenue,
    engagement,
    reach,
    history,
    decisions,
    currentDecision,
    gameOver,
    score
  } = useGameStore()

  // Configuración del gráfico
  const chartData = {
    labels: history.map(h => `Q${h.quarter}`),
    datasets: [
      {
        label: "Ingresos",
        data: history.map(h => h.revenue),
        borderColor: "rgb(34, 197, 94)",
        backgroundColor: "rgba(34, 197, 94, 0.1)",
        fill: true
      },
      {
        label: "Alcance",
        data: history.map(h => h.reach),
        borderColor: "rgb(59, 130, 246)",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        fill: true
      },
      {
        label: "Engagement",
        data: history.map(h => h.engagement),
        borderColor: "rgb(234, 179, 8)",
        backgroundColor: "rgba(234, 179, 8, 0.1)",
        fill: true
      }
    ]
  }

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const
      },
      title: {
        display: true,
        text: "Evolución de KPIs"
      }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  }

  // Manejar decisión
  const handleDecision = (option: DecisionOption) => {
    const newBudget = budget - option.impact.cost
    const newRevenue = revenue + option.impact.revenue
    const newEngagement = engagement + option.impact.engagement
    const newReach = reach + option.impact.reach

    useGameStore.setState({
      budget: newBudget,
      revenue: newRevenue,
      engagement: newEngagement,
      reach: newReach,
      history: [
        ...history,
        {
          quarter,
          revenue: newRevenue,
          engagement: newEngagement,
          reach: newReach,
          budget: newBudget
        }
      ],
      quarter: quarter + 1,
      score: score + (option.impact.revenue / 1000),
      currentDecision: null
    })

    // Mostrar feedback
    toast.success(
      `¡Buena decisión! +${option.impact.revenue}€ en ingresos, +${option.impact.engagement}% engagement`
    )

    // Verificar fin del juego
    if (quarter >= 4 || newBudget <= 0) {
      useGameStore.setState({ gameOver: true })
    }
  }

  // Generar nueva decisión con IA
  const generateNewDecision = async () => {
    try {
      const result = await generateGameContent("marketing_decision")
      const decision = result.content

      useGameStore.setState({
        currentDecision: {
          id: Math.random().toString(36).substr(2, 9),
          title: decision.title,
          description: decision.description,
          options: decision.options.map((opt: any, index: number) => ({
            id: index.toString(),
            text: opt.text,
            impact: opt.impact
          })),
          category: "social"
        }
      })
    } catch (error) {
      toast.error("Error al generar la decisión. Inténtalo de nuevo.")
    }
  }

  // Reiniciar juego
  const resetGame = () => {
    useGameStore.setState({
      quarter: 1,
      budget: 100000,
      revenue: 0,
      engagement: 0,
      reach: 0,
      history: [],
      currentDecision: null,
      gameOver: false,
      score: 0
    })
  }

  // Efecto para iniciar nuevo trimestre
  useEffect(() => {
    if (!currentDecision && !gameOver) {
      generateNewDecision()
    }
  }, [currentDecision, gameOver])

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <h3 className="text-lg font-semibold">
            Simulador de Decisiones de Marketing
          </h3>
          <div className="flex items-center gap-4">
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              className="text-sm text-muted-foreground"
            >
              Trimestre: {quarter}/4
            </motion.div>
          </div>
        </motion.div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Card className="p-4">
                <div className="flex items-center gap-2">
                  <DollarSign className="h-4 w-4 text-green-500" />
                  <span className="text-sm font-medium">Presupuesto</span>
                </div>
                <p className="text-2xl font-bold mt-2">
                  {budget.toLocaleString('es-ES')}€
                </p>
              </Card>
              <Card className="p-4">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-blue-500" />
                  <span className="text-sm font-medium">Ingresos</span>
                </div>
                <p className="text-2xl font-bold mt-2">
                  {revenue.toLocaleString('es-ES')}€
                </p>
              </Card>
              <Card className="p-4">
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4 text-yellow-500" />
                  <span className="text-sm font-medium">Engagement</span>
                </div>
                <p className="text-2xl font-bold mt-2">{engagement}%</p>
              </Card>
              <Card className="p-4">
                <div className="flex items-center gap-2">
                  <BarChart2 className="h-4 w-4 text-purple-500" />
                  <span className="text-sm font-medium">Alcance</span>
                </div>
                <p className="text-2xl font-bold mt-2">{reach}%</p>
              </Card>
            </div>

            <Card className="p-4">
              <Line data={chartData} options={chartOptions} />
            </Card>
          </div>

          <div className="space-y-4">
            {currentDecision && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-4"
              >
                <Card className="p-4">
                  <h4 className="text-lg font-semibold mb-2">
                    {currentDecision.title}
                  </h4>
                  <p className="text-sm text-muted-foreground mb-4">
                    {currentDecision.description}
                  </p>
                  <div className="space-y-2">
                    {currentDecision.options.map((option) => (
                      <Button
                        key={option.id}
                        variant="outline"
                        className="w-full justify-start"
                        onClick={() => handleDecision(option)}
                        disabled={option.impact.cost > budget}
                      >
                        <div className="flex flex-col items-start">
                          <span>{option.text}</span>
                          <span className="text-xs text-muted-foreground">
                            Costo: {option.impact.cost.toLocaleString('es-ES')}€ | Impacto: +
                            {option.impact.revenue.toLocaleString('es-ES')}€
                          </span>
                        </div>
                      </Button>
                    ))}
                  </div>
                </Card>
              </motion.div>
            )}
          </div>
        </div>

        <AnimatePresence>
          {gameOver && (
            <Dialog.Root open={gameOver}>
              <Dialog.Portal>
                <Dialog.Overlay className="fixed inset-0 bg-black/50 backdrop-blur-sm" />
                <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.9, opacity: 0 }}
                  >
                    <Card className="p-6 w-[400px]">
                      <div className="text-center space-y-4">
                        <motion.div
                          initial={{ y: -20, rotate: -10 }}
                          animate={{ y: 0, rotate: 0 }}
                          transition={{ type: "spring", stiffness: 200 }}
                        >
                          <Brain className="h-16 w-16 mx-auto text-blue-500" />
                        </motion.div>
                        <h2 className="text-2xl font-bold">¡Juego Terminado!</h2>
                        <motion.p
                          initial={{ scale: 0.8 }}
                          animate={{ scale: 1 }}
                          className="text-xl"
                        >
                          Puntuación: {Math.round(score)}
                        </motion.p>
                        <p className="text-muted-foreground">
                          {score > 100
                            ? "¡Excelente! ¡Eres un CMO excepcional!"
                            : score > 50
                            ? "¡Buen trabajo! Sigue practicando."
                            : "¡Sigue intentándolo! La práctica hace al maestro."}
                        </p>
                        <motion.div
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <Button onClick={resetGame} className="mt-4 w-full">
                            Jugar de nuevo
                          </Button>
                        </motion.div>
                      </div>
                    </Card>
                  </motion.div>
                </Dialog.Content>
              </Dialog.Portal>
            </Dialog.Root>
          )}
        </AnimatePresence>
      </div>
    </Card>
  )
}        