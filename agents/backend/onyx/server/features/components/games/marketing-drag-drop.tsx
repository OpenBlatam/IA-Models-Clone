"use client"

import { useState } from "react"
import { create } from "zustand"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Trophy, CheckCircle2, XCircle } from "lucide-react"
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from "@dnd-kit/core"
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable"
import { SortableItem } from "./sortable-item"

interface MarketingProcess {
  id: string
  title: string
  steps: Array<{
    id: string
    content: string
  }>
}

const marketingProcesses: MarketingProcess[] = [
  {
    id: "email-marketing",
    title: "Proceso de Email Marketing",
    steps: [
      { id: "email-1", content: "Definir objetivos y audiencia objetivo" },
      { id: "email-2", content: "Crear lista de suscriptores" },
      { id: "email-3", content: "Diseñar plantilla de email" },
      { id: "email-4", content: "Escribir contenido persuasivo" },
      { id: "email-5", content: "Programar envío" },
      { id: "email-6", content: "Analizar resultados y métricas" }
    ]
  },
  {
    id: "social-media",
    title: "Proceso de Marketing en Redes Sociales",
    steps: [
      { id: "social-1", content: "Seleccionar plataformas relevantes" },
      { id: "social-2", content: "Crear calendario de contenido" },
      { id: "social-3", content: "Desarrollar estrategia de contenido" },
      { id: "social-4", content: "Crear y optimizar perfiles" },
      { id: "social-5", content: "Publicar contenido regularmente" },
      { id: "social-6", content: "Interactuar con la audiencia" }
    ]
  },
  {
    id: "content-marketing",
    title: "Proceso de Content Marketing",
    steps: [
      { id: "content-1", content: "Investigar palabras clave" },
      { id: "content-2", content: "Crear calendario editorial" },
      { id: "content-3", content: "Desarrollar contenido valioso" },
      { id: "content-4", content: "Optimizar para SEO" },
      { id: "content-5", content: "Distribuir en canales relevantes" },
      { id: "content-6", content: "Medir y analizar resultados" }
    ]
  }
]

interface GameState {
  currentProcess: number
  score: number
  gameOver: boolean
  setCurrentProcess: (process: number) => void
  setScore: (score: number) => void
  setGameOver: (gameOver: boolean) => void
  resetGame: () => void
}

const useGameStore = create<GameState>((set) => ({
  currentProcess: 0,
  score: 0,
  gameOver: false,
  setCurrentProcess: (process) => set({ currentProcess: process }),
  setScore: (score) => set({ score }),
  setGameOver: (gameOver) => set({ gameOver }),
  resetGame: () => set({ currentProcess: 0, score: 0, gameOver: false }),
}))

export function MarketingDragDrop() {
  const { 
    currentProcess, 
    score, 
    gameOver, 
    setCurrentProcess, 
    setScore, 
    setGameOver, 
    resetGame 
  } = useGameStore()

  const [items, setItems] = useState(marketingProcesses[currentProcess].steps)
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null)
  const [showExplanation, setShowExplanation] = useState(false)

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  )

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event

    if (over && active.id !== over.id) {
      setItems((items) => {
        const oldIndex = items.findIndex((item) => item.id === active.id)
        const newIndex = items.findIndex((item) => item.id === over.id)
        return arrayMove(items, oldIndex, newIndex)
      })
    }
  }

  const checkOrder = () => {
    const correctOrder = marketingProcesses[currentProcess].steps
    const isOrderCorrect = items.every((item, index) => item.id === correctOrder[index].id)
    
    setIsCorrect(isOrderCorrect)
    setShowExplanation(true)
    
    if (isOrderCorrect) {
      setScore(score + 1)
    }
  }

  const handleNext = () => {
    if (currentProcess < marketingProcesses.length - 1) {
      setCurrentProcess(currentProcess + 1)
      setItems(marketingProcesses[currentProcess + 1].steps)
      setIsCorrect(null)
      setShowExplanation(false)
    } else {
      setGameOver(true)
    }
  }

  const handleRestart = () => {
    resetGame()
    setItems(marketingProcesses[0].steps)
    setIsCorrect(null)
    setShowExplanation(false)
  }

  if (gameOver) {
    return (
      <Card className="p-6">
        <div className="text-center space-y-4">
          <Trophy className="h-16 w-16 mx-auto text-yellow-500" />
          <h2 className="text-2xl font-bold">¡Juego Completado!</h2>
          <p className="text-xl">
            Puntuación: {score} de {marketingProcesses.length}
          </p>
          <p className="text-muted-foreground">
            {score === marketingProcesses.length 
              ? "¡Perfecto! ¡Eres un experto en procesos de marketing!" 
              : score >= marketingProcesses.length / 2 
                ? "¡Buen trabajo! Sigue practicando."
                : "¡Sigue practicando! La organización es clave en marketing."}
          </p>
          <Button onClick={handleRestart} className="mt-4">
            Jugar de nuevo
          </Button>
        </div>
      </Card>
    )
  }

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">
            Proceso {currentProcess + 1} de {marketingProcesses.length}
          </h3>
          <div className="text-sm text-muted-foreground">
            Puntuación: {score}
          </div>
        </div>
        
        <Progress value={(currentProcess / marketingProcesses.length) * 100} />
        
        <div className="space-y-4">
          <h2 className="text-xl font-bold">
            {marketingProcesses[currentProcess].title}
          </h2>
          
          <p className="text-sm text-muted-foreground">
            Arrastra y suelta los pasos en el orden correcto
          </p>

          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={items}
              strategy={verticalListSortingStrategy}
            >
              <div className="space-y-2">
                {items.map((item) => (
                  <SortableItem key={item.id} id={item.id}>
                    {item.content}
                  </SortableItem>
                ))}
              </div>
            </SortableContext>
          </DndContext>

          {showExplanation && (
            <div className={`mt-4 p-4 rounded-lg ${
              isCorrect ? "bg-green-50" : "bg-red-50"
            }`}>
              <p className="text-sm">
                {isCorrect 
                  ? "¡Correcto! Has organizado los pasos en el orden adecuado."
                  : "Incorrecto. Revisa el orden de los pasos y vuelve a intentarlo."}
              </p>
            </div>
          )}

          <div className="flex gap-4">
            {!showExplanation ? (
              <Button 
                onClick={checkOrder}
                className="w-full"
              >
                Verificar orden
              </Button>
            ) : (
              <Button 
                onClick={handleNext}
                className="w-full"
              >
                {currentProcess < marketingProcesses.length - 1 
                  ? "Siguiente proceso" 
                  : "Finalizar"}
              </Button>
            )}
          </div>
        </div>
      </div>
    </Card>
  )
} 