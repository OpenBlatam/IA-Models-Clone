"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Trophy, Brain, Target, Users, BarChart2, Mail, MessageSquare, ShoppingBag, Sparkles } from "lucide-react"

interface Concept {
  id: string
  title: string
  description: string
  icon: React.ReactNode
  matched: boolean
  selected: boolean
}

interface Definition {
  id: string
  text: string
  conceptId: string
  matched: boolean
  selected: boolean
}

const marketingConcepts: Concept[] = [
  {
    id: "1",
    title: "ChatGPT Marketing",
    description: "Uso de IA para generar contenido",
    icon: <Sparkles className="h-8 w-8 text-purple-500" />,
    matched: false,
    selected: false
  },
  {
    id: "2",
    title: "Predictive Analytics",
    description: "Análisis predictivo de datos",
    icon: <BarChart2 className="h-8 w-8 text-blue-500" />,
    matched: false,
    selected: false
  },
  {
    id: "3",
    title: "Personalización IA",
    description: "Experiencias personalizadas",
    icon: <Users className="h-8 w-8 text-green-500" />,
    matched: false,
    selected: false
  },
  {
    id: "4",
    title: "Automatización IA",
    description: "Automatización de tareas",
    icon: <Brain className="h-8 w-8 text-orange-500" />,
    matched: false,
    selected: false
  }
]

const definitions: Definition[] = [
  {
    id: "d1",
    text: "Generación de contenido y respuestas automáticas usando modelos de lenguaje avanzados",
    conceptId: "1",
    matched: false,
    selected: false
  },
  {
    id: "d2",
    text: "Uso de algoritmos para predecir comportamientos y tendencias futuras",
    conceptId: "2",
    matched: false,
    selected: false
  },
  {
    id: "d3",
    text: "Adaptación de contenido y ofertas basada en el comportamiento del usuario",
    conceptId: "3",
    matched: false,
    selected: false
  },
  {
    id: "d4",
    text: "Optimización de procesos y tareas repetitivas mediante inteligencia artificial",
    conceptId: "4",
    matched: false,
    selected: false
  }
]

export function MarketingMatch() {
  const [concepts, setConcepts] = useState<Concept[]>([])
  const [defs, setDefs] = useState<Definition[]>([])
  const [score, setScore] = useState(0)
  const [gameOver, setGameOver] = useState(false)

  useEffect(() => {
    startNewGame()
  }, [])

  const startNewGame = () => {
    setConcepts(marketingConcepts.map(c => ({ ...c, matched: false, selected: false })))
    setDefs(definitions.map(d => ({ ...d, matched: false, selected: false })))
    setScore(0)
    setGameOver(false)
  }

  const handleConceptClick = (conceptId: string) => {
    if (concepts.find(c => c.id === conceptId)?.matched) return

    setConcepts(concepts.map(c => ({
      ...c,
      selected: c.id === conceptId ? !c.selected : false
    })))
    setDefs(defs.map(d => ({ ...d, selected: false })))
  }

  const handleDefinitionClick = (definitionId: string) => {
    if (defs.find(d => d.id === definitionId)?.matched) return

    const selectedConcept = concepts.find(c => c.selected)
    if (!selectedConcept) return

    const definition = defs.find(d => d.id === definitionId)
    if (!definition) return

    if (definition.conceptId === selectedConcept.id) {
      // Match correcto
      setConcepts(concepts.map(c => 
        c.id === selectedConcept.id ? { ...c, matched: true, selected: false } : c
      ))
      setDefs(defs.map(d =>
        d.id === definitionId ? { ...d, matched: true, selected: false } : d
      ))
      setScore(score + 1)

      // Verificar si el juego terminó
      if (score + 1 === concepts.length) {
        setGameOver(true)
      }
    } else {
      // Match incorrecto
      setConcepts(concepts.map(c => ({ ...c, selected: false })))
      setDefs(defs.map(d => ({ ...d, selected: false })))
    }
  }

  if (gameOver) {
    return (
      <Card className="p-6">
        <div className="text-center space-y-4">
          <Trophy className="h-16 w-16 mx-auto text-yellow-500" />
          <h2 className="text-2xl font-bold">¡Juego Completado!</h2>
          <p className="text-xl">
            Puntuación: {score} de {concepts.length}
          </p>
          <p className="text-muted-foreground">
            {score === concepts.length
              ? "¡Excelente! ¡Eres un experto en IA y Marketing!"
              : "¡Buen trabajo! Sigue aprendiendo sobre IA en Marketing."}
          </p>
          <Button onClick={startNewGame} className="mt-4">
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
            Relaciona los conceptos
          </h3>
          <div className="text-sm text-muted-foreground">
            Relaciones correctas: {score} de {concepts.length}
          </div>
        </div>
        
        <Progress value={(score / concepts.length) * 100} />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h4 className="font-semibold text-muted-foreground">Conceptos</h4>
            {concepts.map(concept => (
              <Card
                key={concept.id}
                className={`p-4 cursor-pointer transition-all ${
                  concept.matched ? "bg-green-50" : 
                  concept.selected ? "bg-primary/10" : 
                  "hover:bg-muted/50"
                }`}
                onClick={() => handleConceptClick(concept.id)}
              >
                <div className="flex items-center gap-3">
                  {concept.icon}
                  <div>
                    <h4 className="font-semibold">{concept.title}</h4>
                    <p className="text-sm text-muted-foreground">{concept.description}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
          
          <div className="space-y-4">
            <h4 className="font-semibold text-muted-foreground">Definiciones</h4>
            {defs.map(definition => (
              <Card
                key={definition.id}
                className={`p-4 cursor-pointer transition-all ${
                  definition.matched ? "bg-green-50" : 
                  definition.selected ? "bg-primary/10" : 
                  "hover:bg-muted/50"
                }`}
                onClick={() => handleDefinitionClick(definition.id)}
              >
                <p className="text-sm">{definition.text}</p>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </Card>
  )
} 