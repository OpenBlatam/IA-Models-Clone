"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Trophy, Brain, Target, Users, BarChart2, Mail, MessageSquare, ShoppingBag } from "lucide-react"

interface Card {
  id: number
  icon: React.ReactNode
  title: string
  description: string
  isFlipped: boolean
  isMatched: boolean
}

const marketingCards: Card[] = [
  {
    id: 1,
    icon: <Target className="h-8 w-8" />,
    title: "Targeting",
    description: "Identificación de audiencia objetivo",
    isFlipped: false,
    isMatched: false
  },
  {
    id: 2,
    icon: <Users className="h-8 w-8" />,
    title: "Audiencia",
    description: "Segmentación de usuarios",
    isFlipped: false,
    isMatched: false
  },
  {
    id: 3,
    icon: <BarChart2 className="h-8 w-8" />,
    title: "Analytics",
    description: "Análisis de métricas",
    isFlipped: false,
    isMatched: false
  },
  {
    id: 4,
    icon: <Mail className="h-8 w-8" />,
    title: "Email Marketing",
    description: "Campañas por correo",
    isFlipped: false,
    isMatched: false
  },
  {
    id: 5,
    icon: <MessageSquare className="h-8 w-8" />,
    title: "Social Media",
    description: "Marketing en redes sociales",
    isFlipped: false,
    isMatched: false
  },
  {
    id: 6,
    icon: <ShoppingBag className="h-8 w-8" />,
    title: "E-commerce",
    description: "Ventas online",
    isFlipped: false,
    isMatched: false
  }
]

export function MarketingMemory() {
  const [cards, setCards] = useState<Card[]>([])
  const [flippedCards, setFlippedCards] = useState<number[]>([])
  const [moves, setMoves] = useState(0)
  const [gameOver, setGameOver] = useState(false)
  const [score, setScore] = useState(0)

  useEffect(() => {
    startNewGame()
  }, [])

  const startNewGame = () => {
    // Duplicar las cartas para crear pares
    const duplicatedCards = [...marketingCards, ...marketingCards]
      .map((card, index) => ({
        ...card,
        id: index,
        isFlipped: false,
        isMatched: false
      }))
      .sort(() => Math.random() - 0.5)

    setCards(duplicatedCards)
    setFlippedCards([])
    setMoves(0)
    setGameOver(false)
    setScore(0)
  }

  const handleCardClick = (index: number) => {
    if (
      flippedCards.length === 2 ||
      flippedCards.includes(index) ||
      cards[index].isMatched
    ) {
      return
    }

    const newFlippedCards = [...flippedCards, index]
    setFlippedCards(newFlippedCards)

    if (newFlippedCards.length === 2) {
      setMoves(moves + 1)
      const [firstIndex, secondIndex] = newFlippedCards
      const firstCard = cards[firstIndex]
      const secondCard = cards[secondIndex]

      if (firstCard.title === secondCard.title) {
        // Match found
        setCards(cards.map((card, i) => 
          i === firstIndex || i === secondIndex
            ? { ...card, isMatched: true }
            : card
        ))
        setFlippedCards([])
        setScore(score + 1)

        // Check if game is over
        if (score + 1 === marketingCards.length) {
          setGameOver(true)
        }
      } else {
        // No match
        setTimeout(() => {
          setFlippedCards([])
        }, 1000)
      }
    }
  }

  if (gameOver) {
    return (
      <Card className="p-6">
        <div className="text-center space-y-4">
          <Trophy className="h-16 w-16 mx-auto text-yellow-500" />
          <h2 className="text-2xl font-bold">¡Juego Completado!</h2>
          <p className="text-xl">
            Movimientos: {moves}
          </p>
          <p className="text-muted-foreground">
            {moves <= marketingCards.length * 2
              ? "¡Excelente memoria! ¡Eres un experto en marketing!"
              : moves <= marketingCards.length * 3
                ? "¡Buen trabajo! Sigue practicando."
                : "¡Sigue practicando! La memoria mejora con el tiempo."}
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
            Movimientos: {moves}
          </h3>
          <div className="text-sm text-muted-foreground">
            Pares encontrados: {score} de {marketingCards.length}
          </div>
        </div>
        
        <Progress value={(score / marketingCards.length) * 100} />
        
        <div className="grid grid-cols-3 gap-4">
          {cards.map((card, index) => (
            <motion.div
              key={index}
              className="aspect-square"
              initial={false}
              animate={{
                rotateY: flippedCards.includes(index) || card.isMatched ? 180 : 0
              }}
              transition={{ duration: 0.3 }}
            >
              <Card
                className={`h-full cursor-pointer transition-all ${
                  card.isMatched ? "bg-green-50" : "hover:bg-muted/50"
                }`}
                onClick={() => handleCardClick(index)}
              >
                <div className="relative h-full">
                  {/* Front of card */}
                  <AnimatePresence>
                    {!flippedCards.includes(index) && !card.isMatched && (
                      <motion.div
                        className="absolute inset-0 flex items-center justify-center bg-background"
                        initial={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                      >
                        <Brain className="h-8 w-8 text-muted-foreground" />
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Back of card */}
                  <AnimatePresence>
                    {(flippedCards.includes(index) || card.isMatched) && (
                      <motion.div
                        className="absolute inset-0 p-4 flex flex-col items-center justify-center text-center"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                      >
                        <div className="text-primary mb-2">
                          {card.icon}
                        </div>
                        <h4 className="font-semibold mb-1">{card.title}</h4>
                        <p className="text-sm text-muted-foreground">
                          {card.description}
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </Card>
  )
}    