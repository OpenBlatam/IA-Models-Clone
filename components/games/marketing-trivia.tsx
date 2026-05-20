"use client"

import { useState } from "react"
import { create } from "zustand"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { z } from "zod"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Trophy, CheckCircle2, XCircle, BarChart2 } from "lucide-react"

// Zustand store para el estado del juego
interface QuizState {
  score: number
  currentQuestion: number
  gameOver: boolean
  answers: Array<{
    question: string
    userAnswer: string
    correctAnswer: string
    isCorrect: boolean
  }>
  setScore: (score: number) => void
  setCurrentQuestion: (question: number) => void
  setGameOver: (gameOver: boolean) => void
  addAnswer: (answer: QuizState['answers'][0]) => void
  resetGame: () => void
}

const useQuizStore = create<QuizState>((set) => ({
  score: 0,
  currentQuestion: 0,
  gameOver: false,
  answers: [],
  setScore: (score) => set({ score }),
  setCurrentQuestion: (currentQuestion) => set({ currentQuestion }),
  setGameOver: (gameOver) => set({ gameOver }),
  addAnswer: (answer) => set((state) => ({ 
    answers: [...state.answers, answer] 
  })),
  resetGame: () => set({ 
    score: 0, 
    currentQuestion: 0, 
    gameOver: false,
    answers: []
  }),
}))

// Esquema de validación con Zod
const answerSchema = z.object({
  answer: z.number().min(0).max(3),
})

type AnswerForm = z.infer<typeof answerSchema>

interface Question {
  question: string
  options: string[]
  correctAnswer: number
  explanation: string
}

const questions: Question[] = [
  {
    question: "¿Cuál es el objetivo principal del marketing digital?",
    options: [
      "Vender productos a cualquier costo",
      "Conectar con la audiencia y generar valor",
      "Acumular seguidores en redes sociales",
      "Crear anuncios llamativos"
    ],
    correctAnswer: 1,
    explanation: "El marketing digital busca crear conexiones significativas con la audiencia y ofrecer valor, no solo vender."
  },
  {
    question: "¿Qué es el SEO?",
    options: [
      "Una estrategia de redes sociales",
      "Optimización para motores de búsqueda",
      "Un tipo de anuncio en Google",
      "Una herramienta de email marketing"
    ],
    correctAnswer: 1,
    explanation: "SEO (Search Engine Optimization) es el proceso de optimizar el contenido para mejorar su visibilidad en los motores de búsqueda."
  },
  {
    question: "¿Cuál es la diferencia entre tráfico orgánico y de pago?",
    options: [
      "No hay diferencia, son lo mismo",
      "El orgánico es gratuito y natural, el de pago requiere inversión",
      "El orgánico es más rápido que el de pago",
      "El de pago siempre es mejor que el orgánico"
    ],
    correctAnswer: 1,
    explanation: "El tráfico orgánico viene de búsquedas naturales y contenido, mientras que el de pago requiere inversión en publicidad."
  },
  {
    question: "¿Qué es el ROI en marketing?",
    options: [
      "Return on Investment - Retorno sobre la inversión",
      "Rate of Interest - Tasa de interés",
      "Return on Income - Retorno sobre ingresos",
      "Rate of Investment - Tasa de inversión"
    ],
    correctAnswer: 0,
    explanation: "ROI (Return on Investment) mide la rentabilidad de una inversión en marketing comparando el beneficio con el costo."
  },
  {
    question: "¿Qué es el engagement en redes sociales?",
    options: [
      "Solo el número de seguidores",
      "La interacción y participación de la audiencia",
      "La cantidad de publicaciones",
      "El número de visitas al perfil"
    ],
    correctAnswer: 1,
    explanation: "El engagement mide la interacción real de la audiencia con el contenido, no solo números superficiales."
  },
  {
    question: "¿Qué es el CTA en marketing digital?",
    options: [
      "Call to Action - Llamada a la acción",
      "Click Through Analysis - Análisis de clics",
      "Content Type Assessment - Evaluación de tipo de contenido",
      "Customer Traffic Analysis - Análisis de tráfico de clientes"
    ],
    correctAnswer: 0,
    explanation: "CTA (Call to Action) es un elemento que invita al usuario a realizar una acción específica, como suscribirse o comprar."
  },
  {
    question: "¿Qué es el remarketing?",
    options: [
      "Crear una nueva campaña desde cero",
      "Dirigirse a usuarios que ya han interactuado con tu marca",
      "Eliminar campañas antiguas",
      "Cambiar el nombre de una campaña"
    ],
    correctAnswer: 1,
    explanation: "El remarketing permite llegar a usuarios que ya han mostrado interés en tu marca, aumentando las probabilidades de conversión."
  },
  {
    question: "¿Qué es el CTR en marketing digital?",
    options: [
      "Click Through Rate - Tasa de clics",
      "Content Type Ratio - Ratio de tipo de contenido",
      "Customer Traffic Ratio - Ratio de tráfico de clientes",
      "Conversion Time Rate - Tasa de tiempo de conversión"
    ],
    correctAnswer: 0,
    explanation: "CTR (Click Through Rate) mide el porcentaje de personas que hacen clic en un enlace o anuncio en relación con el número total de impresiones."
  },
  {
    question: "¿Qué es el buyer persona?",
    options: [
      "Un tipo de anuncio",
      "Una representación semi-ficticia del cliente ideal",
      "Una herramienta de análisis",
      "Un método de pago"
    ],
    correctAnswer: 1,
    explanation: "El buyer persona es una representación detallada del cliente ideal, basada en datos reales y comportamientos observados."
  },
  {
    question: "¿Qué es el funnel de conversión?",
    options: [
      "Un tipo de anuncio en redes sociales",
      "El proceso que sigue un usuario desde que descubre tu marca hasta que realiza una acción",
      "Una herramienta de email marketing",
      "Un método de análisis de competencia"
    ],
    correctAnswer: 1,
    explanation: "El funnel de conversión representa el camino que sigue un usuario desde el primer contacto hasta la conversión, ayudando a optimizar cada etapa."
  },
  {
    question: "¿Qué es el KPI en marketing digital?",
    options: [
      "Key Performance Indicator - Indicador clave de rendimiento",
      "Key Product Information - Información clave del producto",
      "Key Price Indicator - Indicador clave de precio",
      "Key Process Information - Información clave del proceso"
    ],
    correctAnswer: 0,
    explanation: "KPI (Key Performance Indicator) son métricas que ayudan a medir el éxito de las estrategias de marketing y el rendimiento de las campañas."
  },
  {
    question: "¿Qué es el inbound marketing?",
    options: [
      "Una estrategia de publicidad pagada",
      "Una metodología que atrae clientes a través de contenido valioso",
      "Un tipo de email marketing",
      "Una técnica de ventas directas"
    ],
    correctAnswer: 1,
    explanation: "El inbound marketing atrae clientes potenciales a través de contenido relevante y útil, en lugar de interrumpirlos con publicidad tradicional."
  },
  {
    question: "¿Qué es el A/B testing?",
    options: [
      "Un método de análisis de competencia",
      "Una técnica para comparar dos versiones de algo para ver cuál funciona mejor",
      "Una herramienta de email marketing",
      "Un tipo de anuncio en redes sociales"
    ],
    correctAnswer: 1,
    explanation: "El A/B testing permite comparar dos versiones de una página, email o anuncio para determinar cuál genera mejores resultados."
  },
  {
    question: "¿Qué es el content marketing?",
    options: [
      "Solo publicar en redes sociales",
      "Crear y distribuir contenido valioso para atraer y retener una audiencia",
      "Hacer publicidad pagada",
      "Enviar emails masivos"
    ],
    correctAnswer: 1,
    explanation: "El content marketing se centra en crear contenido relevante y valioso para atraer y retener una audiencia definida, con el objetivo de generar beneficios."
  },
  {
    question: "¿Qué es el lead nurturing?",
    options: [
      "Ignorar a los leads hasta que estén listos para comprar",
      "Cultivar relaciones con leads a través de contenido relevante y comunicación personalizada",
      "Enviar emails masivos a todos los leads",
      "Vender directamente a los leads sin preparación"
    ],
    correctAnswer: 1,
    explanation: "El lead nurturing es el proceso de cultivar relaciones con leads a través de contenido relevante y comunicación personalizada en cada etapa del funnel de ventas."
  },
  {
    question: "¿Qué es el marketing de afiliados?",
    options: [
      "Un tipo de publicidad tradicional",
      "Un programa donde terceros promocionan tus productos a cambio de una comisión",
      "Una estrategia de email marketing",
      "Un método de análisis de competencia"
    ],
    correctAnswer: 1,
    explanation: "El marketing de afiliados permite que terceros promocionen tus productos o servicios a cambio de una comisión por cada venta o acción realizada."
  },
  {
    question: "¿Qué es el marketing automation?",
    options: [
      "Enviar emails manualmente a cada cliente",
      "Automatizar tareas de marketing repetitivas para mejorar la eficiencia",
      "Crear anuncios automáticamente",
      "Analizar datos de forma manual"
    ],
    correctAnswer: 1,
    explanation: "El marketing automation permite automatizar tareas repetitivas como emails, publicaciones en redes sociales y seguimiento de leads, mejorando la eficiencia y personalización."
  }
]

export function MarketingTrivia() {
  const { 
    score, 
    currentQuestion, 
    gameOver, 
    answers,
    setScore, 
    setCurrentQuestion, 
    setGameOver, 
    addAnswer,
    resetGame 
  } = useQuizStore()
  const [showExplanation, setShowExplanation] = useState(false)
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null)
  const [showResults, setShowResults] = useState(false)

  const { handleSubmit, reset } = useForm<AnswerForm>({
    resolver: zodResolver(answerSchema),
  })

  const handleAnswerClick = (index: number) => {
    if (showExplanation) return

    const currentQ = questions[currentQuestion]
    const isCorrect = index === currentQ.correctAnswer
    
    setSelectedAnswer(index)
    setShowExplanation(true)
    
    if (isCorrect) {
      setScore(score + 1)
    }

    addAnswer({
      question: currentQ.question,
      userAnswer: currentQ.options[index],
      correctAnswer: currentQ.options[currentQ.correctAnswer],
      isCorrect
    })
  }

  const handleNext = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1)
      setShowExplanation(false)
      setSelectedAnswer(null)
    } else {
      setGameOver(true)
    }
  }

  const handleRestart = () => {
    resetGame()
    setShowExplanation(false)
    setSelectedAnswer(null)
    setShowResults(false)
    reset()
  }

  const handleShowResults = () => {
    setShowResults(true)
  }

  if (gameOver && showResults) {
    return (
      <Card className="p-6">
        <div className="space-y-6">
          <div className="text-center space-y-4">
            <Trophy className="h-16 w-16 mx-auto text-yellow-500" />
            <h2 className="text-2xl font-bold">Resultados Detallados</h2>
            <p className="text-xl">
              Puntuación: {score} de {questions.length}
            </p>
          </div>

          <div className="space-y-4">
            {answers.map((answer, index) => (
              <div 
                key={index}
                className={`p-4 rounded-lg ${
                  answer.isCorrect ? "bg-green-50" : "bg-red-50"
                }`}
              >
                <h3 className="font-semibold mb-2">Pregunta {index + 1}</h3>
                <p className="text-sm mb-2">{answer.question}</p>
                <div className="space-y-1 text-sm">
                  <p className={`${answer.isCorrect ? "text-green-600" : "text-red-600"}`}>
                    Tu respuesta: {answer.userAnswer}
                  </p>
                  {!answer.isCorrect && (
                    <p className="text-green-600">
                      Respuesta correcta: {answer.correctAnswer}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="flex justify-center gap-4">
            <Button onClick={handleRestart} variant="outline">
              Jugar de nuevo
            </Button>
          </div>
        </div>
      </Card>
    )
  }

  if (gameOver) {
    return (
      <Card className="p-6">
        <div className="text-center space-y-4">
          <Trophy className="h-16 w-16 mx-auto text-yellow-500" />
          <h2 className="text-2xl font-bold">¡Juego Completado!</h2>
          <p className="text-xl">
            Puntuación: {score} de {questions.length}
          </p>
          <p className="text-muted-foreground">
            {score === questions.length 
              ? "¡Perfecto! ¡Eres un experto en marketing!" 
              : score >= questions.length / 2 
                ? "¡Buen trabajo! Sigue aprendiendo."
                : "¡Sigue practicando! El marketing es un viaje de aprendizaje."}
          </p>
          <div className="flex justify-center gap-4">
            <Button onClick={handleShowResults} variant="outline">
              Ver respuestas
            </Button>
            <Button onClick={handleRestart}>
              Jugar de nuevo
            </Button>
          </div>
        </div>
      </Card>
    )
  }

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">
            Pregunta {currentQuestion + 1} de {questions.length}
          </h3>
          <div className="text-sm text-muted-foreground">
            Puntuación: {score}
          </div>
        </div>
        
        <Progress value={(currentQuestion / questions.length) * 100} />
        
        <div className="space-y-4">
          <h2 className="text-xl font-bold">
            {questions[currentQuestion].question}
          </h2>
          
          <div className="grid gap-3">
            {questions[currentQuestion].options.map((option, index) => (
              <Button
                key={index}
                type="button"
                variant={selectedAnswer === index ? "default" : "outline"}
                className={`w-full justify-start transition-colors duration-200 ${
                  showExplanation
                    ? index === questions[currentQuestion].correctAnswer
                      ? "bg-green-500 hover:bg-green-600 text-white"
                      : selectedAnswer === index
                      ? "bg-red-500 hover:bg-red-600 text-white"
                      : ""
                    : ""
                }`}
                onClick={() => handleAnswerClick(index)}
                disabled={showExplanation}
              >
                <div className="flex items-center gap-2">
                  {showExplanation && index === questions[currentQuestion].correctAnswer && (
                    <CheckCircle2 className="h-4 w-4" />
                  )}
                  {showExplanation && selectedAnswer === index && selectedAnswer !== questions[currentQuestion].correctAnswer && (
                    <XCircle className="h-4 w-4" />
                  )}
                  {option}
                </div>
              </Button>
            ))}
          </div>

          {showExplanation && (
            <div className="mt-4 p-4 rounded-lg bg-muted">
              <p className="text-sm">
                {questions[currentQuestion].explanation}
              </p>
            </div>
          )}

          {showExplanation && (
            <Button 
              type="button"
              onClick={handleNext}
              className="w-full mt-4"
            >
              {currentQuestion < questions.length - 1 ? "Siguiente pregunta" : "Finalizar"}
            </Button>
          )}
        </div>
      </div>
    </Card>
  )
} 