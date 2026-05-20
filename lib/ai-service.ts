import OpenAI from "openai"

// Configuración para usar un modelo gratuito
const openai = process.env.OPENAI_API_KEY 
  ? new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    })
  : null

// Tipos de contenido que podemos generar
export type GameContentType = 
  | "trivia_question"
  | "memory_pair"
  | "match_pair"
  | "marketing_decision"
  | "flappy_obstacle"
  | "assistant_response"

interface GameContent {
  type: GameContentType
  content: any
  error?: string
}

// Función para generar contenido del juego
export async function generateGameContent(
  type: GameContentType,
  context?: any
): Promise<GameContent> {
  try {
    if (!process.env.OPENAI_API_KEY || !openai) {
      return getFallbackContent(type)
    }

    let prompt = ""
    let systemPrompt = ""

    switch (type) {
      case "assistant_response":
        systemPrompt = "Eres un asistente experto en marketing digital de Blatam Academy. Tu objetivo es ayudar a los estudiantes a entender conceptos de marketing de manera clara y concisa. Usa ejemplos prácticos y mantén un tono profesional pero amigable."
        prompt = `Contexto de la conversación: ${JSON.stringify(context?.conversationHistory || [])}
        
        Pregunta del usuario: ${context?.userMessage}
        
        Genera una respuesta útil y educativa. La respuesta debe ser en formato JSON:
        {
          "response": "string"
        }`
        break

      case "trivia_question":
        systemPrompt = "Eres un experto en marketing digital. Genera preguntas de trivia sobre marketing."
        prompt = "Genera una pregunta de trivia sobre marketing digital con 4 opciones de respuesta y la respuesta correcta. Formato JSON: {question: string, options: string[], correctAnswer: number}"
        break

      case "memory_pair":
        systemPrompt = "Eres un experto en marketing digital. Genera pares de conceptos relacionados con marketing."
        prompt = "Genera un par de conceptos relacionados con marketing digital. Formato JSON: {term1: string, term2: string, description: string}"
        break

      case "match_pair":
        systemPrompt = "Eres un experto en marketing digital. Genera pares de conceptos y definiciones."
        prompt = "Genera un par concepto-definición sobre marketing digital. Formato JSON: {concept: string, definition: string}"
        break

      case "marketing_decision":
        systemPrompt = "Eres un experto en marketing digital. Genera escenarios de decisión estratégica."
        prompt = "Genera un escenario de decisión de marketing con 3 opciones y sus impactos. Formato JSON: {title: string, description: string, options: [{text: string, impact: {revenue: number, engagement: number, reach: number, cost: number}}]}"
        break

      case "flappy_obstacle":
        systemPrompt = "Eres un experto en marketing digital. Genera obstáculos temáticos de marketing."
        prompt = "Genera un obstáculo temático de marketing para el juego Flappy Bird. Formato JSON: {type: string, text: string, points: number}"
        break

      default:
        throw new Error(`Tipo de contenido no soportado: ${type}`)
    }

    try {
      const completion = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: prompt }
        ],
        temperature: 0.7,
      })

      const response = completion.choices[0]?.message?.content
      if (!response) {
        throw new Error("No se recibió respuesta de la API")
      }

      try {
        const parsedResponse = JSON.parse(response)
        return {
          type,
          content: parsedResponse
        }
      } catch (error) {
        return {
          type,
          content: response,
          error: "Lo sentimos, ha ocurrido un error al procesar la respuesta. Por favor, inténtalo de nuevo más tarde."
        }
      }
    } catch (error: any) {
      // Handle quota exceeded error
      if (error.code === 'insufficient_quota' || error.type === 'insufficient_quota') {
        return {
          type,
          content: getFallbackContent(type).content,
          error: "Lo sentimos, hemos alcanzado el límite de uso de la API. Por favor, inténtalo de nuevo más tarde."
        }
      }

      // Handle rate limiting
      if (error.status === 429) {
        return {
          type,
          content: getFallbackContent(type).content,
          error: "Lo sentimos, demasiadas solicitudes. Por favor, espera un momento antes de intentarlo de nuevo."
        }
      }

      // Handle other API errors
      return {
        type,
        content: getFallbackContent(type).content,
        error: "Lo sentimos, ha ocurrido un error con la API. Por favor, inténtalo de nuevo más tarde."
      }
    }
  } catch (error) {
    return {
      type,
      content: getFallbackContent(type).content,
      error: "Lo sentimos, ha ocurrido un error inesperado. Por favor, inténtalo de nuevo más tarde."
    }
  }
}

// Contenido por defecto para cada tipo de juego
function getFallbackContent(type: GameContentType): GameContent {
  const defaultResponses = [
    "El marketing digital es una estrategia que utiliza canales y dispositivos en línea para promocionar productos y servicios. Incluye SEO, redes sociales, email marketing y más.",
    "El SEO (Search Engine Optimization) es el proceso de optimizar tu sitio web para mejorar su visibilidad en los motores de búsqueda como Google.",
    "Las redes sociales son plataformas digitales que permiten a las empresas conectar con su audiencia, construir marca y generar leads.",
    "El email marketing es una estrategia efectiva para mantener el contacto con tus clientes y promover tus productos o servicios.",
    "El marketing de contenidos se centra en crear y distribuir contenido valioso para atraer y retener una audiencia definida."
  ]

  switch (type) {
    case "assistant_response":
      return {
        type,
        content: {
          response: defaultResponses[Math.floor(Math.random() * defaultResponses.length)]
        }
      }

    case "trivia_question":
      return {
        type,
        content: {
          question: "¿Qué es el SEO?",
          options: [
            "Search Engine Optimization",
            "Social Engine Optimization",
            "Search Engine Organization",
            "Social Engine Organization"
          ],
          correctAnswer: 0
        }
      }

    case "memory_pair":
      return {
        type,
        content: {
          term1: "SEO",
          term2: "Search Engine Optimization",
          description: "Optimización para motores de búsqueda"
        }
      }

    case "match_pair":
      return {
        type,
        content: {
          concept: "ROI",
          definition: "Retorno sobre la inversión en marketing"
        }
      }

    case "marketing_decision":
      return {
        type,
        content: {
          title: "Estrategia de Redes Sociales",
          description: "¿Cómo quieres abordar tu presencia en redes sociales?",
          options: [
            {
              text: "Enfoque orgánico",
              impact: {
                revenue: 5000,
                engagement: 15,
                reach: 10,
                cost: 5000
              }
            },
            {
              text: "Enfoque pagado",
              impact: {
                revenue: 15000,
                engagement: 5,
                reach: 30,
                cost: 20000
              }
            },
            {
              text: "Enfoque híbrido",
              impact: {
                revenue: 10000,
                engagement: 10,
                reach: 20,
                cost: 15000
              }
            }
          ]
        }
      }

    case "flappy_obstacle":
      return {
        type,
        content: {
          type: "social",
          text: "Instagram",
          points: 10
        }
      }

    default:
      return {
        type,
        content: {
          error: "Tipo de contenido no soportado"
        }
      }
  }
}  