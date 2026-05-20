"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { motion, AnimatePresence } from "framer-motion"
import { toast } from "sonner"
import { generateGameContent } from "@/lib/ai-service"
import { AIBadge } from "./ai-badge"
import { Bot, Send, Loader2, AlertCircle } from "lucide-react"
import Image from "next/image"

interface Message {
  role: "user" | "assistant"
  content: string
  error?: string
}

export function BlatamAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState<"checking" | "connected" | "error">("checking")

  useEffect(() => {
    checkApiStatus()
  }, [])

  const checkApiStatus = async () => {
    try {
      const result = await generateGameContent("assistant_response", {
        userMessage: "test"
      })
      setApiStatus(result.error ? "error" : "connected")
    } catch (error) {
      setApiStatus("error")
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput("")
    setMessages(prev => [...prev, { role: "user", content: userMessage }])
    setLoading(true)

    try {
      const result = await generateGameContent("assistant_response", {
        userMessage,
        conversationHistory: messages
      })

      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: result.content.response,
        error: result.error 
      }])

      if (result.error) {
        toast.error(result.error)
      }
    } catch (error) {
      toast.error("Error al obtener respuesta")
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <Image src="/b_logo.png" alt="Blatam Academy Assistant" width={40} height={40} className="rounded-full" />
            <h2 className="text-2xl font-bold">Blatam Academy Assistant</h2>
            <AIBadge />
          </div>
          <div className="flex items-center gap-2">
            {apiStatus === "checking" && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">Verificando API...</span>
              </div>
            )}
            {apiStatus === "connected" && (
              <div className="flex items-center gap-2 text-green-600">
                <Bot className="h-4 w-4" />
                <span className="text-sm">API Conectada</span>
              </div>
            )}
            {apiStatus === "error" && (
              <div className="flex items-center gap-2 text-red-600">
                <AlertCircle className="h-4 w-4" />
                <span className="text-sm">Error de API</span>
              </div>
            )}
          </div>
        </div>
        <div className="space-y-4">
          <div className="h-[400px] overflow-y-auto space-y-4 p-4 bg-muted/50 rounded-lg">
            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] p-4 rounded-lg ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      {message.role === "assistant" && (
                        <Bot className="h-4 w-4" />
                      )}
                      <span className="text-sm font-medium">
                        {message.role === "user" ? "Tú" : "Blatam Assistant"}
                      </span>
                    </div>
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    {message.error && (
                      <div className="mt-2 text-sm text-red-600">
                        {message.error}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            {loading && (
              <div className="flex justify-start">
                <div className="bg-muted p-4 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm">Pensando...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Escribe tu pregunta sobre marketing..."
              disabled={loading || apiStatus === "error"}
              className="flex-1"
            />
            <Button type="submit" disabled={loading || apiStatus === "error"}>
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </form>
        </div>
      </Card>
      <Card className="mt-4 p-6">
        <h3 className="text-xl font-bold mb-4">Explora Rutas de Aprendizaje y Cursos Especializados</h3>
        <p className="text-gray-600 mb-4">Descubre cursos especializados y rutas de aprendizaje diseñadas para mejorar tus habilidades en marketing.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-100 p-4 rounded-lg shadow-md">
            <h4 className="text-lg font-semibold text-gray-800">Agente de Marketing</h4>
            <p className="text-gray-600">Aprende a utilizar herramientas avanzadas de marketing para impulsar tu negocio.</p>
            <button className="mt-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors">Explorar</button>
          </div>
          <div className="bg-gray-100 p-4 rounded-lg shadow-md">
            <h4 className="text-lg font-semibold text-gray-800">Curso de SEO</h4>
            <p className="text-gray-600">Mejora tu visibilidad en línea con técnicas de optimización para motores de búsqueda.</p>
            <button className="mt-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors">Explorar</button>
          </div>
        </div>
      </Card>
    </>
  );
}        