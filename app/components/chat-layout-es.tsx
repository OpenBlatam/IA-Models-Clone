"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { MessageSquare, X, GraduationCap, BookOpen, Users, Calendar, Trophy } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const WELCOME_MESSAGE: Message = {
  role: "assistant",
  content: "¡Bienvenido a Blatam Academy! Soy tu asistente IA. Puedo ayudarte con:\n\n📚 Información de cursos y matriculación\n👥 Soporte estudiantil y recursos\n📅 Próximos eventos y talleres\n🏆 Seguimiento de logros\n\n¿En qué puedo ayudarte hoy?"
};

const QUICK_ACTIONS = [
  { icon: <BookOpen className="h-4 w-4" />, label: "Cursos", prompt: "¿Qué cursos están disponibles?" },
  { icon: <Users className="h-4 w-4" />, label: "Comunidad", prompt: "Cuéntame sobre la comunidad estudiantil" },
  { icon: <Calendar className="h-4 w-4" />, label: "Eventos", prompt: "¿Qué eventos están próximos?" },
  { icon: <Trophy className="h-4 w-4" />, label: "Logros", prompt: "¿Cómo puedo seguir mi progreso?" },
];

export function ChatLayout({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([WELCOME_MESSAGE]);
    }
  }, [isOpen, messages.length]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [...messages, userMessage],
        }),
      });

      if (!response.ok) {
        throw new Error("Error al obtener respuesta");
      }

      const data = await response.json();
      setMessages((prev) => [...prev, data]);
    } catch (error) {
      setMessages((prev) => [...prev, { 
        role: "assistant", 
        content: "Lo siento, hubo un error al procesar tu mensaje. Por favor, inténtalo de nuevo." 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickAction = (prompt: string) => {
    setInput(prompt);
  };

  return (
    <div className="relative min-h-screen">
      {children}
      
      {/* Botón de Chat */}
      <Button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 rounded-full w-12 h-12 p-0 bg-blue-600 hover:bg-blue-700"
      >
        <GraduationCap className="h-6 w-6" />
      </Button>

      {/* Ventana de Chat */}
      {isOpen && (
        <Card className="fixed bottom-20 right-4 w-96 shadow-lg border-blue-200">
          <div className="flex items-center justify-between p-4 border-b bg-blue-50 dark:bg-blue-950">
            <div className="flex items-center gap-2">
              <GraduationCap className="h-5 w-5 text-blue-600" />
              <div>
                <h3 className="font-semibold text-blue-900 dark:text-blue-100">Asistente Blatam IA Academy</h3>
                <Badge variant="secondary" className="text-xs bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">
                  Compañero de Aprendizaje IA
                </Badge>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsOpen(false)}
              className="h-8 w-8 hover:bg-blue-100 dark:hover:bg-blue-900"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <ScrollArea className="h-[400px] w-full p-4">
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`rounded-lg px-4 py-2 max-w-[80%] ${
                      message.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-blue-50 dark:bg-blue-900 text-blue-900 dark:text-blue-100"
                    }`}
                  >
                    {message.content}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="rounded-lg px-4 py-2 bg-blue-50 dark:bg-blue-900 text-blue-900 dark:text-blue-100">
                    Pensando...
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
          <div className="p-4 border-t bg-blue-50 dark:bg-blue-950">
            <div className="grid grid-cols-2 gap-2 mb-4">
              {QUICK_ACTIONS.map((action, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  className="justify-start gap-2 text-xs bg-white dark:bg-blue-900 hover:bg-blue-50 dark:hover:bg-blue-800"
                  onClick={() => handleQuickAction(action.prompt)}
                >
                  {action.icon}
                  {action.label}
                </Button>
              ))}
            </div>
            <form onSubmit={handleSubmit} className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Pregúntame cualquier cosa sobre Blatam Academy..."
                disabled={isLoading}
                className="border-blue-200 focus:border-blue-400"
              />
              <Button 
                type="submit" 
                disabled={isLoading}
                className="bg-blue-600 hover:bg-blue-700"
              >
                Enviar
              </Button>
            </form>
          </div>
        </Card>
      )}
    </div>
  );
}  