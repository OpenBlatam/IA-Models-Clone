"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { MessageSquare, X, GraduationCap, BookOpen, Users, Calendar, Trophy } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import Image from "next/image";
import { useToast } from "@/components/ui/use-toast";
import { ToastAction } from "@/components/ui/toast";
import { motion, AnimatePresence } from "framer-motion";
import { useSpring, animated } from "@react-spring/web";
import { useInView } from "react-intersection-observer";
import Link from "next/link";

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

const WaveEffect = () => {
  const { ref, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  const waveAnimation = useSpring({
    from: { transform: 'translateX(-100%)' },
    to: { transform: 'translateX(100%)' },
    loop: true,
    config: { duration: 4000 }
  });

  return (
    <div ref={ref} className="absolute inset-0 overflow-hidden rounded-2xl">
      <animated.div
        className="wave-animation"
        style={waveAnimation}
      />
    </div>
  );
};

const HeaderText = () => {
  const { ref, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  const textAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: 1, transform: 'translateY(0)' },
    config: { tension: 300, friction: 20 }
  });

  const heartAnimation = useSpring({
    from: { scale: 1, opacity: 0.8 },
    to: { scale: 1.15, opacity: 1 },
    loop: { reverse: true },
    config: { duration: 1200 }
  });

  return (
    <animated.div
      ref={ref}
      style={textAnimation}
      className="header-text"
    >
      iAcademy
      <animated.span style={heartAnimation} className="heart-icon">
        🤍
      </animated.span>
    </animated.div>
  );
};

// Estilos para la animación de la onda
const waveStyles = `
  @keyframes wave {
    0% {
      transform: translateX(-100%) translateY(0) scale(1.2);
    }
    50% {
      transform: translateX(0) translateY(-15%) scale(1.3);
    }
    100% {
      transform: translateX(100%) translateY(0) scale(1.2);
    }
  }

  @keyframes float {
    0% {
      transform: translateY(0px) rotate(0deg);
    }
    50% {
      transform: translateY(-15px) rotate(3deg);
    }
    100% {
      transform: translateY(0px) rotate(0deg);
    }
  }

  @keyframes neo-glow {
    0% {
      box-shadow: 4px 4px 8px rgba(227, 167, 196, 0.15),
                  -4px -4px 8px rgba(255, 255, 255, 0.9),
                  inset 1px 1px 2px rgba(227, 167, 196, 0.05),
                  inset -1px -1px 2px rgba(255, 255, 255, 0.95);
    }
    50% {
      box-shadow: 6px 6px 12px rgba(227, 167, 196, 0.2),
                  -6px -6px 12px rgba(255, 255, 255, 0.95),
                  inset 2px 2px 4px rgba(227, 167, 196, 0.1),
                  inset -2px -2px 4px rgba(255, 255, 255, 0.98);
    }
    100% {
      box-shadow: 4px 4px 8px rgba(227, 167, 196, 0.15),
                  -4px -4px 8px rgba(255, 255, 255, 0.9),
                  inset 1px 1px 2px rgba(227, 167, 196, 0.05),
                  inset -1px -1px 2px rgba(255, 255, 255, 0.95);
    }
  }

  @keyframes neo-bounce {
    0%, 100% {
      transform: scale(1);
      box-shadow: 4px 4px 8px rgba(227, 167, 196, 0.15),
                  -4px -4px 8px rgba(255, 255, 255, 0.9);
    }
    50% {
      transform: scale(1.03);
      box-shadow: 6px 6px 12px rgba(227, 167, 196, 0.2),
                  -6px -6px 12px rgba(255, 255, 255, 0.95);
    }
  }

  .wave-animation {
    position: absolute;
    width: 300%;
    height: 300%;
    background: linear-gradient(90deg, 
      rgba(227, 167, 196, 0) 0%,
      rgba(227, 167, 196, 0.1) 25%,
      rgba(227, 167, 196, 0.2) 50%,
      rgba(227, 167, 196, 0.1) 75%,
      rgba(227, 167, 196, 0) 100%
    );
    animation: wave 4s ease-in-out infinite;
    pointer-events: none;
    transform-origin: center;
    filter: blur(1px);
  }

  .notification-bubble {
    position: absolute;
    top: -45px;
    right: 0;
    background: linear-gradient(135deg, #E3A7C4 0%, #D48FB1 100%);
    color: white;
    padding: 12px 24px;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
    animation: float 3s ease-in-out infinite, neo-glow 2s ease-in-out infinite;
    white-space: nowrap;
    z-index: 50;
    border: 1px solid rgba(227, 167, 196, 0.2);
    backdrop-filter: blur(4px);
    transform-origin: bottom right;
    letter-spacing: 0.5px;
    box-shadow: 4px 4px 8px rgba(227, 167, 196, 0.15),
                -4px -4px 8px rgba(255, 255, 255, 0.9);
  }

  .notification-bubble::after {
    content: '';
    position: absolute;
    bottom: -8px;
    right: 24px;
    width: 16px;
    height: 16px;
    background: #E3A7C4;
    transform: rotate(45deg);
    border-right: 1px solid rgba(227, 167, 196, 0.2);
    border-bottom: 1px solid rgba(227, 167, 196, 0.2);
    box-shadow: 3px 3px 6px rgba(227, 167, 196, 0.15);
  }

  .chat-button-pulse {
    animation: neo-bounce 2s ease-in-out infinite;
    background: linear-gradient(135deg, #E3A7C4 0%, #D48FB1 100%) !important;
    border: 1px solid rgba(227, 167, 196, 0.2) !important;
    box-shadow: 6px 6px 12px rgba(227, 167, 196, 0.2),
                -6px -6px 12px rgba(255, 255, 255, 0.95);
  }

  .message-bubble {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    position: relative;
    overflow: hidden;
    box-shadow: 4px 4px 8px rgba(227, 167, 196, 0.15),
                -4px -4px 8px rgba(255, 255, 255, 0.95);
    padding: 16px 24px;
    font-size: 1rem;
    line-height: 1.5;
  }

  .message-bubble::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, 
      rgba(227, 167, 196, 0) 0%,
      rgba(227, 167, 196, 0.1) 50%,
      rgba(227, 167, 196, 0) 100%
    );
    opacity: 0;
    transition: opacity 0.3s ease;
  }

  .message-bubble:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 4px 4px 8px rgba(227, 167, 196, 0.15),
                -4px -4px 8px rgba(255, 255, 255, 0.9);
  }

  .message-bubble:hover::before {
    opacity: 1;
  }

  .message-bubble:active {
    transform: translateY(0) scale(0.99);
    box-shadow: 2px 2px 4px rgba(227, 167, 196, 0.1),
                -2px -2px 4px rgba(255, 255, 255, 0.8);
  }

  .quick-action-button {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(227, 167, 196, 0.15) !important;
    box-shadow: 2px 2px 4px rgba(227, 167, 196, 0.08),
                -2px -2px 4px rgba(255, 255, 255, 0.9);
  }

  .quick-action-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, 
      rgba(227, 167, 196, 0) 0%,
      rgba(227, 167, 196, 0.08) 50%,
      rgba(227, 167, 196, 0) 100%
    );
    opacity: 0;
    transition: opacity 0.3s ease;
  }

  .quick-action-button:hover::before {
    opacity: 1;
  }

  .quick-action-button:hover {
    box-shadow: 3px 3px 6px rgba(227, 167, 196, 0.12),
                -3px -3px 6px rgba(255, 255, 255, 0.95);
  }

  .luxury-input {
    border: 1px solid rgba(227, 167, 196, 0.15) !important;
    transition: all 0.3s ease;
    box-shadow: inset 2px 2px 4px rgba(227, 167, 196, 0.08),
                inset -2px -2px 4px rgba(255, 255, 255, 0.9);
  }

  .luxury-input:focus {
    border-color: #E3A7C4 !important;
    box-shadow: inset 3px 3px 6px rgba(227, 167, 196, 0.12),
                inset -3px -3px 6px rgba(255, 255, 255, 0.95) !important;
  }

  .luxury-button {
    background: linear-gradient(135deg, #E3A7C4 0%, #D48FB1 100%) !important;
    border: 1px solid rgba(227, 167, 196, 0.2) !important;
    color: white !important;
    box-shadow: 3px 3px 6px rgba(227, 167, 196, 0.15),
                -3px -3px 6px rgba(255, 255, 255, 0.9);
  }

  .luxury-button:hover {
    background: linear-gradient(135deg, #D48FB1 0%, #C47A9E 100%) !important;
    box-shadow: 4px 4px 8px rgba(227, 167, 196, 0.2),
                -4px -4px 8px rgba(255, 255, 255, 0.95) !important;
  }

  .luxury-button:active {
    box-shadow: inset 2px 2px 4px rgba(227, 167, 196, 0.15),
                inset -2px -2px 4px rgba(255, 255, 255, 0.9) !important;
  }

  @keyframes fadeInOut {
    0% {
      opacity: 0;
      transform: translateY(-10px);
    }
    20% {
      opacity: 1;
      transform: translateY(0);
    }
    80% {
      opacity: 1;
      transform: translateY(0);
    }
    100% {
      opacity: 0;
      transform: translateY(-10px);
    }
  }

  @keyframes neonFade {
    0% {
      opacity: 0;
      text-shadow: 0 0 5px rgba(227, 167, 196, 0),
                   0 0 10px rgba(227, 167, 196, 0),
                   0 0 15px rgba(227, 167, 196, 0);
    }
    20% {
      opacity: 1;
      text-shadow: 0 0 5px rgba(227, 167, 196, 0.5),
                   0 0 10px rgba(227, 167, 196, 0.3),
                   0 0 15px rgba(227, 167, 196, 0.2);
    }
    80% {
      opacity: 1;
      text-shadow: 0 0 5px rgba(227, 167, 196, 0.5),
                   0 0 10px rgba(227, 167, 196, 0.3),
                   0 0 15px rgba(227, 167, 196, 0.2);
    }
    100% {
      opacity: 0;
      text-shadow: 0 0 5px rgba(227, 167, 196, 0),
                   0 0 10px rgba(227, 167, 196, 0),
                   0 0 15px rgba(227, 167, 196, 0);
    }
  }

  @keyframes heartBeat {
    0% {
      transform: scale(1);
      opacity: 0.9;
    }
    50% {
      transform: scale(1.15);
      opacity: 1;
    }
    100% {
      transform: scale(1);
      opacity: 0.9;
    }
  }

  .header-text {
    animation: neonFade 2s ease-in-out forwards;
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    font-size: 2rem;
    font-weight: 700;
    color: #E3A7C4;
    white-space: nowrap;
    pointer-events: none;
    z-index: 1;
    letter-spacing: 2px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .heart-icon {
    font-size: 1.8rem;
    filter: drop-shadow(0 0 4px rgba(255, 255, 255, 0.6));
    animation: heartBeat 1.2s ease-in-out infinite;
    display: inline-block;
    transform-origin: center;
  }

  .header-content {
    position: relative;
    z-index: 2;
  }

  @keyframes slideIn {
    0% {
      transform: translateY(100%) scale(0.8);
      opacity: 0;
    }
    60% {
      transform: translateY(-10%) scale(1.05);
      opacity: 1;
    }
    100% {
      transform: translateY(0) scale(1);
      opacity: 1;
    }
  }

  @keyframes bounceIn {
    0% {
      transform: scale(0.3);
      opacity: 0;
    }
    50% {
      transform: scale(1.1);
      opacity: 1;
    }
    70% {
      transform: scale(0.9);
    }
    100% {
      transform: scale(1);
    }
  }
`;

export function ChatLayout({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [showNotification, setShowNotification] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([WELCOME_MESSAGE]);
      toast({
        title: "¡Bienvenido a Blatam Academy!",
        description: "Tu asistente IA está listo para ayudarte.",
        action: (
          <ToastAction altText="Cerrar" onClick={() => window.location.reload()}>
            Cerrar
          </ToastAction>
        ),
        duration: 5000,
      });
    }
  }, [isOpen, messages.length, toast]);

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
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickAction = (prompt: string) => {
    setInput(prompt);
  };

  return (
    <div className="relative min-h-screen">
      <style>{waveStyles}</style>
      {/* Header global IAcademy */}
      {/* Header eliminado, solo queda el de la página principal */}
      {children}
      
      {/* Botón de Chat */}
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ 
          type: "spring",
          stiffness: 400,
          damping: 17
        }}
        className="fixed bottom-4 right-4 z-[100]"
      >
        <AnimatePresence>
          {showNotification && !isOpen && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.8 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.8 }}
              transition={{
                type: "spring",
                stiffness: 500,
                damping: 30
              }}
              className="notification-bubble"
            >
              ¡Hazme clic! Soy tu IA
            </motion.div>
          )}
        </AnimatePresence>
        <motion.div
          whileHover={{ scale: 1.1, rotate: 5 }}
          whileTap={{ scale: 0.9, rotate: -5 }}
          transition={{
            type: "spring",
            stiffness: 400,
            damping: 10
          }}
        >
          <Button
            onClick={() => {
              setIsOpen(true);
              setShowNotification(false);
            }}
            className="rounded-full w-14 h-14 p-0 flex items-center justify-center transition-all duration-300 chat-button-pulse"
          >
            <Image
              src="/b_logo.png"
              alt="Blatam Logo"
              width={32}
              height={32}
              className="object-contain"
              priority
            />
          </Button>
        </motion.div>
      </motion.div>

      {/* Overlay y Ventana de Chat */}
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 pointer-events-none z-[90]"
            />
            <motion.div
              initial={{ opacity: 0, y: 100, scale: 0.8 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 100, scale: 0.8 }}
              transition={{
                type: "spring",
                stiffness: 400,
                damping: 30,
                mass: 0.8
              }}
            >
              <Card className="fixed bottom-16 right-4 w-[360px] rounded-2xl overflow-hidden bg-white/95 backdrop-blur-sm flex flex-col z-[100] max-h-[calc(100vh-6rem)] shadow-[4px_4px_8px_rgba(227,167,196,0.15),-4px_-4px_8px_rgba(255,255,255,0.9)]">
                <div className="flex items-center justify-between p-4 border-b border-[#E3A7C4] bg-gradient-to-r from-[#FFF5F7] to-white backdrop-blur-sm relative">
                  <HeaderText />
                  <div className="flex items-center gap-3 header-content">
                    <motion.div
                      whileHover={{ scale: 1.1, rotate: 5 }}
                      whileTap={{ scale: 0.9, rotate: -5 }}
                      transition={{
                        type: "spring",
                        stiffness: 400,
                        damping: 10
                      }}
                    >
                      <Image
                        src="/b_logo.png"
                        alt="Blatam Logo"
                        width={28}
                        height={28}
                        className="object-contain"
                        priority
                      />
                    </motion.div>
                    <div>
                      <h3 className="font-semibold text-[#521B41] text-base">Asistente Blatam IA</h3>
                      <Badge variant="secondary" className="text-xs bg-[#FFF5F7] backdrop-blur-sm text-[#E3A7C4] border border-[#E3A7C4] mt-0.5">
                        Compañero de Aprendizaje
                      </Badge>
                    </div>
                  </div>
                  <motion.div
                    whileHover={{ scale: 1.1, rotate: 5 }}
                    whileTap={{ scale: 0.9, rotate: -5 }}
                    transition={{
                      type: "spring",
                      stiffness: 400,
                      damping: 10
                    }}
                  >
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setIsOpen(false)}
                      className="h-8 w-8 hover:bg-[#FFF5F7] text-[#E3A7C4] transition-colors duration-200"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </motion.div>
                </div>
                <ScrollArea className="flex-1 w-full p-4 bg-white/95 backdrop-blur-sm">
                  <div className="space-y-4">
                    <AnimatePresence mode="popLayout">
                      {messages.map((message, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, y: 20, scale: 0.9 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={{ opacity: 0, y: -20, scale: 0.9 }}
                          transition={{
                            type: "spring",
                            stiffness: 500,
                            damping: 30,
                            mass: 0.5
                          }}
                          className={`flex ${
                            message.role === "user" ? "justify-end" : "justify-start"
                          }`}
                        >
                          <motion.div
                            whileHover={{ scale: 1.02, y: -2 }}
                            whileTap={{ scale: 0.98 }}
                            transition={{
                              type: "spring",
                              stiffness: 400,
                              damping: 17
                            }}
                            className={`message-bubble relative rounded-2xl px-4 py-2 max-w-[85%] ${
                              message.role === "user"
                                ? "bg-gradient-to-r from-[#E3A7C4] to-[#D48FB1] text-white border border-[#E3A7C4]"
                                : "bg-white text-gray-900 border border-[#E3A7C4]"
                            }`}
                          >
                            {message.role === "user" && <WaveEffect />}
                            <div className="relative z-10">{message.content}</div>
                          </motion.div>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                    {isLoading && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{
                          type: "spring",
                          stiffness: 500,
                          damping: 30
                        }}
                        className="flex justify-start"
                      >
                        <div className="message-bubble relative rounded-2xl px-4 py-2 bg-white text-gray-900 border border-[#E3A7C4]">
                          <WaveEffect />
                          <div className="relative z-10">Pensando...</div>
                        </div>
                      </motion.div>
                    )}
                  </div>
                </ScrollArea>
                <div className="p-4 border-t border-[#E3A7C4] bg-gradient-to-b from-[#FFF5F7] to-white backdrop-blur-sm">
                  <div className="grid grid-cols-2 gap-2 mb-3">
                    {QUICK_ACTIONS.map((action, index) => (
                      <motion.div
                        key={index}
                        whileHover={{ scale: 1.02, y: -2 }}
                        whileTap={{ scale: 0.98 }}
                        transition={{
                          type: "spring",
                          stiffness: 400,
                          damping: 17
                        }}
                      >
                        <Button
                          variant="outline"
                          size="sm"
                          className="quick-action-button justify-start gap-2 text-xs bg-white hover:bg-[#FFF5F7] text-[#E3A7C4] transition-all duration-200 py-3 w-full"
                          onClick={() => handleQuickAction(action.prompt)}
                        >
                          {action.icon}
                          {action.label}
                        </Button>
                      </motion.div>
                    ))}
                  </div>
                  <form onSubmit={handleSubmit} className="flex gap-2">
                    <motion.div
                      whileFocus={{ scale: 1.02 }}
                      className="flex-1"
                    >
                      <Input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Pregúntame cualquier cosa..."
                        disabled={isLoading}
                        className="luxury-input bg-white focus:ring-0 transition-all duration-200 h-10 text-sm"
                      />
                    </motion.div>
                    <motion.div
                      whileHover={{ scale: 1.05, rotate: 5 }}
                      whileTap={{ scale: 0.95, rotate: -5 }}
                      transition={{
                        type: "spring",
                        stiffness: 400,
                        damping: 10
                      }}
                    >
                      <Button 
                        type="submit" 
                        disabled={isLoading}
                        className="luxury-button transition-all duration-200 h-10 px-4 text-sm"
                      >
                        Enviar
                      </Button>
                    </motion.div>
                  </form>
                </div>
              </Card>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
