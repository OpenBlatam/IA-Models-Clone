"use client";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { motion, AnimatePresence } from "framer-motion";
import { useSession, signIn } from "next-auth/react";
import { formatDistanceToNow } from "date-fns";
import { es } from "date-fns/locale";
import { LoadingState } from "@/components/ui/loading-state";
import { useVideoQuestions } from "@/lib/hooks/useVideoQuestions";
import { MessageSquare, Loader2, Bot } from "lucide-react";

interface VideoQuestionsProps {
  videoId: string;
  courseId: string;
}

export function VideoQuestions({ videoId, courseId }: VideoQuestionsProps) {
  const { data: session } = useSession();
  const { questions, isLoading, error, askQuestion } = useVideoQuestions(videoId, courseId);
  const [newQuestion, setNewQuestion] = useState("");

  if (isLoading) {
    return <LoadingState text="Cargando preguntas..." />;
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-destructive">{error}</p>
      </div>
    );
  }

  const handleAskQuestion = async () => {
    if (!newQuestion.trim()) return;
    await askQuestion(newQuestion);
    setNewQuestion("");
  };

  return (
    <div className="space-y-6">
      {/* Question Input */}
      {session?.user ? (
        <div className="flex gap-4">
          <Avatar className="w-10 h-10">
            <AvatarImage src={session.user.image || undefined} />
            <AvatarFallback>
              {session.user.name?.charAt(0) || "U"}
            </AvatarFallback>
          </Avatar>
          <div className="flex-1 space-y-4">
            <Textarea
              placeholder="¿Qué pregunta tienes sobre esta clase?"
              value={newQuestion}
              onChange={(e) => setNewQuestion(e.target.value)}
              className="min-h-[80px] resize-none"
            />
            <div className="flex justify-end gap-2">
              <Button
                variant="ghost"
                onClick={() => setNewQuestion("")}
                disabled={!newQuestion.trim()}
              >
                Cancelar
              </Button>
              <Button
                onClick={handleAskQuestion}
                disabled={!newQuestion.trim()}
              >
                Preguntar
              </Button>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-4 bg-muted rounded-lg text-center">
          <p className="text-sm text-muted-foreground mb-2">
            Inicia sesión para hacer preguntas
          </p>
          <Button variant="default" onClick={() => signIn()}>
            Iniciar sesión
          </Button>
        </div>
      )}

      {/* Questions List */}
      <div className="space-y-6">
        <AnimatePresence>
          {questions.map((question) => (
            <motion.div
              key={question.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex gap-4"
            >
              <Avatar className="w-10 h-10">
                <AvatarImage src={question.user.image} />
                <AvatarFallback>
                  {question.user.name.charAt(0)}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{question.user.name}</span>
                  <span className="text-sm text-muted-foreground">
                    {formatDistanceToNow(new Date(question.createdAt), {
                      addSuffix: true,
                      locale: es,
                    })}
                  </span>
                </div>
                <p className="text-sm">{question.content}</p>

                {/* Answer Section */}
                {question.status === 'pending' && (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Procesando pregunta...
                  </div>
                )}

                {question.status === 'error' && (
                  <div className="text-sm text-destructive">
                    Error al procesar la pregunta. Por favor, intenta de nuevo.
                  </div>
                )}

                {question.status === 'answered' && question.answer && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className="mt-4"
                  >
                    <div className="flex items-start gap-3 p-4 bg-primary/5 rounded-lg border border-primary/10">
                      <div className="flex-shrink-0">
                        <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                          <Bot className="w-4 h-4 text-primary" />
                        </div>
                      </div>
                      <div className="flex-1 space-y-2">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-primary">DeepSeek AI</span>
                          <span className="text-xs text-muted-foreground">
                            {formatDistanceToNow(new Date(question.updatedAt), {
                              addSuffix: true,
                              locale: es,
                            })}
                          </span>
                        </div>
                        <p className="text-sm leading-relaxed">{question.answer}</p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Empty State */}
        {questions.length === 0 && (
          <div className="text-center py-8">
            <p className="text-muted-foreground">
              No hay preguntas aún. ¡Sé el primero en preguntar!
            </p>
          </div>
        )}
      </div>
    </div>
  );
}    