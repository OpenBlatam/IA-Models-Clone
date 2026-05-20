"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send } from "lucide-react";

interface VideoQuestionBarProps {
  onAsk: (question: string) => void;
}

export function VideoQuestionBar({ onAsk }: VideoQuestionBarProps) {
  const [question, setQuestion] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim()) {
      onAsk(question);
      setQuestion("");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h2 className="text-xl font-bold text-foreground">Preguntas</h2>
      <div className="flex gap-4">
        <Input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Escribe tu pregunta..."
          className="bg-muted border-border text-foreground placeholder:text-muted-foreground"
        />
        <Button
          type="submit"
          className="bg-primary hover:bg-primary/90 text-primary-foreground"
          disabled={!question.trim()}
        >
          <Send className="w-4 h-4 mr-2" />
          Enviar
        </Button>
      </div>
    </form>
  );
} 