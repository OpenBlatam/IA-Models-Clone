import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Icons } from "@/components/shared/icons";
import { cn } from "@/lib/utils";

interface Exercise {
  id: string;
  type: "multiple-choice" | "translation" | "listening" | "speaking";
  question: string;
  correctAnswer: string;
  options?: string[];
  audioUrl?: string;
  points: number;
}

interface ExerciseCardProps {
  exercise: Exercise;
  onComplete: (result: { correct: boolean; points: number }) => void;
  className?: string;
}

export function ExerciseCard({
  exercise,
  onComplete,
  className,
}: ExerciseCardProps) {
  const [userAnswer, setUserAnswer] = useState("");
  const [feedback, setFeedback] = useState<{
    correct: boolean;
    message: string;
  } | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!userAnswer.trim()) return;

    setIsSubmitting(true);
    const isCorrect = exercise.correctAnswer.toLowerCase() === userAnswer.toLowerCase();
    
    setFeedback({
      correct: isCorrect,
      message: isCorrect ? "¡Correcto! 🎉" : `Incorrecto. La respuesta correcta era: ${exercise.correctAnswer}`,
    });

    onComplete({
      correct: isCorrect,
      points: isCorrect ? exercise.points : 0,
    });

    setIsSubmitting(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isSubmitting) {
      handleSubmit();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={cn(
        "p-6 bg-white rounded-lg shadow-md border border-gray-200",
        className
      )}
    >
      <h3 className="text-xl font-bold mb-4">{exercise.question}</h3>

      {exercise.type === "multiple-choice" && (
        <div className="space-y-2">
          {exercise.options?.map((option) => (
            <button
              key={option}
              onClick={() => setUserAnswer(option)}
              className={cn(
                "w-full p-3 text-left border rounded-md transition-colors",
                userAnswer === option
                  ? "bg-blue-50 border-blue-500"
                  : "hover:bg-gray-50"
              )}
            >
              {option}
            </button>
          ))}
        </div>
      )}

      {exercise.type === "translation" && (
        <div className="space-y-4">
          <input
            type="text"
            value={userAnswer}
            onChange={(e) => setUserAnswer(e.target.value)}
            onKeyPress={handleKeyPress}
            className="w-full p-3 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="Escribe tu respuesta..."
            disabled={isSubmitting}
          />
        </div>
      )}

      <AnimatePresence>
        {feedback && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={cn(
              "mt-4 p-3 rounded-md",
              feedback.correct ? "bg-green-100" : "bg-red-100"
            )}
          >
            {feedback.message}
          </motion.div>
        )}
      </AnimatePresence>

      <button
        onClick={handleSubmit}
        disabled={isSubmitting || !userAnswer.trim()}
        className={cn(
          "mt-4 w-full bg-blue-500 text-white py-2 rounded-md transition-colors",
          "hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
        )}
      >
        {isSubmitting ? (
          <Icons.spinner className="w-5 h-5 mx-auto animate-spin" />
        ) : (
          "Verificar Respuesta"
        )}
      </button>
    </motion.div>
  );
}    