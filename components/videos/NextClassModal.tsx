"use client";
import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, CheckCircle2 } from "lucide-react";
import Image from "next/image";
import { Button } from "@/components/ui/button";

interface NextClassModalProps {
  open: boolean;
  onClose: () => void;
  onNext: () => void;
  onComplete: () => void;
}

const NextClassModal: React.FC<NextClassModalProps> = ({
  open,
  onClose,
  onNext,
  onComplete,
}) => {
  if (!open) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-zinc-900 rounded-2xl w-full max-w-2xl overflow-hidden border border-zinc-800 shadow-2xl"
        >
          <div className="p-6">
            <div className="text-center mb-6">
              <h2 className="text-2xl font-bold text-white mb-2">
                ¡Clase Completada!
              </h2>
              <p className="text-zinc-400">
                ¿Qué te gustaría hacer a continuación?
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Botón de siguiente clase */}
              <Button
                onClick={onNext}
                className="h-auto p-4 bg-primary hover:bg-primary/90 text-white"
              >
                <Play className="w-5 h-5 mr-2" />
                <div className="text-left">
                  <div className="font-semibold">Siguiente Clase</div>
                  <div className="text-sm text-white/80">
                    Continuar con el siguiente tema
                  </div>
                </div>
              </Button>

              {/* Botón de completar curso */}
              <Button
                onClick={onComplete}
                className="h-auto p-4 bg-zinc-800 hover:bg-zinc-700 text-white"
              >
                <CheckCircle2 className="w-5 h-5 mr-2" />
                <div className="text-left">
                  <div className="font-semibold">Completar Curso</div>
                  <div className="text-sm text-white/80">
                    Finalizar y obtener certificado
                  </div>
                </div>
              </Button>
            </div>

            {/* Botón de cerrar */}
            <div className="mt-6 text-center">
              <Button
                variant="ghost"
                onClick={onClose}
                className="text-zinc-400 hover:text-white"
              >
                Cerrar
              </Button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default NextClassModal;    