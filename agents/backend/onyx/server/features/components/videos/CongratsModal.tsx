"use client";
import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Trophy, Share2, Download } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CongratsModalProps {
  open: boolean;
  onClose: () => void;
  courseTitle: string;
}

const CongratsModal: React.FC<CongratsModalProps> = ({
  open,
  onClose,
  courseTitle,
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
              <div className="w-20 h-20 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <Trophy className="w-10 h-10 text-primary" />
              </div>
              <h2 className="text-2xl font-bold text-white mb-2">
                ¡Felicidades!
              </h2>
              <p className="text-zinc-400">
                Has completado el curso "{courseTitle}"
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Botón de compartir */}
              <Button
                onClick={() => {
                  // Aquí iría la lógica para compartir
                }}
                className="h-auto p-4 bg-zinc-800 hover:bg-zinc-700 text-white"
              >
                <Share2 className="w-5 h-5 mr-2" />
                <div className="text-left">
                  <div className="font-semibold">Compartir Logro</div>
                  <div className="text-sm text-white/80">
                    Comparte tu certificado en redes sociales
                  </div>
                </div>
              </Button>

              {/* Botón de descargar */}
              <Button
                onClick={() => {
                  // Aquí iría la lógica para descargar el certificado
                }}
                className="h-auto p-4 bg-primary hover:bg-primary/90 text-white"
              >
                <Download className="w-5 h-5 mr-2" />
                <div className="text-left">
                  <div className="font-semibold">Descargar Certificado</div>
                  <div className="text-sm text-white/80">
                    Guarda tu certificado en PDF
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

export default CongratsModal;    