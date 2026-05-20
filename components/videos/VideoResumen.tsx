"use client";
import { motion } from "framer-motion";
import { BookOpen } from "lucide-react";

interface VideoResumenProps {
  resumen: {
    titulo: string;
    descripcion: string;
    puntos: string[];
  };
}

export default function VideoResumen({ resumen }: VideoResumenProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-zinc-900 rounded-2xl p-6 border border-zinc-800"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-primary/10 rounded-lg">
          <BookOpen className="w-6 h-6 text-primary" />
        </div>
        <h2 className="text-xl font-semibold text-white">Resumen de la Clase</h2>
      </div>

      <div className="space-y-6">
        {/* Título y descripción */}
        <div>
          <h3 className="text-lg font-medium text-white mb-2">{resumen.titulo}</h3>
          <p className="text-zinc-400">{resumen.descripcion}</p>
        </div>

        {/* Puntos clave */}
        {resumen.puntos.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-zinc-400 mb-3">Puntos Clave</h4>
            <ul className="space-y-2">
              {resumen.puntos.map((punto, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-start gap-2 text-zinc-300"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                  <span>{punto}</span>
                </motion.li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </motion.div>
  );
}    