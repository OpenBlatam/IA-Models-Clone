"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowRight, Sparkles, Star, Zap, ChevronDown, Code2, Brain, Terminal, Rocket, Target, Trophy, Flame, Gem } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useScroll, useTransform } from "framer-motion";
import { useRef, useState, useEffect } from "react";

export function AnimatedHero() {
  const ref = useRef(null);
  const [isHovered, setIsHovered] = useState(false);
  const [matrixLines, setMatrixLines] = useState<Array<{ id: number; x: number; duration: number; delay: number }>>([]);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"]
  });

  useEffect(() => {
    // Generar los valores aleatorios solo en el cliente
    const lines = Array.from({ length: 20 }).map((_, i) => ({
      id: i,
      x: (i * 5) % 100,
      duration: Math.random() * 5 + 5,
      delay: Math.random() * 5
    }));
    setMatrixLines(lines);
  }, []);

  const y = useTransform(scrollYProgress, [0, 1], ["0%", "50%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);

  const stats = [
    { label: "Estudiantes", value: "1000+", icon: <Brain className="h-5 w-5" /> },
    { label: "Cursos", value: "50+", icon: <Code2 className="h-5 w-5" /> },
    { label: "Instructores", value: "100+", icon: <Star className="h-5 w-5" /> }
  ];

  return (
    <section ref={ref} className="relative min-h-screen flex items-center justify-center overflow-hidden bg-[#0A0A0A]">
      {/* Luxury Gradient Border */}
      <div className="absolute inset-0 pointer-events-none z-10">
        <div className="absolute inset-0 rounded-b-[3rem] border-t-8 border-x-8 border-b-0 border-transparent bg-gradient-to-r from-[#00F5A0] via-[#00D9F5] to-[#FFD700] opacity-60 blur-[2px]" />
      </div>
      {/* Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-b from-[#0A0A0A] to-[#1A1A1A]" />
        <div className="absolute inset-0 opacity-5">
          <div className="absolute inset-0" style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, #00F5A0 1px, transparent 0)`,
            backgroundSize: '40px 40px'
          }} />
        </div>
        <motion.div
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-[#00F5A0] rounded-full filter blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3],
            x: [0, 20, 0],
            y: [0, -20, 0],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <motion.div
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[#FFD700]/60 rounded-full filter blur-3xl"
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.5, 0.3, 0.5],
            x: [0, -20, 0],
            y: [0, 20, 0],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </div>
      {/* Matrix Code Rain Effect */}
      <div className="absolute inset-0 overflow-hidden z-0">
        <div className="absolute inset-0 opacity-20">
          {matrixLines.map((line) => (
            <motion.div
              key={`matrix-line-${line.id}`}
              className="absolute text-[#00F5A0] text-sm font-mono"
              initial={{ y: -100, x: `${line.x}%` }}
              animate={{ y: 1000 }}
              transition={{
                duration: line.duration,
                repeat: Infinity,
                delay: line.delay,
              }}
            >
              {`matrix-${line.id}-${Math.floor(Math.random() * 1000)}`}
            </motion.div>
          ))}
        </div>
      </div>
      <motion.div 
        style={{ y, opacity }}
        className="relative max-w-6xl mx-auto px-4 text-center z-20"
      >
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-8"
        >
          <motion.div 
            className="inline-flex items-center px-6 py-3 rounded-full bg-gradient-to-r from-[#FFD700]/20 via-[#00F5A0]/10 to-[#00D9F5]/10 text-[#FFD700] mb-6 border border-[#FFD700]/30 shadow-lg shadow-[#FFD700]/10 backdrop-blur-sm"
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.97 }}
          >
            <Gem className="h-5 w-5 mr-2 text-[#FFD700] animate-pulse" />
            <span className="text-sm font-medium tracking-widest uppercase">Formación Elite</span>
          </motion.div>
          <motion.h1 
            className="text-6xl md:text-7xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-[#FFD700] via-[#00F5A0] to-[#00D9F5] drop-shadow-[0_2px_24px_#FFD70055]"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Domina la Inteligencia Artificial y el Desarrollo Web de Alto Nivel
          </motion.h1>
          <motion.p 
            className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto mb-8 font-light"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Impulsa tu carrera aprendiendo las tecnologías más demandadas, con proyectos prácticos, mentoría personalizada y una comunidad de excelencia.
          </motion.p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="flex flex-col sm:flex-row gap-4 justify-center mb-16"
        >
          <Link href="/courses">
            <Button
              size="lg"
              className="bg-gradient-to-r from-[#FFD700] via-[#00F5A0] to-[#00D9F5] text-black font-bold shadow-lg hover:shadow-[#FFD700]/30 hover:scale-105 transition-all duration-300 border-2 border-[#FFD700]/40"
            >
              <span>Explorar Cursos</span>
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
          <Link href="/contact">
            <Button
              size="lg"
              variant="outline"
              className="border-[#FFD700] text-[#FFD700] hover:bg-[#FFD700]/10 font-bold hover:scale-105 transition-all duration-300"
            >
              <span>Contactar Mentor</span>
              <Flame className="ml-2 h-5 w-5 animate-pulse text-[#FFD700]" />
            </Button>
          </Link>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto"
        >
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="p-6 rounded-2xl bg-[#1A1A1A]/60 backdrop-blur-md border border-[#FFD700]/30 shadow-md"
          >
            <div className="flex items-center justify-center w-12 h-12 rounded-full bg-[#FFD700]/20 mb-4">
              <Brain className="h-6 w-6 text-[#FFD700]" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">IA Avanzada</h3>
            <p className="text-gray-300">Domina Machine Learning, Deep Learning y NLP con proyectos reales.</p>
          </motion.div>
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="p-6 rounded-2xl bg-[#1A1A1A]/60 backdrop-blur-md border border-[#FFD700]/30 shadow-md"
          >
            <div className="flex items-center justify-center w-12 h-12 rounded-full bg-[#FFD700]/20 mb-4">
              <Code2 className="h-6 w-6 text-[#FFD700]" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Desarrollo Web</h3>
            <p className="text-gray-300">Full Stack con las tecnologías más modernas y demandadas.</p>
          </motion.div>
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="p-6 rounded-2xl bg-[#1A1A1A]/60 backdrop-blur-md border border-[#FFD700]/30 shadow-md"
          >
            <div className="flex items-center justify-center w-12 h-12 rounded-full bg-[#FFD700]/20 mb-4">
              <Trophy className="h-6 w-6 text-[#FFD700]" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Certificación</h3>
            <p className="text-gray-300">Obtén certificaciones reconocidas y prepárate para el mercado laboral.</p>
          </motion.div>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1 }}
          className="mt-16 flex items-center justify-center space-x-8 text-[#FFD700] font-semibold"
        >
          <div className="flex items-center">
            <Zap className="h-5 w-5 mr-2 animate-pulse" />
            <span>+1000 Estudiantes</span>
          </div>
          <div className="flex items-center">
            <Terminal className="h-5 w-5 mr-2 animate-pulse" />
            <span>+50 Cursos</span>
          </div>
          <div className="flex items-center">
            <Target className="h-5 w-5 mr-2 animate-pulse" />
            <span>+20 Mentores</span>
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
}    