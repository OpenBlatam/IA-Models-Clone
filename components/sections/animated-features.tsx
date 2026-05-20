"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useScroll, useTransform } from "framer-motion";
import { GraduationCap, BookOpen, Users, Trophy, ArrowRight, Sparkles, Code2, Brain, Zap, Terminal, Crown, Star, Diamond } from "lucide-react";
import { useRef, useState, useEffect } from "react";
import * as HoverCard from "@radix-ui/react-hover-card";
import * as Tooltip from "@radix-ui/react-tooltip";
import { cn } from "@/lib/utils";

export function AnimatedFeatures() {
  const ref = useRef(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [floatingElements, setFloatingElements] = useState<Array<{ left: number; top: number; duration: number; delay: number }>>([]);
  const [matrixElements, setMatrixElements] = useState<Array<{ x: number; duration: number; delay: number }>>([]);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"]
  });

  useEffect(() => {
    // Generar valores aleatorios solo en el cliente
    const floating = Array.from({ length: 12 }).map(() => ({
      left: Math.random() * 100,
      top: Math.random() * 100,
      duration: 5 + Math.random() * 5,
      delay: Math.random() * 2
    }));
    setFloatingElements(floating);

    const matrix = Array.from({ length: 15 }).map(() => ({
      x: Math.random() * 100,
      duration: Math.random() * 10 + 5,
      delay: Math.random() * 5
    }));
    setMatrixElements(matrix);
  }, []);

  const y = useTransform(scrollYProgress, [0, 1], ["0%", "20%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.5, 1], [0, 1, 0]);

  const features = [
    {
      icon: <Brain className="h-12 w-12" />,
      title: "Aprende IA y Machine Learning",
      description: "Domina las tecnologías más demandadas del mercado con proyectos prácticos.",
      color: "from-[#00F5A0] to-[#00D9F5]",
      tooltip: "Curso completo de IA y ML con Python",
      hoverContent: "Aprende desde los fundamentos hasta el desarrollo de modelos avanzados de IA. Incluye proyectos reales y certificación.",
      badge: "Premium"
    },
    {
      icon: <Code2 className="h-12 w-12" />,
      title: "Desarrollo Web Full Stack",
      description: "Conviértete en desarrollador web con las últimas tecnologías.",
      color: "from-[#00D9F5] to-[#00F5A0]",
      tooltip: "MERN Stack y más tecnologías modernas",
      hoverContent: "Aprende React, Node.js, MongoDB y más. Desarrolla aplicaciones web completas desde cero.",
      badge: "VIP"
    },
    {
      icon: <Zap className="h-12 w-12" />,
      title: "Mentoría Personalizada",
      description: "Recibe guía individual de expertos en la industria.",
      color: "from-[#00F5A0] to-[#00D9F5]",
      tooltip: "Sesiones 1:1 con mentores expertos",
      hoverContent: "Accede a mentores que trabajan en empresas líderes y recibe feedback personalizado.",
      badge: "Exclusivo"
    },
    {
      icon: <Trophy className="h-12 w-12" />,
      title: "Certificación y Empleo",
      description: "Obtén certificaciones reconocidas y acceso a oportunidades laborales.",
      color: "from-[#00D9F5] to-[#00F5A0]",
      tooltip: "Preparación para el mercado laboral",
      hoverContent: "Recibe ayuda con tu CV, entrevistas y conecta con empresas asociadas.",
      badge: "Premium"
    }
  ];

  return (
    <section ref={ref} className="py-32 px-4 relative overflow-hidden bg-[#0A0A0A]">
      <div className="absolute inset-0 bg-gradient-to-b from-[#0A0A0A] to-[#1A1A1A]" />
      
      {/* Luxury Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, #00F5A0 1px, transparent 0)`,
          backgroundSize: '40px 40px'
        }} />
      </div>
      
      {/* Animated Background Elements */}
      <motion.div
        className="absolute top-0 left-0 w-full h-full"
        style={{ opacity: 0.1 }}
      >
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
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[#00D9F5] rounded-full filter blur-3xl"
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
      </motion.div>

      {/* Floating Elements */}
      {floatingElements.map((element, i) => (
        <motion.div
          key={i}
          className="absolute"
          style={{
            left: `${element.left}%`,
            top: `${element.top}%`,
          }}
          animate={{
            y: [0, -20, 0],
            rotate: [0, 360],
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: element.duration,
            repeat: Infinity,
            ease: "easeInOut",
            delay: element.delay,
          }}
        >
          <div className={cn(
            "w-4 h-4 rounded-full bg-gradient-to-r from-[#00F5A0] to-[#00D9F5] opacity-20",
            i % 3 === 0 && "w-6 h-6",
            i % 4 === 0 && "w-8 h-8"
          )} />
        </motion.div>
      ))}

      {/* Matrix-like Code Rain Effect */}
      <div className="absolute inset-0 overflow-hidden">
        {matrixElements.map((element, i) => (
          <motion.div
            key={i}
            className="absolute text-[#00F5A0]/10 text-sm font-mono"
            initial={{ y: -100, x: `${element.x}%` }}
            animate={{ y: "100vh" }}
            transition={{
              duration: element.duration,
              repeat: Infinity,
              delay: element.delay,
            }}
          >
            {[...Array(10)].map((_, j) => (
              <span key={j} className="inline-block mx-1">
                {Math.random() > 0.5 ? "1" : "0"}
              </span>
            ))}
          </motion.div>
        ))}
      </div>

      <motion.div 
        style={{ y, opacity }}
        className="relative max-w-6xl mx-auto"
      >
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <motion.div 
            className="inline-flex items-center px-6 py-3 rounded-full bg-gradient-to-r from-[#00F5A0]/10 to-[#00D9F5]/10 text-[#00F5A0] mb-6 border border-[#00F5A0]/20 backdrop-blur-sm"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Diamond className="h-5 w-5 mr-2" />
            <span className="text-sm font-medium">Características Premium</span>
          </motion.div>
          <motion.h2 
            className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-[#00F5A0] to-[#00D9F5]"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
          >
            Experiencia de Aprendizaje Elite
          </motion.h2>
          <motion.p 
            className="text-xl text-gray-400 max-w-2xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            viewport={{ once: true }}
          >
            Descubre un nuevo nivel de excelencia en educación tecnológica
          </motion.p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <HoverCard.Root key={index} open={hoveredIndex === index} onOpenChange={(open) => setHoveredIndex(open ? index : null)}>
              <HoverCard.Trigger asChild>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="group relative cursor-pointer"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <motion.div 
                    className="absolute inset-0 bg-gradient-to-br from-[#1A1A1A] to-[#2A2A2A] rounded-2xl transform transition-transform group-hover:scale-105"
                    animate={{
                      boxShadow: [
                        "0 0 0 0 rgba(0, 245, 160, 0)",
                        "0 0 30px 0 rgba(0, 245, 160, 0.2)",
                        "0 0 0 0 rgba(0, 245, 160, 0)",
                      ],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                  />
                  <div className="relative p-8 rounded-2xl border border-[#333333] bg-[#1A1A1A]/50 backdrop-blur-sm">
                    <div className="absolute top-4 right-4">
                      <motion.div
                        className="px-3 py-1 rounded-full bg-gradient-to-r from-[#00F5A0] to-[#00D9F5] text-black text-xs font-semibold"
                        whileHover={{ scale: 1.1 }}
                      >
                        {feature.badge}
                      </motion.div>
                    </div>
                    <motion.div 
                      className={`p-4 rounded-full bg-gradient-to-br ${feature.color} text-black mb-6 group-hover:scale-110 transition-transform duration-300`}
                      animate={{
                        rotate: [0, 360],
                      }}
                      transition={{
                        duration: 20,
                        repeat: Infinity,
                        ease: "linear",
                      }}
                    >
                      {feature.icon}
                    </motion.div>
                    <h3 className="text-xl font-semibold mb-3 text-white group-hover:text-[#00F5A0] transition-colors">
                      {feature.title}
                    </h3>
                    <p className="text-gray-400 mb-6">
                      {feature.description}
                    </p>
                    <motion.div 
                      className="flex items-center text-[#00F5A0] opacity-0 group-hover:opacity-100 transition-opacity"
                      whileHover={{ x: 5 }}
                    >
                      <span className="text-sm font-medium">Descubre más</span>
                      <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                    </motion.div>
                  </div>
                </motion.div>
              </HoverCard.Trigger>
              <HoverCard.Portal>
                <HoverCard.Content
                  className="w-80 p-4 bg-[#1A1A1A] rounded-lg shadow-lg border border-[#333333] backdrop-blur-sm"
                  sideOffset={5}
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold text-[#00F5A0]">{feature.title}</h4>
                      <Star className="h-4 w-4 text-[#00F5A0]" />
                    </div>
                    <p className="text-sm text-gray-400">{feature.hoverContent}</p>
                  </div>
                  <HoverCard.Arrow className="fill-[#1A1A1A]" />
                </HoverCard.Content>
              </HoverCard.Portal>
            </HoverCard.Root>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          viewport={{ once: true }}
          className="text-center mt-12"
        >
          <Tooltip.Provider>
            <Tooltip.Root>
              <Tooltip.Trigger asChild>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-8 py-4 rounded-full bg-gradient-to-r from-[#00F5A0] to-[#00D9F5] text-black font-semibold hover:shadow-lg hover:shadow-[#00F5A0]/20 transition-all duration-300"
                >
                  <div className="flex items-center">
                    <Crown className="h-5 w-5 mr-2" />
                    <span>Accede a Contenido Premium</span>
                  </div>
                </motion.button>
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content
                  className="px-3 py-2 bg-[#1A1A1A] text-white text-sm rounded-lg shadow-lg border border-[#333333] backdrop-blur-sm"
                  sideOffset={5}
                >
                  Desbloquea todo el contenido premium
                  <Tooltip.Arrow className="fill-[#1A1A1A]" />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
          </Tooltip.Provider>
        </motion.div>
      </motion.div>
    </section>
  );
}    