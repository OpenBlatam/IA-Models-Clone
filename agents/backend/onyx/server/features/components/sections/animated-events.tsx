"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useScroll, useTransform } from "framer-motion";
import { Calendar, Clock, MapPin, Users, ArrowRight, Sparkles, Code2, Brain, Zap, Terminal, Rocket, Crown, Star, Trophy } from "lucide-react";
import { useRef, useState, useEffect } from "react";
import * as HoverCard from "@radix-ui/react-hover-card";
import * as Tooltip from "@radix-ui/react-tooltip";
import { cn } from "@/lib/utils";

export function AnimatedEvents() {
  const ref = useRef(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [isMounted, setIsMounted] = useState(false);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"]
  });

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const y = useTransform(scrollYProgress, [0, 1], ["0%", "20%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.5, 1], [0, 1, 0]);

  const events = [
    {
      date: "15 de Marzo, 2024",
      time: "18:00 - 20:00",
      location: "Online",
      attendees: "150+",
      title: "Taller de IA y Machine Learning",
      description: "Aprende los fundamentos de la IA y ML con Python",
      link: "#",
      color: "from-[#00F5A0] to-[#00D9F5]",
      icon: <Brain className="h-12 w-12" />,
      hoverContent: "Taller práctico donde aprenderás a implementar modelos de machine learning desde cero. Incluye certificado de participación.",
      badge: "Premium"
    },
    {
      date: "20 de Marzo, 2024",
      time: "19:00 - 21:00",
      location: "Online",
      attendees: "200+",
      title: "Seminario de Desarrollo Web",
      description: "Domina las últimas tecnologías web",
      link: "#",
      color: "from-[#00D9F5] to-[#00F5A0]",
      icon: <Code2 className="h-12 w-12" />,
      hoverContent: "Seminario intensivo sobre React, Node.js y bases de datos. Aprende a construir aplicaciones web modernas.",
      badge: "VIP"
    },
    {
      date: "25 de Marzo, 2024",
      time: "17:00 - 22:00",
      location: "Online",
      attendees: "100+",
      title: "Hackathon de Innovación",
      description: "Desarrolla proyectos innovadores con IA",
      link: "#",
      color: "from-[#00F5A0] to-[#00D9F5]",
      icon: <Zap className="h-12 w-12" />,
      hoverContent: "Hackathon donde podrás desarrollar proyectos innovadores utilizando IA y tecnologías web. Premios para los mejores proyectos.",
      badge: "Exclusivo"
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

      {/* Floating Elements - Only render on client side */}
      {isMounted && [...Array(12)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute"
          style={{
            left: `${(i * 8.33) % 100}%`,
            top: `${(i * 7.5) % 100}%`,
          }}
          animate={{
            y: [0, -20, 0],
            rotate: [0, 360],
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 5 + (i % 5),
            repeat: Infinity,
            ease: "easeInOut",
            delay: i * 0.2,
          }}
        >
          <div className={cn(
            "w-4 h-4 rounded-full bg-gradient-to-r from-[#00F5A0] to-[#00D9F5] opacity-20",
            i % 3 === 0 && "w-6 h-6",
            i % 4 === 0 && "w-8 h-8"
          )} />
        </motion.div>
      ))}

      {/* Matrix-like Code Rain Effect - Only render on client side */}
      {isMounted && (
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute text-[#00F5A0]/10 text-sm font-mono"
              initial={{ y: -100, x: `${(i * 6.66) % 100}%` }}
              animate={{ y: "100vh" }}
              transition={{
                duration: 5 + (i % 5),
                repeat: Infinity,
                delay: i * 0.3,
              }}
            >
              {[...Array(10)].map((_, j) => (
                <span key={j} className="inline-block mx-1">
                  {(i + j) % 2 === 0 ? "1" : "0"}
                </span>
              ))}
            </motion.div>
          ))}
        </div>
      )}

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
            <Crown className="h-5 w-5 mr-2" />
            <span className="text-sm font-medium">Eventos Exclusivos</span>
          </motion.div>
          <motion.h2 
            className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-[#00F5A0] to-[#00D9F5]"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
          >
            Experiencias Premium
          </motion.h2>
          <motion.p 
            className="text-xl text-gray-400 max-w-2xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            viewport={{ once: true }}
          >
            Únete a eventos exclusivos con expertos líderes en la industria
          </motion.p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {events.map((event, index) => (
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
                        {event.badge}
                      </motion.div>
                    </div>
                    <motion.div 
                      className={`p-4 rounded-full bg-gradient-to-br ${event.color} text-black mb-6 group-hover:scale-110 transition-transform duration-300`}
                      animate={{
                        rotate: [0, 360],
                      }}
                      transition={{
                        duration: 20,
                        repeat: Infinity,
                        ease: "linear",
                      }}
                    >
                      {event.icon}
                    </motion.div>
                    <h3 className="text-xl font-semibold mb-3 text-white group-hover:text-[#00F5A0] transition-colors">
                      {event.title}
                    </h3>
                    <p className="text-gray-400 mb-6">
                      {event.description}
                    </p>
                    <div className="space-y-4">
                      <div className="flex items-center text-gray-400">
                        <Calendar className="h-4 w-4 mr-2" />
                        <span className="text-sm">{event.date}</span>
                      </div>
                      <div className="flex items-center text-gray-400">
                        <Clock className="h-4 w-4 mr-2" />
                        <span className="text-sm">{event.time}</span>
                      </div>
                      <div className="flex items-center text-gray-400">
                        <MapPin className="h-4 w-4 mr-2" />
                        <span className="text-sm">{event.location}</span>
                      </div>
                      <div className="flex items-center text-gray-400">
                        <Users className="h-4 w-4 mr-2" />
                        <span className="text-sm">{event.attendees} asistentes</span>
                      </div>
                    </div>
                    <motion.div 
                      className="flex items-center text-[#00F5A0] mt-6 opacity-0 group-hover:opacity-100 transition-opacity"
                      whileHover={{ x: 5 }}
                    >
                      <span className="text-sm font-medium">Reserva tu lugar</span>
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
                      <h4 className="font-semibold text-[#00F5A0]">{event.title}</h4>
                      <Trophy className="h-4 w-4 text-[#00F5A0]" />
                    </div>
                    <p className="text-sm text-gray-400">{event.hoverContent}</p>
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
                    <Star className="h-5 w-5 mr-2" />
                    <span>Accede a Eventos VIP</span>
                  </div>
                </motion.button>
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content
                  className="px-3 py-2 bg-[#1A1A1A] text-white text-sm rounded-lg shadow-lg border border-[#333333] backdrop-blur-sm"
                  sideOffset={5}
                >
                  Desbloquea acceso a eventos exclusivos
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