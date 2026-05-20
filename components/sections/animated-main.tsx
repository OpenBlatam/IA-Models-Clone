"use client";

import { motion } from "framer-motion";
import { useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import { AnimatedHero } from "./animated-hero";
import { AnimatedFeatures } from "./animated-features";
import { AnimatedEvents } from "./animated-events";
import { AnimatedTestimonials } from "./animated-testimonials";

export function AnimatedMain() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end end"]
  });

  const opacity = useTransform(scrollYProgress, [0, 0.1, 0.9, 1], [0, 1, 1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.1, 0.9, 1], [0.8, 1, 1, 0.8]);

  return (
    <motion.main
      ref={ref}
      style={{ opacity, scale }}
      className="relative min-h-screen bg-[#0A0A0A]"
    >
      {/* Background Elements */}
      <div className="fixed inset-0 pointer-events-none">
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
      </div>

      {/* Content Sections */}
      <div className="relative">
        <AnimatedHero />
        <AnimatedFeatures />
        <AnimatedEvents />
        <AnimatedTestimonials />
      </div>

      {/* Scroll Progress Indicator */}
      <motion.div
        className="fixed bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-[#00F5A0] to-[#00D9F5] origin-left"
        style={{ scaleX: scrollYProgress }}
      />
    </motion.main>
  );
}    