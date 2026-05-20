import React from "react";
import { motion } from "framer-motion";
import { Sparkles, Brain, Rocket, Target, Trophy, Flame, Gem } from "lucide-react";

const features = [
  {
    title: "AI-Powered Marketing",
    description: "Leverage advanced AI to create compelling marketing content that resonates with your audience.",
    icon: <Brain className="h-6 w-6" />,
  },
  {
    title: "Brand Control",
    description: "Maintain consistent brand voice and style across all your marketing materials.",
    icon: <Target className="h-6 w-6" />,
  },
  {
    title: "Workflow Automation",
    description: "Streamline your marketing processes with intelligent automation tools.",
    icon: <Rocket className="h-6 w-6" />,
  },
  {
    title: "Performance Analytics",
    description: "Track and optimize your marketing campaigns with detailed analytics.",
    icon: <Trophy className="h-6 w-6" />,
  },
];

export function MarketingFeatureCards() {
  return (
    <section className="relative w-full py-20 overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-radial opacity-30" />
      <div className="absolute inset-0 bg-dots opacity-20" />

      <div className="max-w-7xl mx-auto px-4 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Powerful Features for{" "}
            <span className="text-gradient">Modern Marketing</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Everything you need to create, manage, and optimize your marketing campaigns in one place.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <motion.div
            key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="group relative"
            >
              <div className="glass-effect rounded-xl p-6 hover-glow transition-all duration-300">
                <div className="mb-4 text-gradient">
                  {feature.icon}
            </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
          </div>
              <div className="absolute inset-0 bg-gradient opacity-0 group-hover:opacity-10 transition-opacity duration-300 rounded-xl" />
            </motion.div>
          ))}
        </div>

        {/* Floating Elements */}
        <motion.div
          className="absolute top-1/3 right-1/4 w-48 h-48 bg-primary/20 rounded-full filter blur-3xl"
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
          className="absolute bottom-1/3 left-1/4 w-48 h-48 bg-secondary/20 rounded-full filter blur-3xl"
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
    </section>
  );
}   