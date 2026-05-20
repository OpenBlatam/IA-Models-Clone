<<<<<<< HEAD
import React from "react";
import { motion } from "framer-motion";
import Image from "next/image";

const testimonials = [
  {
    content: "This platform has transformed how we handle our marketing content. The AI capabilities are incredible, and the brand voice control is exactly what we needed.",
    author: "Sarah Johnson",
    role: "Marketing Director",
    company: "TechCorp",
    image: "/testimonials/sarah.jpg",
  },
  {
    content: "The automation features have saved us countless hours. Our team can now focus on strategy while the platform handles the content creation.",
    author: "Michael Chen",
    role: "Head of Growth",
    company: "StartupX",
    image: "/testimonials/michael.jpg",
  },
  {
    content: "The analytics and insights have helped us optimize our campaigns like never before. It's like having a data scientist on our team.",
    author: "Emily Rodriguez",
    role: "Digital Marketing Manager",
    company: "GrowthLabs",
    image: "/testimonials/emily.jpg",
  },
];

export function MarketingTestimonialsSection() {
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
            Loved by{" "}
            <span className="text-gradient">Marketers</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            See what our customers have to say about their experience with our platform.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={testimonial.author}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="group relative"
            >
              <div className="glass-effect rounded-xl p-8 hover-glow transition-all duration-300">
                <div className="flex items-center mb-6">
                  <div className="relative w-12 h-12 rounded-full overflow-hidden mr-4">
                    <Image
                      src={testimonial.image}
                      alt={testimonial.author}
                      fill
                      className="object-cover"
                    />
                  </div>
                  <div>
                    <h3 className="font-semibold">{testimonial.author}</h3>
                    <p className="text-sm text-muted-foreground">
                      {testimonial.role} at {testimonial.company}
                    </p>
                  </div>
                </div>
                <p className="text-muted-foreground italic">"{testimonial.content}"</p>
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