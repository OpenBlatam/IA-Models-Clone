"use client";

import { motion } from "framer-motion";
import { Star, Quote, Award } from "lucide-react";
import { useState } from "react";
import { Theme } from "@radix-ui/themes";
import * as HoverCard from "@radix-ui/react-hover-card";
import { NavBar } from "@/components/layout/navbar";
import { SiteFooter } from "@/components/layout/site-footer";

const testimonials = [
  {
    name: "María García",
    role: "Senior Developer",
    company: "Tech Corp",
    image: "/testimonials/1.jpg",
    rating: 5,
    content: "Una experiencia de aprendizaje excepcional. Los mentores son expertos en su campo y el contenido es de primera calidad.",
    badge: "Premium",
    achievements: ["Top 1%", "Mentor", "Speaker"]
  },
  {
    name: "Carlos Rodríguez",
    role: "AI Engineer",
    company: "AI Solutions",
    image: "/testimonials/2.jpg",
    rating: 5,
    content: "El curso de IA superó todas mis expectativas. Proyectos prácticos y mentoría personalizada de primer nivel.",
    badge: "VIP",
    achievements: ["AI Expert", "Researcher", "Innovator"]
  },
  {
    name: "Ana Martínez",
    role: "Full Stack Developer",
    company: "Web Innovators",
    image: "/testimonials/3.jpg",
    rating: 5,
    content: "La mejor inversión en mi carrera profesional. El contenido premium y la comunidad son increíbles.",
    badge: "Exclusivo",
    achievements: ["Tech Lead", "Architect", "Mentor"]
  }
];

export function AnimatedTestimonials() {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <Theme>
      <div className="min-h-screen flex flex-col bg-[#0A0A0A]">
        <NavBar scroll={false} />
        <main className="flex-1 pt-0">
          <section className="py-8 px-4 relative bg-[#0A0A0A]">
            <div className="max-w-6xl mx-auto">
              <div className="text-center mb-8">
                <div className="inline-flex items-center px-4 py-2 rounded-full bg-[#FFD700]/20 text-[#FFD700] mb-4">
                  <span className="text-sm font-medium">Testimonios</span>
                </div>
                <h2 className="text-4xl font-bold mb-4 text-white">
                  Experiencias de Éxito
                </h2>
                <p className="text-lg text-gray-300 max-w-2xl mx-auto">
                  Descubre las historias de nuestros estudiantes
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {testimonials.map((testimonial, index) => (
                  <HoverCard.Root key={index} open={hoveredIndex === index} onOpenChange={(open) => setHoveredIndex(open ? index : null)}>
                    <HoverCard.Trigger asChild>
                      <div className="group cursor-pointer">
                        <div className="p-6 rounded-xl bg-white/5 border border-[#FFD700]/20 hover:border-[#FFD700]/40 transition-all duration-200">
                          <div className="relative">
                            <div className="absolute top-2 right-2">
                              <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                                testimonial.badge === "Premium" ? "bg-green-500/80" : 
                                testimonial.badge === "VIP" ? "bg-purple-500/80" : 
                                "bg-blue-500/80"
                              } text-white`}>
                                {testimonial.badge}
                              </span>
                            </div>
                            
                            <div className="flex flex-col gap-3">
                              <div className="flex items-center gap-3">
                                <div className="w-12 h-12 rounded-full overflow-hidden border border-[#FFD700]">
                                  <img
                                    src={testimonial.image}
                                    alt={testimonial.name}
                                    className="w-full h-full object-cover"
                                  />
                                </div>
                                <div>
                                  <h3 className="text-white font-semibold">{testimonial.name}</h3>
                                  <p className="text-gray-400 text-sm">{testimonial.role}</p>
                                  <p className="text-[#FFD700] text-sm">{testimonial.company}</p>
                                </div>
                              </div>

                              <div className="flex gap-1">
                                {[...Array(testimonial.rating)].map((_, i) => (
                                  <Star key={i} className="h-4 w-4 text-[#FFD700]" fill="#FFD700" />
                                ))}
                              </div>

                              <p className="text-gray-200 text-sm">
                                <Quote className="inline-block h-4 w-4 text-[#FFD700] mr-1" />
                                {testimonial.content}
                              </p>

                              <div className="flex flex-wrap gap-1">
                                {testimonial.achievements.map((achievement, i) => (
                                  <span
                                    key={i}
                                    className="px-2 py-1 rounded-full bg-[#FFD700]/20 text-[#FFD700] text-xs"
                                  >
                                    {achievement}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </HoverCard.Trigger>
                    <HoverCard.Portal>
                      <HoverCard.Content
                        className="w-72 p-4 bg-[#1A1A1A] rounded-lg shadow-lg border border-[#FFD700]/20"
                        sideOffset={5}
                      >
                        <div className="space-y-2">
                          <h4 className="font-semibold text-[#FFD700]">{testimonial.name}</h4>
                          <p className="text-sm text-gray-300">{testimonial.content}</p>
                          <div className="flex flex-wrap gap-1 mt-2">
                            {testimonial.achievements.map((achievement, i) => (
                              <span
                                key={i}
                                className="px-2 py-1 rounded-full bg-[#FFD700]/20 text-[#FFD700] text-xs"
                              >
                                {achievement}
                              </span>
                            ))}
                          </div>
                        </div>
                        <HoverCard.Arrow className="fill-[#1A1A1A]" />
                      </HoverCard.Content>
                    </HoverCard.Portal>
                  </HoverCard.Root>
                ))}
              </div>

              <div className="text-center mt-8">
                <button
                  className="px-6 py-3 rounded-full bg-[#FFD700] text-black font-semibold hover:bg-[#FFD700]/90 transition-colors"
                >
                  Únete a Nuestros Éxitos
                </button>
              </div>
            </div>
          </section>
        </main>
        <SiteFooter />
      </div>
    </Theme>
  );
}    