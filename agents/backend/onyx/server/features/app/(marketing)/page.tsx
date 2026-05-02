import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowRight, GraduationCap, BookOpen, Users, Trophy, Calendar, Sparkles } from "lucide-react";
import { AnimatedHero } from "@/components/sections/animated-hero";
import { AnimatedFeatures } from "@/components/sections/animated-features";
import { AnimatedEvents } from "@/components/sections/animated-events";
import { AnimatedCTA } from "@/components/sections/animated-cta";
import { MarketingParallax } from "@/components/sections/marketing-parallax"
import { MarketingHeroSection } from "@/components/sections/marketing-hero-section"
import { MarketingFeatureCards } from "@/components/sections/marketing-feature-cards"

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Aquí irían los logos */}
      {/* Hero tipo b debajo de los logos */}
      <MarketingHeroSection />
      {/* Cards tipo b */}
      <MarketingFeatureCards />
      <MarketingParallax />
      {/* Hero Section */}
      <AnimatedHero />

      {/* Features Section */}
      <AnimatedFeatures />

      {/* Upcoming Events Section */}
      <AnimatedEvents />

      {/* CTA Section */}
      <AnimatedCTA />
    </div>
  );
}
