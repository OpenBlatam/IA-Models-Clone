import React from "react";
import Link from "next/link";

export function MarketingHeroSection() {
  return (
    <section className="relative w-full py-24 overflow-hidden" style={{background: "linear-gradient(135deg, #F3F7FF 0%, #E3D6FF 100%)"}}>
      {/* Pastel Background Effects */}
      <div className="absolute inset-0 pointer-events-none z-0" style={{background: "radial-gradient(circle at 60% 40%, #E3D6FF 30%, transparent 70%), radial-gradient(circle at 20% 80%, #FFD6E3 30%, transparent 70%)", filter: "blur(24px)", opacity: 0.7}} />
      {/* Decorative pastel bubbles */}
      <div className="absolute -top-16 left-1/4 w-44 h-44 bg-[#FFD6E3]/60 rounded-full filter blur-2xl z-0 opacity-70" />
      <div className="absolute -bottom-16 right-1/3 w-36 h-36 bg-[#E3D6FF]/60 rounded-full filter blur-2xl z-0 opacity-60" />
      
      <div className="max-w-7xl mx-auto px-4 grid grid-cols-1 md:grid-cols-2 gap-16 items-center relative z-10">
        {/* Left: Title & CTA */}
        <div className="flex flex-col items-center md:items-start text-center md:text-left w-full space-y-10">
          <span className="inline-block px-5 py-2 rounded-full border-2" style={{borderColor: '#E3D6FF', background: '#fff', color: '#A259FF', fontWeight: 700, fontSize: '0.75rem', letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: '0.5rem', boxShadow: '0 2px 8px 0 #E3D6FF22'}}>
            The Everything App for Marketers
          </span>
          <h1 className="text-6xl md:text-8xl font-extrabold leading-tight tracking-tight mb-4" style={{lineHeight: 1.1, color: '#23235F'}}>
            Your AI Marketing <span style={{background: 'linear-gradient(90deg, #FFD6E3 0%, #E3D6FF 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text'}}>
              Superpower
            </span>
          </h1>
          <div className="flex flex-col items-center w-full gap-3 mb-2">
            <button
              className="w-full sm:w-auto px-16 py-6 rounded-3xl font-extrabold text-2xl shadow-2xl transition-transform duration-200 hover:scale-105 border-0"
              style={{
                background: "linear-gradient(90deg, #FFD6E3 0%, #E3D6FF 100%)",
                color: "#23235F",
                boxShadow: "0 8px 40px 0 #E3D6FF33",
                letterSpacing: "0.02em"
              }}
            >
              Get started. It's FREE!
            </button>
            <span className="mt-2 text-[#6B6B8D] text-base font-medium">Free Forever. No Credit Card.</span>
          </div>
          <p className="text-lg md:text-2xl" style={{color: '#23235F', opacity: 0.92, fontWeight: 500}}>
            All your marketing, powered by AI. Collaborate, create, and launch campaigns faster than ever.
          </p>
        </div>
        {/* Right: Feature Preview */}
        <div className="relative flex justify-center md:justify-end mt-12 md:mt-0">
          <div className="glass-effect rounded-3xl p-10 shadow-2xl w-full max-w-md mx-auto border-2" style={{borderColor: '#E3D6FF', background: 'rgba(255,255,255,0.75)', backdropFilter: 'blur(18px)'}}>
            <div className="aspect-video bg-gradient-to-br from-[#FFD6E3]/60 to-[#E3D6FF]/60 rounded-2xl mb-6" />
            <div className="space-y-4">
              <div className="h-4 bg-[#E3D6FF] rounded-full w-3/4" />
              <div className="h-4 bg-[#FFD6E3] rounded-full w-1/2" />
            </div>
          </div>
          {/* Subtle floating pastel bubble */}
          <div className="absolute -top-8 -right-8 w-24 h-24 bg-[#FFD6E3]/60 rounded-full filter blur-2xl z-0 opacity-70" />
        </div>
      </div>
    </section>
  );
} 