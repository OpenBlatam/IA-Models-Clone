import React from "react";

export default function HeroSection({
  title,
  subtitle,
  backgroundClass = "bg-black",
  textClass = "text-white",
  containerClass = "",
  multicolorBackground = false,
  linearGradientBackground = false,
  children,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  backgroundClass?: string;
  textClass?: string;
  containerClass?: string;
  multicolorBackground?: boolean;
  linearGradientBackground?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <section className={`relative w-full ${backgroundClass} px-4 pt-12 pb-8 flex flex-col items-center overflow-hidden ${containerClass}`}>
      {multicolorBackground && (
        <div
          className="absolute inset-0 z-0 pointer-events-none"
          aria-hidden="true"
          style={{
            background:
              "radial-gradient(circle at 20% 40%, #7b61ff 30%, transparent 60%), " +
              "radial-gradient(circle at 80% 60%, #22c55e 30%, transparent 60%), " +
              "radial-gradient(circle at 60% 20%, #fbbf24 30%, transparent 60%), " +
              "radial-gradient(circle at 40% 80%, #ec4899 30%, transparent 60%), " +
              "radial-gradient(circle at 70% 70%, #f59e42 30%, transparent 60%)",
            filter: "blur(40px)",
            opacity: 0.85,
          }}
        />
      )}
      {linearGradientBackground && (
        <div
          className="absolute inset-0 z-0 pointer-events-none"
          aria-hidden="true"
          style={{
            background:
              "linear-gradient(90deg, #FAF3F3 0%, #FF6FA5 25%, #FFD6E0 50%, #B2F0F0 75%, #F39AC1 100%)",
            width: "100%",
            height: "100%",
            opacity: 0.85,
          }}
        />
      )}
      <div className="relative z-10 w-full flex flex-col items-center">
        <h1 className={`text-4xl md:text-5xl font-bold text-center mb-2 ${textClass}`}>{title}</h1>
        {subtitle && (
          <p className={`text-lg text-center mb-8 max-w-2xl ${textClass} opacity-80`}>{subtitle}</p>
        )}
        {children && <div className="w-full max-w-2xl flex flex-col gap-2">{children}</div>}
      </div>
    </section>
  );
} 