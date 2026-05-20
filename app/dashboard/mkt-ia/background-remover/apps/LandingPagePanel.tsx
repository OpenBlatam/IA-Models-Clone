import React from "react";

export default function LandingPagePanel() {
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl">🌀</span>
        <span className="font-bold text-lg">Landing Page</span>
        <span className="ml-2 bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span>
      </div>
      <p className="text-gray-600 mb-4 text-sm">Transform site traffic into valuable leads through engaging landing pages</p>
      {/* Aquí puedes agregar controles específicos para esta app */}
    </div>
  );
} 