"use client";
import React from "react";

export default function ToolLayout({
  title,
  description,
  sidebar,
  children,
}: {
  title: string;
  description?: string;
  sidebar: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col md:flex-row h-screen w-full bg-white">
      {/* Sidebar/tool */}
      <div className="w-full md:w-[420px] bg-white/80 border-r border-gray-100 p-8 flex flex-col">
        <h2 className="text-xl font-bold mb-1">{title}</h2>
        {description && <p className="text-gray-500 mb-6">{description}</p>}
        {sidebar}
      </div>
      {/* Main area */}
      <div className="flex-1 bg-white/60 p-8 flex flex-col">{children}</div>
    </div>
  );
} 