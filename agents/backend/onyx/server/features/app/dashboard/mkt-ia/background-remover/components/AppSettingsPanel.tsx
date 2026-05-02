import React, { useState } from "react";

export default function AppSettingsPanel() {
  const [open, setOpen] = useState(false);
  // Estos valores pueden venir de props o estado global
  const brandVoice = "No Brand Voice";
  const audience = "No Audience";
  const language = "English (American)";

  return (
    <div className="border-2 border-blue-500 rounded-xl bg-white/90 p-4 mb-4">
      <button
        className="flex items-center justify-between w-full focus:outline-none"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="font-bold text-lg">App settings</span>
        <svg
          className={`w-5 h-5 text-gray-500 transition-transform ${open ? "rotate-180" : "rotate-0"}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div className="flex items-center gap-2 mt-3 text-gray-700 text-base">
        <span className="text-xl">📢</span>
        <span>{brandVoice}</span>
        <span className="mx-1">•</span>
        <span>{audience}</span>
        <span className="mx-1">•</span>
        <span>{language}</span>
      </div>
      {open && (
        <div className="mt-4 flex flex-col gap-4 animate-fade-in">
          <div className="text-xs text-gray-500 mb-1">Select the voice, audience, and language for your generation</div>
          <select className="w-full border rounded px-2 py-1 text-sm mb-2">
            <option>Select a Brand Voice</option>
          </select>
          <select className="w-full border rounded px-2 py-1 text-sm mb-2">
            <option>Select an Audience</option>
          </select>
          <select className="w-full border rounded px-2 py-1 text-sm">
            <option>English (American)</option>
          </select>
          <div className="text-xs mb-1 mt-2">Include up to 5 source materials to guide content and accuracy</div>
          <div className="flex gap-2 flex-wrap mb-2">
            <button className="border rounded px-3 py-1 text-xs bg-white/80 hover:bg-blue-50 transition">Upload file</button>
            <button className="border rounded px-3 py-1 text-xs bg-white/80 hover:bg-blue-50 transition">Add text</button>
            <button className="border rounded px-3 py-1 text-xs bg-white/80 hover:bg-blue-50 transition">Add URL</button>
            <button className="border rounded px-3 py-1 text-xs bg-white/80 hover:bg-blue-50 transition">Attach Knowledge</button>
          </div>
          <div className="flex items-center gap-2 mt-2">
            <input type="checkbox" checked readOnly className="accent-blue-600" />
            <span className="text-xs">Enable project settings context</span>
          </div>
          <input className="w-full border rounded px-2 py-1 text-xs mt-2" placeholder="Your project context will go here" disabled />
          <button className="w-full border border-blue-600 text-blue-600 rounded mt-2 py-2 font-semibold hover:bg-blue-50 transition">Open project settings</button>
          <button className="w-full bg-blue-600 text-white rounded mt-2 py-2 font-semibold shadow hover:bg-blue-700 transition">Next</button>
        </div>
      )}
    </div>
  );
} 