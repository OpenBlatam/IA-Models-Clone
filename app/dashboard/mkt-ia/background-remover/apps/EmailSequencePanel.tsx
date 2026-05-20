import React from "react";
import AppSettingsPanel from "../components/AppSettingsPanel";

export default function EmailSequencePanel() {
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl">✉️</span>
        <span className="font-bold text-lg">Email Sequence</span>
        <span className="ml-2 bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span>
      </div>
      <p className="text-gray-600 mb-4 text-sm">Guide customer journeys and boost conversions with a tailored email sequence</p>
      <AppSettingsPanel />
      <div className="bg-white/60 backdrop-blur-md border border-blue-200 rounded-xl p-5 flex flex-col gap-4 shadow-md mt-2">
        <div className="flex items-center justify-between mb-2">
          <span className="font-semibold text-sm">Add project assets for context</span>
          <span className="text-gray-400 text-xs">1 / 5</span>
        </div>
        <div className="flex flex-col gap-2 mb-2">
          <div className="bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-sm font-medium text-blue-900 flex items-center gap-2">
            <span className="bg-blue-200 text-blue-700 rounded-full px-2 py-0.5 text-xs font-bold">AI Marketing Platform Ads</span>
          </div>
        </div>
        <select className="w-full border rounded px-2 py-1 text-sm text-gray-400" disabled>
          <option>Goal</option>
        </select>
        <select className="w-full border rounded px-2 py-1 text-sm text-gray-400" disabled>
          <option>Outline</option>
        </select>
        <div className="rounded-xl border border-blue-200 bg-blue-50/60 p-3 flex flex-col gap-2">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-blue-500 text-lg">💡</span>
            <span className="font-medium text-blue-900 text-sm">Add more context</span>
          </div>
          <div className="flex flex-wrap gap-2">
            <span className="bg-white border border-blue-200 text-blue-700 text-xs px-3 py-1 rounded-full font-medium cursor-pointer">+ Keywords</span>
            <span className="bg-white border border-blue-200 text-blue-700 text-xs px-3 py-1 rounded-full font-medium cursor-pointer">+ Key Points</span>
            <span className="bg-white border border-blue-200 text-blue-700 text-xs px-3 py-1 rounded-full font-medium cursor-pointer">+ Custom information</span>
          </div>
        </div>
        <div className="rounded-xl border border-blue-200 bg-blue-50/60 p-3 flex items-center gap-2">
          <span className="bg-blue-100 text-blue-600 rounded p-1">
            <svg width="18" height="18" fill="none" viewBox="0 0 24 24">
              <path d="M5 12h14M12 5v14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </span>
          <span className="font-medium text-blue-900 text-sm">Create an image <span className="bg-blue-200 text-blue-700 text-xs font-bold px-2 py-0.5 rounded ml-1 align-middle">PRO</span></span>
          <button className="ml-auto bg-blue-600 text-white rounded px-3 py-1 text-xs font-semibold hover:bg-blue-700 transition">+</button>
        </div>
      </div>
    </div>
  );
} 