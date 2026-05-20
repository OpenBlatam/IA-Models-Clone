import React from "react";

export default function BlogPostPanel() {
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl">🖊️</span>
        <span className="font-bold text-lg">Blog Post</span>
        <span className="ml-2 bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span>
      </div>
      <p className="text-gray-600 mb-4 text-sm">Write long-form content that provides value, drives traffic, and enhances SEO</p>
      <div className="bg-white/60 backdrop-blur-md border border-gray-200 rounded-xl p-5 flex flex-col gap-4 shadow-md overflow-y-auto max-h-[70vh]">
        <div className="font-semibold mb-1">App settings</div>
        <div className="mb-2">
          <label className="block text-xs mb-1">Select a Brand Voice</label>
          <select className="w-full border rounded px-2 py-1 text-sm">
            <option>Select a Brand Voice</option>
          </select>
        </div>
        <div className="mb-2">
          <label className="block text-xs mb-1">Select an Audience</label>
          <select className="w-full border rounded px-2 py-1 text-sm">
            <option>Select an Audience</option>
          </select>
        </div>
        <div className="mb-2">
          <label className="block text-xs mb-1">Language</label>
          <select className="w-full border rounded px-2 py-1 text-sm">
            <option>English (American)</option>
          </select>
        </div>
        <div className="mb-2">
          <div className="text-xs mb-1 flex items-center justify-between">
            <span>Add project assets for context</span>
            <span className="text-gray-400 text-xs">0 / 5</span>
          </div>
          <input className="w-full border border-dashed rounded px-2 py-2 text-xs mb-2 bg-white/70" placeholder="Select assets in the project" disabled />
        </div>
        <div className="mb-2">
          <select className="w-full border rounded px-2 py-1 text-sm text-gray-400" disabled>
            <option>Topic</option>
          </select>
        </div>
        <div className="mb-2">
          <select className="w-full border rounded px-2 py-1 text-sm text-gray-400" disabled>
            <option>Outline</option>
          </select>
        </div>
        <div className="mb-2">
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
        </div>
        <div className="mb-2">
          <div className="rounded-xl border border-blue-200 bg-blue-50/60 p-3 flex items-center gap-2">
            <span className="bg-blue-100 text-blue-600 rounded p-1"><svg width="18" height="18" fill="none" viewBox="0 0 24 24"><path d="M5 12h14M12 5v14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg></span>
            <span className="font-medium text-blue-900 text-sm">Create an image <span className="bg-blue-200 text-blue-700 text-xs font-bold px-2 py-0.5 rounded ml-1 align-middle">PRO</span></span>
            <button className="ml-auto bg-blue-600 text-white rounded px-3 py-1 text-xs font-semibold hover:bg-blue-700 transition">+</button>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-2">
          <input type="checkbox" checked readOnly className="accent-blue-600" />
          <span className="text-xs">Enable project settings context</span>
        </div>
        <div className="mt-2">
          <input className="w-full border rounded px-2 py-1 text-xs" placeholder="Your project context will go here" disabled />
        </div>
        <button className="w-full bg-blue-600 text-white rounded mt-4 py-2 font-semibold shadow hover:bg-blue-700 transition">Generate now</button>
      </div>
    </div>
  );
} 