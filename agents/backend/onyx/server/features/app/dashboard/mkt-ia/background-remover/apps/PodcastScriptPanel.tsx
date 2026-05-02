import React from "react";

export default function PodcastScriptPanel() {
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl">🎤</span>
        <span className="font-bold text-lg">Podcast Script</span>
      </div>
      <p className="text-gray-600 mb-4 text-sm">Craft well-structured podcast scripts that keep listeners engaged from start to finish</p>
      <div className="bg-gray-50 border rounded-lg p-4 flex-1">
        <div className="font-semibold mb-2">App settings</div>
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
          <div className="text-xs mb-1">Include up to 5 source materials to guide content and accuracy</div>
          <div className="flex gap-2 flex-wrap">
            <button className="border rounded px-2 py-1 text-xs">Upload file</button>
            <button className="border rounded px-2 py-1 text-xs">Add text</button>
            <button className="border rounded px-2 py-1 text-xs">Add URL</button>
            <button className="border rounded px-2 py-1 text-xs">Attach Knowledge</button>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-2">
          <input type="checkbox" checked readOnly className="accent-blue-600" />
          <span className="text-xs">Enable project settings context</span>
        </div>
        <div className="mt-2">
          <input className="w-full border rounded px-2 py-1 text-xs" placeholder="Your project context will go here" disabled />
        </div>
        <button className="w-full bg-blue-600 text-white rounded mt-4 py-2 font-semibold">Generate now</button>
      </div>
    </div>
  );
} 