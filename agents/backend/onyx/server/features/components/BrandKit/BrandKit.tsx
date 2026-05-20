import React from "react";

export type BrandKitData = {
  colors: string[];
  fonts: { title: string; subtitle: string; body: string };
};

export function BrandKit({ data }: { data: BrandKitData }) {
  return (
    <div className="space-y-6">
      {/* Colors */}
      <div className="bg-white rounded-xl p-6 shadow">
        <div className="flex justify-between items-center mb-4">
          <span className="font-semibold text-lg">Colors</span>
          <button className="text-primary font-medium text-sm hover:underline">+ New Color</button>
        </div>
        <div className="flex gap-6">
          {data.colors.map((color) => (
            <div key={color} className="flex flex-col items-center">
              <div
                className="w-14 h-14 rounded-full border"
                style={{ background: color }}
              />
              <span className="mt-2 text-xs font-mono">{color}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Fonts */}
      <div className="bg-white rounded-xl p-6 shadow">
        <div className="flex justify-between items-center mb-4">
          <span className="font-semibold text-lg">Fonts</span>
          <button className="text-primary font-medium text-sm hover:underline">Manage Fonts</button>
        </div>
        <div className="divide-y">
          <div className="flex items-center justify-between py-2">
            <span className="font-bold">Title</span>
            <span>{data.fonts.title}</span>
          </div>
          <div className="flex items-center justify-between py-2">
            <span className="text-lg text-gray-500">Subtitle</span>
            <span>{data.fonts.subtitle}</span>
          </div>
          <div className="flex items-center justify-between py-2">
            <span className="text-gray-500">Body</span>
            <span>{data.fonts.body}</span>
          </div>
        </div>
      </div>
    </div>
  );
} 