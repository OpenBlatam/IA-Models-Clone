import React, { useRef, useState } from "react";

export default function BackgroundRemoverPanel() {
  const [image, setImage] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => setImage(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => setImage(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="mb-2">
        <h2 className="text-3xl font-bold">Background Remover</h2>
        <p className="text-gray-500 text-lg mt-1">Effortlessly remove backgrounds from any image</p>
      </div>
      <div className="bg-white border border-gray-200 rounded-2xl p-6 mt-4">
        <div className="font-semibold text-lg mb-4 flex items-center gap-2">
          <span className={`inline-block w-3 h-3 rounded-full border-2 border-dashed mr-2 ${image ? 'border-green-400' : 'border-red-400'}`}></span>
          Image
        </div>
        <div
          className={`flex flex-col items-center justify-center border-2 rounded-xl border-dashed ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-blue-400 bg-blue-50/50'} py-12 transition-all duration-200 cursor-pointer`}
          onClick={() => inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={e => { e.preventDefault(); setDragActive(false); }}
          onDrop={handleDrop}
        >
          {image ? (
            <div className="relative w-full flex flex-col items-center">
              <img src={image} alt="Preview" className="max-h-48 mb-4 rounded shadow" />
              <button
                className="absolute top-2 right-2 bg-white/80 hover:bg-red-100 text-red-500 rounded-full p-1 shadow transition"
                style={{ position: 'absolute', top: 8, right: 8 }}
                onClick={e => { e.stopPropagation(); setImage(null); }}
                aria-label="Remove image"
              >
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M6 6l8 8M14 6l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
              </button>
              <button
                className="text-sm text-blue-600 underline mb-2 mt-2"
                onClick={e => { e.stopPropagation(); setImage(null); }}
              >Remove image</button>
            </div>
          ) : (
            <>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                ref={inputRef}
                onChange={handleFileChange}
              />
              <span className="text-lg font-medium underline mb-2">Browse files</span>
              <div className="flex flex-col items-center mt-2">
                <div className="bg-blue-100 rounded-full p-3 mb-2">
                  <svg width="32" height="32" fill="none" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#60A5FA" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M7 10l5-5 5 5M12 5v12" stroke="#60A5FA" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                </div>
                <span className="text-base text-gray-700 font-medium">IMAGE</span>
              </div>
            </>
          )}
        </div>
        <button
          className={`w-full mt-8 py-4 text-lg font-semibold rounded-xl shadow-lg backdrop-blur-md transition border-none focus:outline-none text-white ${image ? 'bg-green-600/80 hover:bg-green-700/90' : 'bg-blue-600/80 hover:bg-blue-700/90'}`}
        >
          Generate now
        </button>
      </div>
    </div>
  );
} 