import React, { useState } from "react";

interface PreferredLengthPanelProps {
  title: string;
  description: string;
  value: string;
  onChange: (v: string) => void;
  onNext: () => void;
  onDelete: () => void;
  options: string[];
  isOpen?: boolean;
  onToggle?: () => void;
}

export default function PreferredLengthPanel({
  title,
  description,
  value,
  onChange,
  onNext,
  onDelete,
  options,
  isOpen: isOpenProp,
  onToggle
}: PreferredLengthPanelProps) {
  const [isOpen, setIsOpen] = useState(isOpenProp ?? true);
  const open = isOpenProp !== undefined ? isOpenProp : isOpen;
  const handleToggle = () => {
    if (onToggle) onToggle();
    else setIsOpen((v) => !v);
  };
  return (
    <div className="bg-white/60 backdrop-blur-md border border-gray-200 rounded-2xl p-6 mb-4 shadow-md">
      <button
        className="flex items-center gap-2 w-full text-left focus:outline-none"
        onClick={handleToggle}
        aria-expanded={open}
      >
        <span className={`inline-block w-4 h-4 rounded-full border-2 border-dashed ${value ? 'border-green-400' : 'border-red-400'}`}></span>
        <span className="font-semibold text-lg text-gray-900">{title}</span>
        <svg className={`ml-auto w-5 h-5 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
      </button>
      {open && (
        <div className="mt-4">
          <div className="text-gray-500 mb-2 text-base">{description}</div>
          <select
            className="w-full border-2 border-blue-400 rounded-xl p-3 text-base focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white/80 transition appearance-none"
            value={value}
            onChange={e => onChange(e.target.value)}
          >
            <option value="" disabled>Select an option</option>
            {options.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
          <div className="flex items-center mt-4 gap-2">
            <button
              className="bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl p-3 flex items-center justify-center"
              type="button"
              onClick={onDelete}
              aria-label="Delete"
            >
              <svg width="22" height="22" fill="none" viewBox="0 0 24 24"><rect x="5" y="6" width="14" height="12" rx="2" stroke="#222" strokeWidth="2"/><path d="M10 11v4M14 11v4" stroke="#222" strokeWidth="2" strokeLinecap="round"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" stroke="#222" strokeWidth="2"/></svg>
            </button>
            <button
              className="ml-auto bg-blue-600/80 hover:bg-blue-700/90 text-white font-semibold rounded-xl px-8 py-3 text-lg shadow-lg backdrop-blur-md transition border-none focus:outline-none"
              type="button"
              onClick={onNext}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
} 