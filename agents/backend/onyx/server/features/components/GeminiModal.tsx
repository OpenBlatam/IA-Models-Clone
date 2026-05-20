import React from "react";

interface GeminiModalProps {
  open: boolean;
  onClose: () => void;
  message: string;
  color?: string;
}

const GeminiModal: React.FC<GeminiModalProps> = ({ open, onClose, message, color = "purple" }) => {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm animate-fade-in">
      <div className="bg-white rounded-3xl shadow-2xl px-10 py-8 flex flex-col items-center gap-6 min-w-[340px] max-w-[90vw] animate-scale-in">
        <span className={`relative flex h-14 w-14 mb-2`}>
          <span className={`animate-spin inline-block w-full h-full rounded-full border-4 border-${color}-300 border-t-transparent`}/>
        </span>
        <div className="text-center">
          <div className="text-lg font-semibold text-gray-900 mb-1">{message}</div>
          <div className="text-sm text-gray-400">Waiting for response</div>
        </div>
        <button
          onClick={onClose}
          className="mt-2 px-4 py-2 rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium transition"
        >
          Cancelar
        </button>
      </div>
    </div>
  );
};

export default GeminiModal; 