import React from "react";

interface FloatingStatusWidgetProps {
  message: string;
  onCancel?: () => void;
  visible?: boolean;
  color?: string; // Tailwind color, e.g. 'purple'
}

export const FloatingStatusWidget: React.FC<FloatingStatusWidgetProps> = ({
  message,
  onCancel,
  visible = true,
  color = "purple",
}) => {
  if (!visible) return null;

  return (
    <div
      className={`fixed bottom-6 right-6 z-50 flex items-center gap-3 px-5 py-3 rounded-2xl shadow-xl bg-white border border-gray-200 animate-fade-in`}
      style={{ minWidth: 220 }}
    >
      <span className={`relative flex h-7 w-7`}>
        <span
          className={`animate-spin inline-block w-full h-full rounded-full border-4 border-${color}-300 border-t-transparent`}
        />
      </span>
      <div className="flex flex-col">
        <span className="font-medium text-gray-900 text-sm">{message}</span>
        <span className="text-xs text-gray-400">Waiting for response</span>
      </div>
      {onCancel && (
        <button
          onClick={onCancel}
          className="ml-2 p-1 rounded-full hover:bg-gray-100 transition-colors"
          aria-label="Cancelar"
        >
          <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  );
};

export default FloatingStatusWidget; 