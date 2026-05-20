import React from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Wand2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function UrlInputBox({
  value,
  onChange,
  loading,
  onGenerate,
  onTryExample,
  onUseText,
  label = "Pega aquí la URL de cualquier web",
  placeholder = "https://ejemplo.com",
  buttonText = "Generar",
  extraClass = "",
}: {
  value: string;
  onChange: (v: string) => void;
  loading: boolean;
  onGenerate: () => void;
  onTryExample: () => void;
  onUseText: () => void;
  label?: string;
  placeholder?: string;
  buttonText?: string;
  extraClass?: string;
}) {
  return (
    <>
      <div className={`flex items-center bg-[#18181b] rounded-xl shadow border-2 border-[#7b61ff] px-4 py-2 ${extraClass}`}>
        <span className="text-base font-medium text-gray-200 mr-4 whitespace-nowrap">{label}</span>
        <Input
          className="flex-1 bg-transparent outline-none text-lg px-2 border-0 shadow-none focus:ring-0 text-white placeholder:text-gray-400"
          placeholder={placeholder}
          value={value}
          onChange={e => onChange(e.target.value)}
          disabled={loading}
          type="url"
          style={{ background: "transparent" }}
        />
        <Button
          onClick={onGenerate}
          disabled={loading || !value}
          className="ml-4 px-6 py-2 rounded-lg bg-[#7b61ff] hover:bg-[#8f6fff] text-white font-semibold text-lg shadow transition disabled:opacity-60 flex items-center gap-2 border-0 relative"
        >
          <span className="relative flex items-center">
            <Wand2 className="w-6 h-6 mr-2" />
            <AnimatePresence>
              {loading && (
                <motion.span
                  key="sparkle"
                  initial={{ opacity: 0, scale: 0.5, y: 0 }}
                  animate={{ opacity: 1, scale: 1.2, y: -8 }}
                  exit={{ opacity: 0, scale: 0.5, y: 0 }}
                  transition={{ duration: 0.4 }}
                  className="absolute -top-2 left-3 text-yellow-300"
                >
                  ✨
                </motion.span>
              )}
            </AnimatePresence>
          </span>
          {loading ? (
            <span className="ml-2 animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full" />
          ) : (
            <span>{buttonText}</span>
          )}
        </Button>
      </div>
      <div className="flex items-center justify-between mt-2">
        <button onClick={onTryExample} className="flex items-center gap-2 text-base text-gray-300 hover:underline">
          <svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path d="M5 12h14M12 5l7 7-7 7" stroke="#a084ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
          Probar un ejemplo
        </button>
        <span className="text-gray-500">o</span>
        <button onClick={onUseText} className="flex items-center gap-2 text-base text-gray-200 border border-gray-700 rounded-lg px-3 py-1 hover:bg-[#23232a]">
          <svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path d="M17 7a5 5 0 00-7.07 0l-4 4a5 5 0 007.07 7.07l1.41-1.41M7 17a5 5 0 007.07 0l4-4a5 5 0 00-7.07-7.07L9.59 6.59" stroke="#a084ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
          Usar texto en vez de URL
        </button>
      </div>
    </>
  );
}    