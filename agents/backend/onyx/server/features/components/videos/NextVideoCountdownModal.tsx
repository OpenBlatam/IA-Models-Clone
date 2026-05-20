"use client";
import React, { useEffect, useState, Fragment } from "react";
import { Dialog, Transition } from "@headlessui/react";
import { motion } from "framer-motion";
import { CheckCircle, X } from "lucide-react";

interface NextVideoCountdownModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNext: () => void;
  nextTitle: string;
  nextThumbnail: string;
  nextDuration?: string;
  completedTitle: string;
  countdownSeconds?: number;
}

const NextVideoCountdownModal: React.FC<NextVideoCountdownModalProps> = ({
  isOpen,
  onClose,
  onNext,
  nextTitle,
  nextThumbnail,
  nextDuration,
  completedTitle,
  countdownSeconds = 5,
}) => {
  const [seconds, setSeconds] = useState(countdownSeconds);

  useEffect(() => {
    if (!isOpen) return;
    setSeconds(countdownSeconds);
    if (countdownSeconds <= 0) return;
    const interval = setInterval(() => {
      setSeconds((s) => {
        if (s <= 1) {
          clearInterval(interval);
          onNext();
          return 0;
        }
        return s - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [isOpen, countdownSeconds, onNext]);

  return (
    <Transition.Root show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300" enterFrom="opacity-0" enterTo="opacity-100"
          leave="ease-in duration-200" leaveFrom="opacity-100" leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm transition-opacity" />
        </Transition.Child>
        <div className="fixed inset-0 z-50 flex items-center justify-center px-2">
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300" enterFrom="opacity-0 scale-95" enterTo="opacity-100 scale-100"
            leave="ease-in duration-200" leaveFrom="opacity-100 scale-100" leaveTo="opacity-0 scale-95"
          >
            <Dialog.Panel className="relative bg-[#18181b] rounded-2xl shadow-2xl w-full max-w-lg mx-auto flex flex-col items-center p-0">
              {/* Botón de cerrar */}
              <button
                onClick={onClose}
                className="absolute top-4 left-4 z-10 p-2 rounded-full bg-black/60 hover:bg-black/80 text-white transition-colors"
                aria-label="Cerrar"
              >
                <X className="w-5 h-5" />
              </button>
              {/* Encabezado */}
              <div className="w-full px-8 pt-10 pb-2">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="text-green-500 w-6 h-6" />
                  <span className="text-lg font-semibold text-white">Ya viste:</span>
                </div>
                <div className="ml-8 text-white text-base font-medium truncate">
                  {completedTitle}
                </div>
                <div className="ml-8 mt-2 text-sm text-white/80">
                  Próxima clase en <span className="font-bold text-white">{seconds}</span>
                </div>
              </div>
              {/* Vista previa de la siguiente clase */}
              <div className="w-full flex justify-center items-center px-8 pt-2 pb-4">
                <div className="rounded-2xl overflow-hidden shadow-lg w-full max-w-xs aspect-video bg-black flex items-center justify-center">
                  {nextThumbnail ? (
                    <img
                      src={nextThumbnail}
                      alt={nextTitle}
                      className="object-cover w-full h-full"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-white/60 text-sm">Sin vista previa</div>
                  )}
                </div>
              </div>
              {/* Botones */}
              <div className="w-full flex gap-4 px-8 pb-8">
                <button
                  onClick={onClose}
                  className="flex-1 py-3 rounded-lg bg-zinc-700 text-white text-lg font-semibold hover:bg-zinc-600 transition-colors"
                >
                  Detener
                </button>
                <button
                  onClick={onNext}
                  className="flex-1 py-3 rounded-lg bg-white text-zinc-900 text-lg font-semibold hover:bg-zinc-100 transition-colors shadow"
                >
                  Ver siguiente clase
                </button>
              </div>
            </Dialog.Panel>
          </Transition.Child>
        </div>
      </Dialog>
    </Transition.Root>
  );
};

export default NextVideoCountdownModal;    