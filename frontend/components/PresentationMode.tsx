'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiMaximize2, FiMinimize2, FiChevronLeft, FiChevronRight, FiX } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useHotkeys } from 'react-hotkeys-hook';

interface PresentationModeProps {
  content: string;
  title: string;
  onClose: () => void;
}

export default function PresentationMode({ content, title, onClose }: PresentationModeProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentSlide, setCurrentSlide] = useState(0);

  // Split content into slides (by ## headings)
  const slides = content.split(/\n##\s+/).map((slide, index) => {
    if (index === 0) return slide;
    return '## ' + slide;
  });

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const toggleFullscreen = async () => {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  };

  useHotkeys('escape', () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    }
    onClose();
  });

  useHotkeys('arrowleft', () => {
    setCurrentSlide((prev) => Math.max(0, prev - 1));
  });

  useHotkeys('arrowright', () => {
    setCurrentSlide((prev) => Math.min(slides.length - 1, prev + 1));
  });

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black z-50 flex flex-col"
      >
        {/* Header */}
        <div className="absolute top-0 left-0 right-0 p-4 bg-black/50 backdrop-blur-sm z-10 flex items-center justify-between">
          <h2 className="text-white font-semibold">{title}</h2>
          <div className="flex items-center gap-2">
            <span className="text-white text-sm">
              {currentSlide + 1} / {slides.length}
            </span>
            <button
              onClick={toggleFullscreen}
              className="text-white hover:bg-white/20 p-2 rounded"
            >
              {isFullscreen ? <FiMinimize2 size={20} /> : <FiMaximize2 size={20} />}
            </button>
            <button onClick={onClose} className="text-white hover:bg-white/20 p-2 rounded">
              <FiX size={20} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex items-center justify-center p-20 overflow-hidden">
          <motion.div
            key={currentSlide}
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            className="max-w-4xl w-full"
          >
            <div className="prose prose-invert prose-lg max-w-none text-white">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {slides[currentSlide] || content}
              </ReactMarkdown>
            </div>
          </motion.div>
        </div>

        {/* Navigation */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-black/50 backdrop-blur-sm z-10 flex items-center justify-between">
          <button
            onClick={() => setCurrentSlide((prev) => Math.max(0, prev - 1))}
            disabled={currentSlide === 0}
            className="text-white hover:bg-white/20 p-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <FiChevronLeft size={24} />
          </button>
          <div className="flex gap-2">
            {slides.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentSlide(index)}
                className={`w-2 h-2 rounded-full transition-colors ${
                  index === currentSlide ? 'bg-white' : 'bg-white/30'
                }`}
              />
            ))}
          </div>
          <button
            onClick={() => setCurrentSlide((prev) => Math.min(slides.length - 1, prev + 1))}
            disabled={currentSlide === slides.length - 1}
            className="text-white hover:bg-white/20 p-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <FiChevronRight size={24} />
          </button>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


