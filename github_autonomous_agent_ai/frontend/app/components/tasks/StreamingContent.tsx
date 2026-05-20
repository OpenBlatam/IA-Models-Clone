'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { cn } from '../../utils/cn';

interface StreamingContentProps {
  content: string;
  isProcessing: boolean;
  className?: string;
}

export function StreamingContent({ content, isProcessing, className }: StreamingContentProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [displayedContent, setDisplayedContent] = useState('');
  const [isScrolling, setIsScrolling] = useState(false);

  // Auto-scroll al final cuando hay nuevo contenido (más suave y en tiempo real)
  useEffect(() => {
    if (scrollRef.current) {
      const element = scrollRef.current;
      const isNearBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 150;
      
      // Si está cerca del final o está procesando, hacer scroll automático
      if (isNearBottom || isProcessing) {
        // Usar requestAnimationFrame para scroll más suave
        requestAnimationFrame(() => {
          if (scrollRef.current) {
            scrollRef.current.scrollTo({
              top: scrollRef.current.scrollHeight,
              behavior: isProcessing ? 'smooth' : 'auto',
            });
          }
        });
      }
    }
  }, [content, isProcessing]);

  // Actualizar contenido inmediatamente para tiempo real
  useEffect(() => {
    setDisplayedContent(content);
  }, [content]);

  // Detectar si el usuario está scrolleando manualmente
  const handleScroll = () => {
    if (scrollRef.current) {
      const element = scrollRef.current;
      const isAtBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 50;
      setIsScrolling(!isAtBottom);
    }
  };

  // Detectar código en el contenido (más eficiente)
  const hasCodeBlocks = displayedContent.includes('```');
  const hasJson = displayedContent.trim().startsWith('{') || displayedContent.trim().startsWith('[');
  const isMarkdown = hasCodeBlocks || displayedContent.includes('##') || displayedContent.includes('**');

  return (
    <div className={cn("relative", className)}>
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100"
        style={{
          scrollbarWidth: 'thin',
        }}
      >
        <div className="p-4 space-y-2">
          {isMarkdown || hasJson ? (
            // Si hay código, usar markdown con syntax highlighting
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ node, inline, className, children, ...props }: any) {
                  const match = /language-(\w+)/.exec(className || '');
                  const language = match ? match[1] : hasJson ? 'json' : 'text';
                  
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={language}
                      PreTag="div"
                      className="rounded-lg !my-2"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className={cn("px-1.5 py-0.5 bg-gray-200 rounded text-sm font-mono", className)} {...props}>
                      {children}
                    </code>
                  );
                },
                p: ({ children }: any) => (
                  <p className="mb-2 text-sm text-gray-800 leading-relaxed">{children}</p>
                ),
                h1: ({ children }: any) => (
                  <h1 className="text-lg font-bold mb-2 text-gray-900">{children}</h1>
                ),
                h2: ({ children }: any) => (
                  <h2 className="text-base font-semibold mb-2 text-gray-900">{children}</h2>
                ),
                ul: ({ children }: any) => (
                  <ul className="list-disc list-inside mb-2 space-y-1 text-sm text-gray-800">{children}</ul>
                ),
                ol: ({ children }: any) => (
                  <ol className="list-decimal list-inside mb-2 space-y-1 text-sm text-gray-800">{children}</ol>
                ),
                li: ({ children }: any) => (
                  <li className="text-sm text-gray-800">{children}</li>
                ),
              }}
            >
              {displayedContent}
            </ReactMarkdown>
          ) : (
            // Si no hay código, mostrar como texto formateado con renderizado optimizado
            <div className="text-sm text-gray-800 font-mono whitespace-pre-wrap leading-relaxed">
              {displayedContent.split('\n').map((line, idx, arr) => {
                // Solo animar las últimas 10 líneas para mejor rendimiento
                const shouldAnimate = isProcessing && idx >= arr.length - 10;
                return (
                  <motion.div
                    key={`${idx}-${line.substring(0, 20)}`}
                    initial={shouldAnimate ? { opacity: 0, x: -5 } : false}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    {line || '\u00A0'}
                  </motion.div>
                );
              })}
            </div>
          )}

          {/* Indicador de escritura animado */}
          <AnimatePresence>
            {isProcessing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="inline-flex items-center gap-1 text-gray-500"
              >
                <motion.span
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  className="inline-block w-2 h-2 bg-green-500 rounded-full"
                />
                <motion.span
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                  className="inline-block w-2 h-2 bg-green-500 rounded-full"
                />
                <motion.span
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                  className="inline-block w-2 h-2 bg-green-500 rounded-full"
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Indicador de scroll */}
      {isScrolling && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="absolute bottom-4 right-4 bg-blue-500 text-white px-3 py-1.5 rounded-lg text-xs font-medium shadow-lg flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
          Nuevo contenido disponible
        </motion.div>
      )}
    </div>
  );
}

