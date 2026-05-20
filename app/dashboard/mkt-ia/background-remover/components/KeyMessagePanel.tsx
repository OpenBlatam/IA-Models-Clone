import React, { useState, useEffect, useRef } from "react";
import { KeyMessageService } from '../frontend';
import { MessageType, MessageTone, KeyMessageRequest } from '../types/key-messages';
import ReactMarkdown from 'react-markdown';

interface KeyMessagePanelProps {
  title: string;
  description: string;
  value: string;
  onChange: (v: string) => void;
  onGenerate: () => void;
  onNext: () => void;
  onDelete: () => void;
  maxLength?: number;
  isOpen?: boolean;
  onToggle?: () => void;
  messageType?: MessageType;
  messageTone?: MessageTone;
  targetAudience?: string;
  context?: string;
  keywords?: string[];
}

export default function KeyMessagePanel({
  title,
  description,
  value,
  onChange,
  onGenerate,
  onNext,
  onDelete,
  maxLength = 10000,
  isOpen: isOpenProp,
  onToggle,
  messageType = MessageType.INFORMATIONAL,
  messageTone = MessageTone.PROFESSIONAL,
  targetAudience,
  context,
  keywords = []
}: KeyMessagePanelProps) {
  const [isOpen, setIsOpen] = useState(isOpenProp ?? true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [placeholder, setPlaceholder] = useState("Type your key message...");
  const [localValue, setLocalValue] = useState(value);
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const typingTimeoutRef = useRef<NodeJS.Timeout>();
  const keyMessageService = new KeyMessageService();

  // Update local value when prop value changes
  useEffect(() => {
    console.log('Value prop changed:', value);
    setLocalValue(value);
  }, [value]);

  // Effect for typing animation
  useEffect(() => {
    if (localValue && isGenerating) {
      setIsTyping(true);
      let currentIndex = 0;
      const text = localValue;
      let lastTimestamp = 0;
      const minDelay = 30; // Velocidad mínima más lenta para simular escritura real
      const maxDelay = 100; // Velocidad máxima más lenta
      let currentDelay = minDelay;
      let accumulatedTime = 0;

      const typeNextChar = (timestamp: number) => {
        if (!lastTimestamp) lastTimestamp = timestamp;
        const elapsed = timestamp - lastTimestamp;
        accumulatedTime += elapsed;

        if (accumulatedTime >= currentDelay) {
          if (currentIndex < text.length) {
            // Escribir solo un carácter a la vez
            const nextChar = text[currentIndex];
            setDisplayedText(prev => prev + nextChar);
            currentIndex++;

            // Ajustar la velocidad para que sea más natural
            // Más lento al principio y al final de las palabras
            const isEndOfWord = currentIndex < text.length && text[currentIndex] === ' ';
            const isStartOfWord = currentIndex > 0 && text[currentIndex - 1] === ' ';
            
            if (isEndOfWord || isStartOfWord) {
              currentDelay = maxDelay; // Pausa más larga entre palabras
            } else {
              // Velocidad variable para simular escritura humana
              currentDelay = Math.random() * (maxDelay - minDelay) + minDelay;
            }

            accumulatedTime = 0;
            lastTimestamp = timestamp;
          } else {
            setIsTyping(false);
            return;
          }
        }

        typingTimeoutRef.current = requestAnimationFrame(typeNextChar);
      };

      typingTimeoutRef.current = requestAnimationFrame(typeNextChar);

      return () => {
        if (typingTimeoutRef.current) {
          cancelAnimationFrame(typingTimeoutRef.current);
        }
      };
    } else {
      setDisplayedText(localValue);
    }
  }, [localValue, isGenerating]);

  const open = isOpenProp !== undefined ? isOpenProp : isOpen;
  
  const handleToggle = () => {
    if (onToggle) onToggle();
    else setIsOpen((v) => !v);
  };

  const handleGenerate = async () => {
    try {
      setIsGenerating(true);
      setError(null);
      setPlaceholder("Generating your message...");
      setDisplayedText("");
      
      // If the text area is empty, set a default message based on the message type
      let messageToGenerate = localValue;
      if (!messageToGenerate.trim()) {
        const defaultMessages = {
          [MessageType.MARKETING]: "Create a compelling marketing message for our product",
          [MessageType.EDUCATIONAL]: "Write an educational message about our service",
          [MessageType.PROMOTIONAL]: "Create a promotional message for our offer",
          [MessageType.INFORMATIONAL]: "Write an informative message about our company",
          [MessageType.CALL_TO_ACTION]: "Create a call to action message for our campaign"
        };
        messageToGenerate = defaultMessages[messageType];
        setLocalValue(messageToGenerate);
        onChange(messageToGenerate);
      }
      
      const request: KeyMessageRequest = {
        text: messageToGenerate,
        type: messageType,
        tone: messageTone,
        targetAudience,
        context,
        keywords,
        maxLength
      };

      console.log('Sending request:', request);
      const response = await keyMessageService.generateMessage(request);
      console.log('Received response:', response);

      if (response.success && response.data) {
        console.log('Updating text area with response:', response.data.response);
        const responseText = response.data.response;
        console.log('Response text to be set:', responseText);
        
        // Ensure proper line breaks and formatting
        const formattedText = responseText
          .replace(/\n\s*\n/g, '\n\n') // Normalize multiple line breaks
          .replace(/\*\*(.*?)\*\*/g, '**$1**') // Preserve bold
          .replace(/\*(.*?)\*/g, '*$1*') // Preserve italic
          .trim();
        
        setLocalValue(formattedText);
        onChange(formattedText);
        onGenerate();
        setPlaceholder("Type your key message...");
      } else {
        console.error('Response indicates failure:', response);
        throw new Error(response.error || 'Failed to generate message');
      }
    } catch (err) {
      console.error('Generation error:', err);
      setError(err instanceof Error ? err.message : 'Failed to generate message');
      setPlaceholder("Type your key message...");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    console.log('Textarea onChange:', newValue);
    setLocalValue(newValue);
    setDisplayedText(newValue);
    onChange(newValue);
  };

  return (
    <div className="bg-white/60 backdrop-blur-md border border-gray-200 rounded-2xl p-6 mb-4 shadow-md">
      <button
        className="flex items-center gap-2 w-full text-left focus:outline-none"
        onClick={handleToggle}
        aria-expanded={open}
      >
        <span className={`inline-block w-4 h-4 rounded-full border-2 border-dashed ${localValue ? 'border-green-400' : 'border-red-400'}`}></span>
        <span className="font-semibold text-lg text-gray-900">{title}</span>
        <svg className={`ml-auto w-5 h-5 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
      </button>
      {open && (
        <div className="mt-4">
          <div className="text-gray-500 mb-2 text-base">{description}</div>
          <div className="relative">
            <textarea
              className={`w-full min-h-[200px] border-2 ${isGenerating ? 'border-blue-200' : 'border-blue-400'} rounded-xl p-3 text-base focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white transition whitespace-pre-wrap font-mono`}
              placeholder={placeholder}
              value={localValue}
              maxLength={maxLength}
              onChange={handleTextChange}
              disabled={isGenerating}
              style={{ 
                minHeight: '300px', 
                resize: 'vertical',
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                color: '#1a1a1a',
                lineHeight: '1.6',
                fontSize: '14px',
                padding: '12px',
                borderColor: localValue ? '#3b82f6' : '#e5e7eb',
                boxShadow: localValue ? '0 0 0 2px rgba(59, 130, 246, 0.1)' : 'none',
                whiteSpace: 'pre-wrap',
                wordWrap: 'break-word',
                fontFamily: 'monospace'
              }}
            />
            {isTyping && (
              <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
                <div className="p-3">
                  <ReactMarkdown className="prose prose-sm max-w-none">
                    {displayedText}
                  </ReactMarkdown>
                  <span className="inline-block w-0.5 h-4 bg-blue-500 animate-pulse ml-0.5"></span>
                </div>
              </div>
            )}
          </div>
          <div className="flex items-center justify-between mt-1 text-xs text-gray-400">
            <button
              className={`flex items-center gap-1 ${isGenerating ? 'text-gray-400' : 'text-blue-600'} font-semibold hover:underline disabled:opacity-50`}
              type="button"
              onClick={handleGenerate}
              disabled={isGenerating}
            >
              <svg width="16" height="16" fill="none" viewBox="0 0 24 24"><path d="M13 2v8h8" stroke={isGenerating ? "#9ca3af" : "#2563eb"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M21 3l-9 9-4-4-6 6" stroke={isGenerating ? "#9ca3af" : "#2563eb"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
              {isGenerating ? 'Generating...' : 'Generate response'}
            </button>
            <span>{localValue.length} / {maxLength}</span>
          </div>
          {error && (
            <div className="mt-2 text-sm text-red-500">
              {error}
            </div>
          )}
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