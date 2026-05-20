import React, { useState, useRef, useEffect } from "react";
import { Editor, EditorContent, useEditor } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Underline from '@tiptap/extension-underline';
import Link from '@tiptap/extension-link';
import Image from '@tiptap/extension-image';
import BulletList from '@tiptap/extension-bullet-list';
import Ai from '@tiptap-pro/extension-ai';
import AppSettingsPanel from "../components/AppSettingsPanel";
import KeyMessagePanel from "../components/KeyMessagePanel";
import PreferredLengthPanel from "../components/PreferredLengthPanel";
import EditorToolbar from "../components/EditorToolbar";
import { KeyMessageService } from '../frontend';
import { MessageType, MessageTone, KeyMessageRequest } from '../types/key-messages';
import { variables } from '../variables';
import { TIPTAP_AI_CONFIG } from '../config/ai';

const FACEBOOK_POST_MAX_LENGTH = 700;
const EMOJIS = ['😀', '🚀', '🔥', '💡', '🎉', '👍', '✨', '😎', '📢', '🤩'];
const LUXURY_PINK = '#E0115F';
const TYPING_DURATION = 4000; // ms

const TONES = [
  MessageTone.PROFESSIONAL,
  MessageTone.CASUAL,
  MessageTone.FRIENDLY,
  MessageTone.AUTHORITATIVE,
  MessageTone.CONVERSATIONAL,
];

const STYLE_INSTRUCTIONS = [
  "Start with an emoji.",
  "Make it rhyme.",
  "Use a question.",
  "Add a call to action.",
  "Use a fun fact.",
  "Make it sound urgent.",
  "Use a quote.",
  "Make it playful.",
  "Use a metaphor.",
  "Make it concise.",
  "Make it poetic.",
  "Use a surprising fact.",
  "Make it sound like a story.",
];

function getRandomItem<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function insertRandomEmojis(text: string, count = 2) {
  let result = text;
  for (let i = 0; i < count; i++) {
    const pos = Math.floor(Math.random() * (result.length + 1));
    const emoji = EMOJIS[Math.floor(Math.random() * EMOJIS.length)];
    result = result.slice(0, pos) + ' ' + emoji + ' ' + result.slice(pos);
  }
  return result;
}

interface EditorInstance {
  id: string;
  editor: Editor;
  typing: boolean;
  progress: number; // 0 to 1
}

interface AiState {
  isLoading: boolean;
  errorMessage: string | null;
  response: string | null;
}

export default function FacebookPostPanel() {
  const [keyMessage, setKeyMessage] = useState("");
  const [additionalContent, setAdditionalContent] = useState("");
  const [preferredLength, setPreferredLength] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editors, setEditors] = useState<EditorInstance[]>([]);
  const [streamMode, setStreamMode] = useState(true);
  const [aiState, setAiState] = useState<AiState>({
    isLoading: false,
    errorMessage: null,
    response: null,
  });
  const keyMessageService = new KeyMessageService();
  const typingTimeoutRef = useRef<{ [id: string]: number }>({});
  const postsContainerRef = useRef<HTMLDivElement>(null);
  const [variantCount, setVariantCount] = useState(1);
  const [aiToken, setAiToken] = useState<string | null>(null);

  // Scroll automático al agregar un nuevo post
  useEffect(() => {
    if (postsContainerRef.current) {
      postsContainerRef.current.scrollLeft = postsContainerRef.current.scrollWidth;
    }
  }, [editors.length]);

  useEffect(() => {
    const fetchToken = async () => {
      try {
        const response = await fetch('/api/tiptap-ai');
        if (!response.ok) {
          throw new Error('Failed to fetch token');
        }
        const data = await response.json();
        setAiToken(data.token);
      } catch (error) {
        console.error('Error fetching Tiptap AI token:', error);
        setAiState({
          isLoading: false,
          errorMessage: 'Failed to initialize AI features',
          response: null,
        });
      }
    };

    fetchToken();
  }, []);

  const createEditor = () => {
    if (!aiToken) {
      throw new Error('AI token not available');
    }

    const editor = new Editor({
      extensions: [
        StarterKit.configure({
          bulletList: false,
        }),
        Underline,
        Link.configure({
          openOnClick: false,
        }),
        Image,
        BulletList,
        Ai.configure({
          appId: TIPTAP_AI_CONFIG.appId,
          baseUrl: TIPTAP_AI_CONFIG.baseUrl,
          token: aiToken,
          autocompletion: true,
          onLoading: () => {
            setAiState({
              isLoading: true,
              errorMessage: null,
              response: null,
            });
          },
          onChunk: ({ response }) => {
            setAiState({
              isLoading: true,
              errorMessage: null,
              response: response || null,
            });
          },
          onSuccess: ({ response }) => {
            setAiState({
              isLoading: false,
              errorMessage: null,
              response: response || null,
            });
          },
          onError: error => {
            console.log(error);
            setAiState({
              isLoading: false,
              errorMessage: error.message || 'An error occurred',
              response: null,
            });
          },
        }),
      ],
      content: '',
      editable: true,
    });

    return editor;
  };

  const handleGenerate = async () => {
    try {
      setIsGenerating(true);
      setError(null);

      const randomTone = getRandomItem(TONES);
      const randomStyle = getRandomItem(STYLE_INSTRUCTIONS);
      const randomSeed = Math.random().toString(36).substring(2, 8);
      const variantInstruction = `\n\nThis is variant number ${variantCount} (seed: ${randomSeed}). ${randomStyle} Make it different from previous ones.`;

      const request: KeyMessageRequest = {
        text: (keyMessage || "Create an engaging Facebook post") + variantInstruction,
        type: MessageType.MARKETING,
        tone: randomTone,
        targetAudience: "Facebook users",
        context: additionalContent,
        keywords: [],
        maxLength: FACEBOOK_POST_MAX_LENGTH
      };

      const response = await keyMessageService.generateMessage(request);

      if (response.success && response.data) {
        let responseText = response.data.response.trim().slice(0, FACEBOOK_POST_MAX_LENGTH);
        responseText = insertRandomEmojis(responseText, 2);

        const newEditor = createEditor();
        const id = Math.random().toString(36).substr(2, 9);
        setEditors((eds) => [...eds, { id, editor: newEditor, typing: true, progress: 0 }]);
        setVariantCount((v) => v + 1);

        // Typing animation
        let currentIndex = 0;
        const totalLength = responseText.length;
        const startTime = performance.now();
        function typeNextChar(ts: number) {
          const elapsed = ts - startTime;
          const progress = Math.min(elapsed / TYPING_DURATION, 1);
          const charsToShow = Math.floor(progress * totalLength);
          if (charsToShow > currentIndex) {
            const nextChunk = responseText.slice(currentIndex, charsToShow);
            newEditor.commands.insertContent(nextChunk);
            currentIndex = charsToShow;
          }
          setEditors((eds) => eds.map(e => e.id === id ? { ...e, progress, typing: progress < 1 } : e));
          if (progress < 1) {
            typingTimeoutRef.current[id] = requestAnimationFrame(typeNextChar);
          }
        }
        typingTimeoutRef.current[id] = requestAnimationFrame(typeNextChar);
      } else {
        throw new Error(response.error || 'Failed to generate message');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate message');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCloseEditor = (id: string) => {
    setEditors((eds) => eds.filter(e => e.id !== id));
    if (typingTimeoutRef.current[id]) {
      cancelAnimationFrame(typingTimeoutRef.current[id]);
      delete typingTimeoutRef.current[id];
    }
  };

  return (
    <>
      <div className="flex flex-col h-full">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-2xl">👍</span>
          <span className="font-bold text-lg">Facebook Post</span>
          <span className="ml-2 bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span>
        </div>
        <p className="text-gray-600 mb-4 text-sm">Foster engagement and amplify reach using engaging Facebook updates</p>
        <AppSettingsPanel />
        <KeyMessagePanel
          title="Key message"
          description="What key message do you want to convey?"
          value={keyMessage}
          onChange={setKeyMessage}
          onGenerate={() => setKeyMessage("Generated key message example.")}
          onNext={() => alert("Next pressed")}
          onDelete={() => setKeyMessage("")}
          maxLength={FACEBOOK_POST_MAX_LENGTH}
        />
        <KeyMessagePanel
          title="Additional content"
          description="Are there any hashtags, links or call to actions you want to include?"
          value={additionalContent}
          onChange={setAdditionalContent}
          onGenerate={() => setAdditionalContent("Generated additional content example.")}
          onNext={() => alert("Next pressed for additional content")}
          onDelete={() => setAdditionalContent("")}
          maxLength={FACEBOOK_POST_MAX_LENGTH}
        />
        <PreferredLengthPanel
          title="Preferred length"
          description="How long do you want the caption to be?"
          value={preferredLength}
          onChange={setPreferredLength}
          onNext={() => alert("Next pressed for preferred length")}
          onDelete={() => setPreferredLength("")}
          options={["Short", "Medium", "Long"]}
        />
        <div className="flex flex-col gap-4 mt-4">
          <div className="control-group">
            <div className="flex-row">
              <div className="switch-group">
                <label>
                  <input
                    type="radio"
                    name="option-switch"
                    onChange={() => setStreamMode(false)}
                    checked={streamMode === false}
                  />
                  Non-streaming
                </label>
                <label>
                  <input
                    type="radio"
                    name="option-switch"
                    onChange={() => setStreamMode(true)}
                    checked={streamMode === true}
                  />
                  Streaming
                </label>
              </div>
              <div className="hint">💡 Select text to improve</div>
            </div>
          </div>

          <button
            className="w-full py-4 text-lg font-semibold rounded-xl bg-blue-600/80 text-white shadow-lg backdrop-blur-md hover:bg-blue-700/90 transition border-none focus:outline-none"
            onClick={handleGenerate}
            disabled={isGenerating}
          >
            {isGenerating ? 'Generating...' : 'Generar nuevo post'}
          </button>
          {error && <div className="text-red-500 text-sm mt-2">{error}</div>}
          {aiState.errorMessage && <div className="hint error">{aiState.errorMessage}</div>}
          {aiState.isLoading && (
            <div className="hint purple-spinner">AI is generating</div>
          )}
        </div>
      </div>
      {/* Floating posts container, scrollable and responsive */}
      <div
        ref={postsContainerRef}
        className="fixed top-0 bottom-0 right-0 z-50 flex flex-row gap-4 items-start pointer-events-none overflow-x-auto overflow-y-hidden px-4 py-8"
        style={{ maxWidth: '100vw', minWidth: '0' }}
      >
        {editors.map(({ id, editor, typing, progress }) => (
          <div
            key={id}
            className="min-w-[250px] max-w-[320px] sm:min-w-[320px] sm:max-w-[370px] bg-white rounded-2xl shadow-2xl border border-gray-200 flex flex-col pointer-events-auto relative overflow-hidden mx-1"
            style={{ flex: '0 0 auto' }}
          >
            {/* Luxury pink bar with 'blatam' and progress */}
            <div className="w-full relative rounded-t-2xl overflow-hidden" style={{height: '38px', background: LUXURY_PINK}}>
              <div
                className="absolute left-0 top-0 h-full transition-all duration-75"
                style={{
                  width: `${Math.round(progress * 100)}%`,
                  background: 'linear-gradient(90deg, #E0115F 0%, #FFB6D5 100%)',
                  zIndex: 1,
                  transition: 'width 0.08s linear',
                }}
              />
              <div className="absolute inset-0 flex items-center justify-center z-10">
                <span className="text-white font-bold tracking-widest text-base uppercase drop-shadow">blatam</span>
              </div>
            </div>
            {/* Facebook-like header */}
            <div className="flex items-center gap-3 px-4 pt-3 pb-2">
              <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center overflow-hidden">
                <img src="https://placekitten.com/100/100" alt="Profile" className="w-full h-full object-cover" />
              </div>
              <div>
                <div className="font-semibold text-gray-900 leading-tight">Your Page Name</div>
                <div className="text-gray-500 text-xs flex items-center gap-1">
                  <span>Just now</span>
                  <span>·</span>
                  <span>🌍</span>
                  <span>·</span>
                  <svg width="14" height="14" fill="none" viewBox="0 0 24 24" className="text-gray-400">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z" fill="currentColor"/>
                  </svg>
                </div>
              </div>
            </div>
            <button
              className="absolute top-2 right-2 text-gray-400 hover:text-red-500 z-20"
              onClick={() => handleCloseEditor(id)}
              aria-label="Cerrar post"
              type="button"
            >
              <svg width="20" height="20" fill="none" viewBox="0 0 24 24">
                <path d="M6 18L18 6M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
            <EditorToolbar editor={editor} />
            <div className="px-4 pb-2 pt-2">
              <EditorContent editor={editor} className="prose prose-sm max-w-none text-gray-900" />
              {typing && (
                <span className="inline-block w-2 h-4 bg-pink-400 animate-pulse ml-0.5 rounded"></span>
              )}
              <div className="mt-2 text-right text-xs text-gray-400">
                {editor.getText().length} / {FACEBOOK_POST_MAX_LENGTH} caracteres
              </div>
            </div>
            {/* AI Controls */}
            <div className="px-4 py-2 border-t border-gray-100 bg-gray-50">
              <div className="button-group">
                <button
                  onClick={() => editor.chain().focus().aiRephrase({ stream: streamMode }).run()}
                  className="text-sm text-gray-600 hover:text-blue-600"
                >
                  Rephrase
                </button>
                <button
                  onClick={() => editor.chain().focus().aiShorten({ stream: streamMode }).run()}
                  className="text-sm text-gray-600 hover:text-blue-600"
                >
                  Shorten
                </button>
                <button
                  onClick={() => editor.chain().focus().aiExtend({ stream: streamMode }).run()}
                  className="text-sm text-gray-600 hover:text-blue-600"
                >
                  Extend
                </button>
                <button
                  onClick={() => editor.chain().focus().aiEmojify({ stream: streamMode }).run()}
                  className="text-sm text-gray-600 hover:text-blue-600"
                >
                  Add Emojis
                </button>
              </div>
            </div>
            {/* Facebook-like actions */}
            <div className="flex items-center justify-between px-4 py-2 border-t border-gray-100 bg-gray-50">
              <button className="flex items-center gap-1 text-gray-500 hover:text-blue-600 font-semibold text-sm px-2 py-1 rounded transition">
                <svg width="18" height="18" fill="none" viewBox="0 0 24 24">
                  <path d="M14 10h2.5l1-4H14v-2c0-1.03 0-2 2-2h1.5V2.14c-.326-.043-1.557-.14-2.857-.14C11.928 2 10 3.657 10 6.7v2.8H7v4h3V22h4v-8z" fill="currentColor"/>
                </svg>
                Like
              </button>
              <button className="flex items-center gap-1 text-gray-500 hover:text-blue-600 font-semibold text-sm px-2 py-1 rounded transition">
                <svg width="18" height="18" fill="none" viewBox="0 0 24 24">
                  <path d="M21.99 4c0-1.1-.89-2-1.99-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h14l4 4-.01-18z" fill="currentColor"/>
                </svg>
                Comment
              </button>
              <button className="flex items-center gap-1 text-gray-500 hover:text-blue-600 font-semibold text-sm px-2 py-1 rounded transition">
                <svg width="18" height="18" fill="none" viewBox="0 0 24 24">
                  <path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92 1.61 0 2.92-1.31 2.92-2.92s-1.31-2.92-2.92-2.92z" fill="currentColor"/>
                </svg>
                Share
              </button>
            </div>
          </div>
        ))}
      </div>
    </>
  );
} 