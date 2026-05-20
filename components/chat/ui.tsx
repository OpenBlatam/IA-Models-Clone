import { useState, useRef, useEffect, useCallback } from "react";
import {
  Box,
  Flex,
  Input,
  Textarea,
  Button,
  Avatar,
  Spinner,
  chakra,
  VisuallyHidden,
  Text,
} from "@chakra-ui/react";
import { ArrowUpIcon } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { AnimatePresence, motion } from "framer-motion";
import { useDropzone } from "react-dropzone";

const OPENROUTER_API_KEY = process.env.NEXT_PUBLIC_OPENROUTER_API_KEY || "";

type Message = {
  id: string;
  type: "user" | "bot";
  text?: string;
  image?: string;
};

const MotionBox = motion(chakra.div);

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: crypto.randomUUID(),
      type: "bot",
      text: "Hola 👋, escribe `imagen un dragón volando` o hazme una pregunta.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  // const toast = useToast(); // Removed due to compatibility issues
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 150;
    if (isNearBottom) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [messages]);

  const addMessage = useCallback(
    (msg: Omit<Message, "id">) => {
      setMessages((prev) => [...prev, { ...msg, id: crypto.randomUUID() }]);
    },
    [setMessages]
  );

  // Handle sending text or image generation
  const handleSend = async () => {
    if (!input.trim()) return;
    addMessage({ type: "user", text: input });
    setLoading(true);
    setIsTyping(true);
    const prompt = input.trim();
    setInput("");

    try {
      if (prompt.toLowerCase().startsWith("imagen ")) {
        const imagePrompt = prompt.slice(7);

        const response = await fetch("https://openrouter.ai/api/v1/generate", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${OPENROUTER_API_KEY}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: "openai/dall-e-3",
            prompt: imagePrompt,
            size: "1024x1024",
          }),
        });

        const data = await response.json();
        if (data?.url) {
          addMessage({ type: "bot", image: data.url });
        } else {
          console.error("No se pudo generar la imagen.");
          addMessage({ type: "bot", text: "❌ No se pudo generar la imagen." });
        }
      } else {
        // Chat completion call (api/chat)
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }),
        });
        const data = await response.json();

        if (data?.text) {
          addMessage({ type: "bot", text: data.text });
        } else {
          console.error("No se pudo obtener respuesta del modelo.");
          addMessage({ type: "bot", text: "❌ No se pudo obtener respuesta." });
        }
      }
    } catch (error) {
      console.error("Ocurrió un error inesperado.");
      addMessage({ type: "bot", text: "❌ Error en la petición." });
    } finally {
      setLoading(false);
      setIsTyping(false);
    }
  };

  // Drag and drop usando react-dropzone
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      acceptedFiles.forEach(async (file) => {
        if (!file.type.startsWith("image/")) return;
        setLoading(true);
        setIsTyping(true);
        addMessage({ type: "user", image: URL.createObjectURL(file) });

        // Aquí se puede enviar el archivo a backend o API de generación para reconocimiento o almacenamiento
        // Simulación de respuesta rápida de bot
        setTimeout(() => {
          addMessage({ type: "bot", text: "Imagen recibida correctamente 📷" });
          setLoading(false);
          setIsTyping(false);
        }, 1200);
      });
    },
    [addMessage]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { "image/*": [] } });

  return (
    <Flex
      direction="column"
      height="100vh"
      bgGradient="linear(to-br, gray.900, gray.800)"
      color="white"
      p={4}
      maxW="3xl"
      mx="auto"
      borderRadius="xl"
      boxShadow="2xl"
    >
      <chakra.header
        fontWeight="bold"
        fontSize="xl"
        textAlign="center"
        p={3}
        borderBottom="1px solid"
        borderColor="gray.700"
        userSelect="none"
      >
        ChatGPT - UI Mejorada con Chakra UI
      </chakra.header>

      <Box
        flex="1"
        overflowY="auto"
        mt={4}
        mb={4}
        px={2}
        {...getRootProps()}
        borderWidth="2px"
        borderColor={isDragActive ? "blue.400" : "gray.700"}
        borderStyle="dashed"
        borderRadius="md"
        position="relative"
        _focusWithin={{ boxShadow: "outline" }}
        aria-label="Área de mensajes y arrastrar imagenes"
      >
        <input {...getInputProps()} aria-label="Subir imagenes" />

        {isDragActive && (
          <Box
            position="absolute"
            inset={0}
            bg="blue.500"
            opacity={0.25}
            borderRadius="md"
            zIndex={10}
            pointerEvents="none"
            display="flex"
            alignItems="center"
            justifyContent="center"
            fontWeight="bold"
            fontSize="lg"
            color="white"
          >
            Suelta para subir imágenes
          </Box>
        )}

        <AnimatePresence initial={false}>
          {messages.map(({ id, type, text, image }) => {
            const isUser = type === "user";
            return (
              <MotionBox
                key={id}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 15 }}
                mb={4}
                display="flex"
                justifyContent={isUser ? "flex-end" : "flex-start"}
              >
                <Box
                  maxW="70%"
                  bg={isUser ? "blue.600" : "gray.700"}
                  color="white"
                  p={4}
                  borderRadius="2xl"
                  position="relative"
                  _after={{
                    content: '""',
                    position: "absolute",
                    bottom: 0,
                    width: 8,
                    height: 8,
                    backgroundColor: isUser ? "blue.600" : "gray.700",
                    transform: isUser ? "translateX(50%) rotate(45deg)" : "translateX(-50%) rotate(45deg)",
                    left: isUser ? "100%" : "0",
                    borderBottomRightRadius: 2,
                  }}
                >
                  {image ? (
                    <img
                      src={image}
                      alt="Imagen generada"
                      style={{
                        borderRadius: "8px",
                        maxWidth: "100%",
                        maxHeight: "400px",
                        objectFit: "contain"
                      }}
                      loading="lazy"
                    />
                  ) : (
                    <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap">
                      <ReactMarkdown
                        components={{
                          p: (props) => <Text mb={2} {...props} />,
                          a: (props) => (
                            <a
                              {...props}
                              style={{
                                color: "#63b3ed",
                                textDecoration: "underline"
                              }}
                              target="_blank"
                              rel="noopener noreferrer"
                            />
                          ),
                        }}
                      >
                        {text || ""}
                      </ReactMarkdown>
                    </div>
                  )}
                </Box>
              </MotionBox>
            );
          })}
        </AnimatePresence>

        {isTyping && (
          <Flex align="center" color="gray.400" mb={2} pl={2}>
            <Spinner size="sm" mr={2} />
            Escribiendo...
          </Flex>
        )}

        <div ref={scrollRef} />
      </Box>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!loading) handleSend();
        }}
      >
        <Flex gap={2} align="end">
          <Textarea
            resize="vertical"
            rows={1}
            maxHeight="150px"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Escribe 'imagen un dragón' o hazme una pregunta..."
            disabled={loading}
            aria-label="Entrada de texto"
            flex="1"
          />
          <Button
            colorScheme="blue"
            type="submit"
            loading={loading}
            loadingText="Enviando"
            aria-label="Enviar mensaje"
          >
            Enviar <ArrowUpIcon style={{ marginLeft: '8px', width: '16px', height: '16px' }} />
          </Button>
        </Flex>
      </form>
    </Flex>
  );
}
