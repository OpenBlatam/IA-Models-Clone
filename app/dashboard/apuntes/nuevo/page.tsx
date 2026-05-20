"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft, Save } from "lucide-react";

export default function NewNotePage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch("/api/notes", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title,
          content,
        }),
      });

      if (!response.ok) throw new Error("Error al crear el apunte");

      router.push("/dashboard/apuntes");
    } catch (error) {
    } finally {
      setLoading(false);
    }
  };

  if (status === "loading") {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-[#00F5A0]"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <Button
          variant="ghost"
          onClick={() => router.back()}
          className="mb-8 text-[#00F5A0] hover:text-[#00F5A0] hover:bg-[#00F5A0]/10"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Volver
        </Button>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-gray-200 mb-2">
              Título
            </label>
            <Input
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Título del apunte"
              className="bg-[#1A1A1A] border-[#333333] text-white placeholder:text-gray-500 focus:border-[#00F5A0]"
              required
            />
          </div>

          <div>
            <label htmlFor="content" className="block text-sm font-medium text-gray-200 mb-2">
              Contenido
            </label>
            <Textarea
              id="content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Escribe tu apunte aquí..."
              className="min-h-[400px] bg-[#1A1A1A] border-[#333333] text-white placeholder:text-gray-500 focus:border-[#00F5A0]"
              required
            />
          </div>

          <div className="flex justify-end">
            <Button
              type="submit"
              disabled={loading}
              className="bg-gradient-to-r from-[#00F5A0] to-[#00D9F5] text-black hover:opacity-90"
            >
              <Save className="mr-2 h-4 w-4" />
              {loading ? "Guardando..." : "Guardar Apunte"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}  