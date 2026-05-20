"use client";

import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft, Save } from "lucide-react";

interface Note {
  id: string;
  title: string;
  content: string;
  createdAt: string;
  updatedAt: string;
}

export default function EditNotePage({ params }: { params: Promise<{ id: string }> }) {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [note, setNote] = useState<Note | null>(null);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [noteId, setNoteId] = useState<string>("");

  useEffect(() => {
    params.then((resolvedParams) => {
      setNoteId(resolvedParams.id);
    });
  }, [params]);

  useEffect(() => {
    const fetchNote = async () => {
      if (!noteId) return;
      
      try {
        const response = await fetch(`/api/notes/${noteId}`);
        if (!response.ok) throw new Error("Error al cargar el apunte");
        const data = await response.json();
        setNote(data);
        setTitle(data.title);
        setContent(data.content);
      } catch (error) {
        router.push("/dashboard/apuntes");
      } finally {
        setLoading(false);
      }
    };

    if (status === "authenticated" && noteId) {
      fetchNote();
    }
  }, [noteId, status, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);

    try {
      const response = await fetch(`/api/notes/${noteId}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title,
          content,
        }),
      });

      if (!response.ok) throw new Error("Error al actualizar el apunte");

      router.push("/dashboard/apuntes");
    } catch (error) {
    } finally {
      setSaving(false);
    }
  };

  if (status === "loading" || loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-[#00F5A0]"></div>
      </div>
    );
  }

  if (!note) {
    return null;
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
              disabled={saving}
              className="bg-gradient-to-r from-[#00F5A0] to-[#00D9F5] text-black hover:opacity-90"
            >
              <Save className="mr-2 h-4 w-4" />
              {saving ? "Guardando..." : "Guardar Cambios"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}      