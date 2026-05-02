"use client";
import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Placeholder from "@tiptap/extension-placeholder";
import React from "react";
import { Button } from "@/components/ui/button";
import { Bold, Italic, Underline, List, Link, Image, Undo2, Redo2, MoreHorizontal } from "lucide-react";

export default function WordEditor() {
  const editor = useEditor({
    extensions: [
      StarterKit,
      Placeholder.configure({
        placeholder: "Start writing your document...",
      }),
    ],
    content: "",
  });

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-lg mx-auto w-full max-w-3xl min-h-[400px] p-0 mt-8">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-6 py-3 border-b border-gray-100 bg-gray-50 rounded-t-xl">
        <Button variant="outline" size="sm" className="flex items-center gap-2"><Bold className="w-4 h-4" /> </Button>
        <Button variant="outline" size="sm" className="flex items-center gap-2"><Italic className="w-4 h-4" /> </Button>
        <Button variant="outline" size="sm" className="flex items-center gap-2"><Underline className="w-4 h-4" /> </Button>
        <span className="mx-2 h-5 w-px bg-gray-200" />
        <Button variant="outline" size="sm" className="flex items-center gap-2"><List className="w-4 h-4" /> </Button>
        <Button variant="outline" size="sm" className="flex items-center gap-2"><Link className="w-4 h-4" /> </Button>
        <Button variant="outline" size="sm" className="flex items-center gap-2"><Image className="w-4 h-4" /> </Button>
        <span className="mx-2 h-5 w-px bg-gray-200" />
        <Button variant="outline" size="sm" className="flex items-center gap-2"><Undo2 className="w-4 h-4" /> </Button>
        <Button variant="outline" size="sm" className="flex items-center gap-2"><Redo2 className="w-4 h-4" /> </Button>
        <span className="ml-auto" />
        <Button variant="outline" size="sm" className="flex items-center gap-2"><MoreHorizontal className="w-4 h-4" /></Button>
      </div>
      <div className="p-8">
        <EditorContent editor={editor} />
      </div>
    </div>
  );
} 