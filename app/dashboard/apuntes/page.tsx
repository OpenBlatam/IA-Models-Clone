"use client";

import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";

const OnlyOfficeEditor = dynamic(() => import("@/components/onlyoffice-editor"), {
  ssr: false,
});

export default function NotesPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [documentUrl, setDocumentUrl] = useState<string>("");
  const [documentTitle, setDocumentTitle] = useState<string>("Nuevo Documento");

  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/login");
    }
  }, [status, router]);

  if (status === "loading") {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-[#00F5A0]"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-white">Apuntes</h1>
        <div className="flex gap-4">
          <button
            onClick={() => setDocumentTitle("Nuevo Documento")}
            className="px-4 py-2 bg-[#00F5A0] text-black rounded-lg hover:bg-[#00D48A] transition-colors"
          >
            Nuevo Documento
          </button>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <OnlyOfficeEditor
          documentUrl={documentUrl}
          documentTitle={documentTitle}
          documentType="text"
          onSave={(url) => setDocumentUrl(url)}
        />
      </div>
    </div>
  );
} 