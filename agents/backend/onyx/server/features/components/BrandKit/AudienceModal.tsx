import React, { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { RefreshCw, Image as ImageIcon } from "lucide-react";
import { RefinePostDetailsModal } from "./RefinePostDetailsModal";

const brandKits = [
  { id: 1, name: "Blatam", icon: "💎" },
  { id: 2, name: "Academia AI", icon: "🤖" },
];

const languages = [
  { code: "es", label: "Spanish" },
  { code: "en", label: "English" },
];

export function AudienceModal({
  open,
  onClose,
  onRegenerate,
  onNext,
  suggestion,
  loading,
  brandKitId = 1,
  language = "es",
}: {
  open: boolean;
  onClose: () => void;
  onRegenerate: () => void;
  onNext: (audience: string, brandKitId: number, language: string) => void;
  suggestion: string;
  loading?: boolean;
  brandKitId?: number;
  language?: string;
}) {
  const [audience, setAudience] = useState(suggestion || "");
  const [manual, setManual] = useState(false);
  const [selectedBrandKit, setSelectedBrandKit] = useState(brandKitId);
  const [selectedLang, setSelectedLang] = useState(language);
  const [showRefineModal, setShowRefineModal] = useState(false);

  React.useEffect(() => {
    if (!manual && suggestion) setAudience(suggestion);
  }, [suggestion, manual]);

  const handleRefinePostDetails = () => {
    onClose();
    setShowRefineModal(true);
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onClose}>
        <DialogContent className="max-w-xl p-8 rounded-2xl bg-white shadow-xl space-y-6">
          <DialogTitle className="sr-only">Set purpose and goals</DialogTitle>
          <div>
            <div className="mb-1 text-xl font-semibold text-gray-900">What is your post about? <span className="text-red-500">*</span></div>
            <div className="mb-3 text-gray-500 text-base">
              We used your Brand Kit to suggest this prompt
            </div>
            <div className="flex items-start gap-2 w-full">
              <textarea
                value={audience}
                onChange={e => setAudience(e.target.value)}
                className="w-full h-24 min-h-[96px] text-lg px-5 py-4 border-2 border-[#0099FF] focus:border-[#0099FF] rounded-xl bg-white shadow resize-none leading-relaxed transition"
                placeholder="Ej: emprendedores, profesores, estudiantes..."
                disabled={loading}
                style={{ fontFamily: "inherit" }}
              />
              <Button
                variant="ghost"
                onClick={onRegenerate}
                disabled={loading}
                className="mt-2 ml-2 flex items-center justify-center h-10 w-10 rounded-full hover:bg-[#E6F0FA] transition"
                style={{ color: "#0099FF" }}
                size="icon"
              >
                <RefreshCw className="w-6 h-6" />
              </Button>
            </div>
            <div className="flex items-center mt-2">
              <span className="text-base text-gray-400 mr-2">Brand Kit</span>
              <select
                className="bg-[#F6F7F9] border-none text-base font-medium focus:outline-none rounded-xl px-3 py-2"
                value={selectedBrandKit}
                onChange={e => setSelectedBrandKit(Number(e.target.value))}
              >
                {brandKits.map(bk => (
                  <option key={bk.id} value={bk.id}>{bk.icon} {bk.name}</option>
                ))}
              </select>
            </div>
          </div>
          <Button
            variant="outline"
            className="mt-4 border border-gray-200 text-gray-900 font-semibold py-3 px-6 rounded-xl bg-white hover:bg-gray-50 w-fit"
            onClick={() => setManual(true)}
          >
            I&apos;ll write my own
          </Button>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 bg-[#F6F7F9] rounded-xl px-4 py-3">
              <span className="text-base text-gray-500">Generate in</span>
              <select
                className="bg-transparent border-none text-base font-medium focus:outline-none"
                value={selectedLang}
                onChange={e => setSelectedLang(e.target.value)}
              >
                {languages.map(l => (
                  <option key={l.code} value={l.code}>{l.label}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex justify-end items-center mt-6">
            <Button
              className="h-12 px-8 text-lg rounded-xl bg-black text-white hover:bg-gray-900 font-semibold shadow"
              disabled={!audience || loading}
              onClick={handleRefinePostDetails}
            >
              Refine Post Details
            </Button>
          </div>
        </DialogContent>
      </Dialog>
      <RefinePostDetailsModal
        open={showRefineModal}
        onClose={() => setShowRefineModal(false)}
        onBack={() => {
          setShowRefineModal(false);
          onClose();
        }}
        onGenerate={(templateId, caption) => {
          setShowRefineModal(false);
        }}
        initialCaption=""
        loading={false}
      />
    </>
  );
}
