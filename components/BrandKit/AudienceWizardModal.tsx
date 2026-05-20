import React, { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const brandKits = [
  { id: 1, name: "Blatam", icon: "💎" },
  { id: 2, name: "Academia AI", icon: "🤖" },
];

const languages = [
  { code: "es", label: "Spanish" },
  { code: "en", label: "English" },
];

export function AudienceWizardModal({
  open,
  onClose,
  onNext,
  onRegenerate,
  suggestion,
  loading,
  brandKitId = 1,
  language = "es",
}: {
  open: boolean;
  onClose: () => void;
  onNext: (audience: string, brandKitId: number, language: string) => void;
  onRegenerate: () => void;
  suggestion: string;
  loading?: boolean;
  brandKitId?: number;
  language?: string;
}) {
  const [audience, setAudience] = useState("");
  const [manual, setManual] = useState(false);
  const [selectedBrandKit, setSelectedBrandKit] = useState(brandKitId);
  const [selectedLang, setSelectedLang] = useState(language);

  React.useEffect(() => {
    if (!manual && suggestion) setAudience(suggestion);
  }, [suggestion, manual]);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-lg">
        <div className="flex items-center gap-2 mb-2 text-sm">
          <span className="font-semibold text-primary">2</span>
          <span>Set Audience</span>
        </div>
        <DialogHeader>
          <DialogTitle>¿A quién va dirigido el contenido?</DialogTitle>
        </DialogHeader>
        <div className="mb-2 text-muted-foreground text-sm">
          We used your Brand Kit to suggest this audience
        </div>
        <div className="flex items-center gap-2 mb-2">
          <Button variant="ghost" size="sm" onClick={onRegenerate} disabled={loading}>
            Regenerar
          </Button>
          <Button variant="outline" size="sm" onClick={() => setManual(true)}>
            I'll write my own
          </Button>
        </div>
        <Input
          value={audience}
          onChange={e => setAudience(e.target.value)}
          className="mb-2"
          placeholder="Ej: emprendedores, profesores, estudiantes..."
          disabled={loading}
        />
        <div className="flex gap-2 items-center mb-2">
          <span className="text-xs">Brand Kit</span>
          <select
            className="border rounded px-2 py-1 text-sm"
            value={selectedBrandKit}
            onChange={e => setSelectedBrandKit(Number(e.target.value))}
          >
            {brandKits.map(bk => (
              <option key={bk.id} value={bk.id}>{bk.icon} {bk.name}</option>
            ))}
          </select>
          <span className="text-xs ml-4">Generate in</span>
          <select
            className="border rounded px-2 py-1 text-sm"
            value={selectedLang}
            onChange={e => setSelectedLang(e.target.value)}
          >
            {languages.map(l => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button
            disabled={!audience || loading}
            onClick={() => onNext(audience, selectedBrandKit, selectedLang)}
          >
            Next
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
} 