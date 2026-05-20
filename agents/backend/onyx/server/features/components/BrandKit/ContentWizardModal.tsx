import React, { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Star, Zap, Heart, Book, Store, Video, Wrench, Megaphone, Mail, MoreHorizontal, Upload, Plus, Clock, Flame, Instagram, Facebook, Linkedin, Twitter } from "lucide-react";

const menu = [
  { label: "Recently Used", icon: <Clock className="w-5 h-5" /> },
  { label: "Popular", icon: <Star className="w-5 h-5 text-yellow-400" /> },
  { label: "Brainstorm Ideas", icon: <Zap className="w-5 h-5 text-blue-400" /> },
  { label: "Start Blank", icon: <Plus className="w-5 h-5 text-gray-400" /> },
  { label: "Social Media", icon: <Heart className="w-5 h-5 text-red-400" /> },
  { label: "Blog/SEO", icon: <Book className="w-5 h-5 text-blue-600" /> },
  { label: "Marketplace", icon: <Store className="w-5 h-5 text-orange-400" /> },
  { label: "Video", icon: <Video className="w-5 h-5 text-indigo-400" /> },
  { label: "Advanced Tools", icon: <Wrench className="w-5 h-5 text-gray-500" /> },
  { label: "Ads", icon: <Megaphone className="w-5 h-5 text-pink-500" /> },
  { label: "Emails", icon: <Mail className="w-5 h-5 text-green-500" /> },
  { label: "Other", icon: <MoreHorizontal className="w-5 h-5 text-gray-400" /> },
  { label: "Upload", icon: <Upload className="w-5 h-5 text-gray-400" /> },
  { label: "New Project", icon: <Flame className="w-5 h-5 text-orange-500" /> },
];

const wizardIcons: Record<string, React.ReactNode> = {
  Instagram: <Instagram className="w-6 h-6 text-pink-500" />,
  Facebook: <Facebook className="w-6 h-6 text-blue-600" />,
  LinkedIn: <Linkedin className="w-6 h-6 text-blue-700" />,
  Twitter: <Twitter className="w-6 h-6 text-black" />,
};

export interface ContentWizard {
  title: string;
  image: string;
  platform: string;
}

// Hero superior estilo HeyGen
function ContentWizardHero({
  value,
  onChange,
  onGenerate,
  onTryExample,
  onUseUrl,
  loading
}: {
  value: string;
  onChange: (v: string) => void;
  onGenerate: () => void;
  onTryExample: () => void;
  onUseUrl: () => void;
  loading?: boolean;
}) {
  return (
    <div className="w-full bg-gradient-to-b from-[#f5f6ff] to-white px-8 pt-8 pb-6 rounded-t-2xl flex flex-col items-center">
      <h1 className="text-4xl md:text-5xl font-bold text-center mb-2">Transform content into engaging videos</h1>
      <p className="text-lg text-muted-foreground text-center mb-8 max-w-2xl">
        Write a topic or paste a URL and let HeyGen transform the content into a ready-to-use video script. Fast, smart, and effortless.
      </p>
      <div className="w-full max-w-2xl flex flex-col gap-2">
        <div className="flex items-center bg-white rounded-xl shadow border-2 border-[#bcbcff] px-4 py-2">
          <span className="text-base font-medium text-gray-700 mr-4 whitespace-nowrap">Create a video about</span>
          <input
            className="flex-1 bg-transparent outline-none text-lg px-2"
            placeholder="your favorite topic..."
            value={value}
            onChange={e => onChange(e.target.value)}
            disabled={loading}
          />
          <button
            onClick={onGenerate}
            disabled={loading || !value}
            className="ml-4 px-6 py-2 rounded-lg bg-[#7b61ff] text-white font-semibold text-lg shadow hover:bg-[#6a4fff] transition disabled:opacity-60 flex items-center gap-2"
          >
            {loading ? (
              <span className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full" />
            ) : (
              <>
                <svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path d="M5 12h14M12 5l7 7-7 7" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                Generate
              </>
            )}
          </button>
        </div>
        <div className="flex items-center justify-between mt-2">
          <button onClick={onTryExample} className="flex items-center gap-2 text-base text-gray-700 hover:underline">
            <svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path d="M5 12h14M12 5l7 7-7 7" stroke="#7b61ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
            Try an example
          </button>
          <span className="text-gray-400">or</span>
          <button onClick={onUseUrl} className="flex items-center gap-2 text-base text-gray-700 border border-gray-200 rounded-lg px-3 py-1 hover:bg-gray-50">
            <svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path d="M17 7a5 5 0 00-7.07 0l-4 4a5 5 0 007.07 7.07l1.41-1.41M7 17a5 5 0 007.07 0l4-4a5 5 0 00-7.07-7.07L9.59 6.59" stroke="#7b61ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
            Use URL instead
          </button>
        </div>
      </div>
    </div>
  );
}

export function ContentWizardModal({
  open,
  onClose,
  wizards,
  onSelect,
}: {
  open: boolean;
  onClose: () => void;
  wizards: ContentWizard[];
  onSelect?: (wizard: ContentWizard) => void;
}) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl p-0 overflow-hidden rounded-2xl">
        <ContentWizardHero
          value={input}
          onChange={setInput}
          onGenerate={() => setLoading(true)}
          onTryExample={() => setInput("How to grow on Instagram")}
          onUseUrl={() => setInput("https://example.com")}
          loading={loading}
        />
        <div className="flex h-[400px] bg-white">
          {/* Sidebar */}
          <aside className="w-56 bg-muted/40 border-r p-4 flex flex-col gap-2">
            {menu.map((item) => (
              <button
                key={item.label}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-lg text-left hover:bg-muted/70 transition font-medium text-gray-700",
                  item.label === "Recently Used" && "bg-white shadow"
                )}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            ))}
          </aside>
          {/* Main */}
          <main className="flex-1 flex flex-col p-8">
            <h2 className="text-2xl font-bold mb-4">Your recently used content wizards</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 mb-6">
              {wizards.map((wiz) => (
                <Card
                  key={wiz.title}
                  className="rounded-xl p-0 overflow-hidden cursor-pointer hover:shadow-lg transition group"
                  onClick={() => onSelect?.(wiz)}
                >
                  <div className="relative w-full h-32 bg-muted flex items-center justify-center">
                    <img src={wiz.image} alt={wiz.title} className="object-cover w-full h-full" />
                    <div className="absolute left-2 top-2 bg-white/80 rounded-full p-1">
                      {wizardIcons[wiz.platform]}
                    </div>
                  </div>
                  <div className="p-4">
                    <div className="font-semibold text-lg text-gray-900 group-hover:text-primary">
                      {wiz.title}
                    </div>
                  </div>
                </Card>
              ))}
            </div>
            <div className="mt-auto">
              <div className="rounded-xl bg-gradient-to-r from-green-400 to-green-300 text-lg font-semibold text-center py-4 text-white shadow">
                Generate more content like your best posts
              </div>
            </div>
          </main>
        </div>
      </DialogContent>
    </Dialog>
  );
} 