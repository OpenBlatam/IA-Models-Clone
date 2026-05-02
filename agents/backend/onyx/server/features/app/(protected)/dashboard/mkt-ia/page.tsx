"use client";
import { useState, useRef } from "react";
import { useSession } from "next-auth/react";
import { DashboardHeader } from "@/components/dashboard/header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Megaphone, BarChart3, Target, Sparkles, Paperclip, ArrowUp, Package, FileText, MessageSquare, Mail, User, ArrowRight, Pen, ThumbsUp, RefreshCcw, X as LucideX, ArrowLeft } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogClose } from "@/components/ui/dialog";

const featuredApps = [
  {
    id: 1,
    icon: <Sparkles className="w-6 h-6 text-violet-600" />,
    title: "Background Remover",
    badges: [
      { label: "NEW", color: "bg-green-100 text-green-700" },
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
    description: "Effortlessly remove backgrounds from any image",
    overview: "Transform any image into a clean, background-free version. Perfect for product photos or creative visuals, this app makes it easy to isolate your subject and create professional-looking results. Simply upload your image and watch as AI works its magic.",
    why: [
      "Create Polished Visuals: Give your product images or creative designs a sleek and professional touch by removing distractions from the background.",
      "Save Time and Effort: Forget complex photo editing software—this app gets the job done in seconds.",
      "Versatile Applications: Whether you're refreshing your e-commerce store or crafting a social media post, this tool meets a variety of creative needs."
    ],
    categories: ["Image", "Content Marketing", "Performance Marketing", "Social Media Marketing"],
    details: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
      { label: "NEW", color: "bg-green-100 text-green-700" },
    ],
    toolPath: "/dashboard/mkt-ia/background-remover"
  },
  // ...puedes agregar más apps aquí...
];

const popularApps = [
  {
    id: 101,
    icon: <Pen className="w-6 h-6 text-violet-600" />,
    title: "Blog Post",
    badges: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
    description: "Write long-form content that provides value, drives traffic, and...",
    overview: "Write long-form content that provides value, drives traffic, and establishes authority. Perfect for content marketing and SEO.",
    why: [
      "Establish Authority: Position your brand as a thought leader with in-depth articles.",
      "Drive Traffic: Attract organic visitors through valuable, keyword-rich content.",
      "Engage Readers: Keep your audience coming back for more with insightful posts."
    ],
    categories: ["Content Marketing", "Blog", "SEO"],
    details: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
    toolPath: "/dashboard/mkt-ia/blog-post"
  },
  {
    id: 102,
    icon: <Package className="w-6 h-6 text-violet-600" />,
    title: "Product Description",
    badges: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
    description: "Compose detailed descriptions that highlight the benefits and features ...",
    overview: "Compose detailed descriptions that highlight the benefits and features of your products, increasing conversions and customer trust.",
    why: [
      "Boost Sales: Persuasive descriptions drive more purchases.",
      "Highlight Features: Clearly communicate what makes your product unique.",
      "Save Time: Generate descriptions quickly for large catalogs."
    ],
    categories: ["E-commerce", "Product", "Content Marketing"],
    details: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
  },
  {
    id: 103,
    icon: <ThumbsUp className="w-6 h-6 text-violet-600" />,
    title: "Facebook Post",
    badges: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
    description: "Foster engagement and amplify reach using engaging Facebook...",
    overview: "Foster engagement and amplify reach using engaging Facebook posts tailored to your audience.",
    why: [
      "Increase Engagement: Posts designed to spark likes, shares, and comments.",
      "Brand Awareness: Reach new audiences with viral content.",
      "Consistency: Maintain a steady posting schedule with ease."
    ],
    categories: ["Social Media Marketing", "Facebook", "Content"],
    details: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
  },
  {
    id: 104,
    icon: <RefreshCcw className="w-6 h-6 text-violet-600" />,
    title: "Content Rewriter",
    badges: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
    description: "Transform existing content to fit specific goals or audiences...",
    overview: "Transform existing content to fit specific goals or audiences, ensuring your messaging is always on point.",
    why: [
      "Adapt Messaging: Tailor content for different platforms or audiences.",
      "Improve Clarity: Make your message more concise and effective.",
      "Save Resources: Repurpose content instead of starting from scratch."
    ],
    categories: ["Content Marketing", "Editing", "SEO"],
    details: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
  },
  {
    id: 105,
    icon: <Package className="w-6 h-6 text-violet-600" />,
    title: "Landing Page",
    badges: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
    description: "Transform site traffic into valuable leads through engaging landing...",
    overview: "Transform site traffic into valuable leads through engaging landing pages that convert.",
    why: [
      "Increase Conversions: Optimized layouts and copy for maximum impact.",
      "Easy Customization: Quickly adapt pages for campaigns or products.",
      "Professional Design: Stand out with modern, high-converting templates."
    ],
    categories: ["Landing Pages", "Lead Generation", "Marketing"],
    details: [
      { label: "POPULAR", color: "bg-purple-100 text-purple-700" },
    ],
  },
];

export default function MKTIAPage() {
  const { data: session } = useSession();
  const userName = session?.user?.name || "Usuario";
  const [input, setInput] = useState("");
  const carouselRef = useRef<HTMLDivElement>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedApp, setSelectedApp] = useState<any>(null);
  const [showTool, setShowTool] = useState(false);

  const handleCarouselScroll = () => {
    if (carouselRef.current) {
      carouselRef.current.scrollBy({ left: 340, behavior: "smooth" });
    }
  };

  const handleCarouselScrollLeft = () => {
    if (carouselRef.current) {
      carouselRef.current.scrollBy({ left: -340, behavior: "smooth" });
    }
  };

  const openAppModal = (app) => {
    setSelectedApp(app);
    setModalOpen(true);
  };
  const closeAppModal = () => {
    setModalOpen(false);
    setSelectedApp(null);
  };

  return (
    <>
      <DashboardHeader
        heading="MKT IA"
        text="Herramientas de marketing impulsadas por IA"
      />
      {/* Friendly prompt section */}
      <div className="w-full flex flex-col items-center mb-8">
        <div className="w-full max-w-2xl rounded-2xl bg-violet-50 p-8 shadow-sm border border-violet-100">
          <h2 className="text-2xl font-bold mb-4 text-gray-800">
            Hey {userName}, what do you want to create?
          </h2>
          <form className="flex items-center gap-2 mb-6">
            <div className="relative flex-1">
              <input
                type="text"
                className="w-full rounded-full border border-gray-200 bg-white px-4 py-3 pr-20 text-base shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-200"
                placeholder="Ask MKT IA anything..."
                value={input}
                onChange={e => setInput(e.target.value)}
              />
              <button type="button" className="absolute right-12 top-1/2 -translate-y-1/2 text-gray-400 hover:text-violet-500">
                <Paperclip className="w-5 h-5" />
              </button>
              <button type="submit" className="absolute right-3 top-1/2 -translate-y-1/2 bg-violet-500 hover:bg-violet-600 text-white rounded-full p-2 shadow transition-colors">
                <ArrowUp className="w-5 h-5" />
              </button>
            </div>
          </form>
          {/* Row of three cards/buttons */}
          <div className="flex flex-col gap-4 md:flex-row md:space-x-6 md:gap-0 justify-center">
            <button className="flex-1 flex items-center gap-4 rounded-2xl border border-gray-200 bg-white px-6 py-4 shadow-sm hover:shadow-md transition group">
              <span className="flex items-center justify-center rounded-full bg-violet-100 p-3">
                <Package className="w-6 h-6 text-violet-600" />
              </span>
              <span className="text-left">
                <span className="block font-semibold text-gray-900">Start with an App</span>
                <span className="block text-sm text-gray-500">Marketing Apps to supercharge your work</span>
              </span>
            </button>
            <button className="flex-1 flex items-center gap-4 rounded-2xl border border-gray-200 bg-white px-6 py-4 shadow-sm hover:shadow-md transition group">
              <span className="flex items-center justify-center rounded-full bg-violet-100 p-3">
                <FileText className="w-6 h-6 text-violet-600" />
              </span>
              <span className="text-left">
                <span className="block font-semibold text-gray-900">Create a document</span>
                <span className="block text-sm text-gray-500">Start fresh with a blank document</span>
              </span>
            </button>
            <button className="flex-1 flex items-center gap-4 rounded-2xl border border-gray-200 bg-white px-6 py-4 shadow-sm hover:shadow-md transition group">
              <span className="flex items-center justify-center rounded-full bg-violet-100 p-3">
                <MessageSquare className="w-6 h-6 text-violet-600" />
              </span>
              <span className="text-left">
                <span className="block font-semibold text-gray-900">Start a conversation</span>
                <span className="block text-sm text-gray-500">Use Chat to brainstorm or research</span>
              </span>
            </button>
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-8">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Generador de Contenido</CardTitle>
              <Sparkles className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">AI Content</div>
              <p className="text-xs text-muted-foreground">
                Crea contenido optimizado para SEO con IA
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Análisis de Competencia</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">Competitor AI</div>
              <p className="text-xs text-muted-foreground">
                Analiza y compara tu estrategia con la competencia
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Optimización de Campañas</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">Campaign AI</div>
              <p className="text-xs text-muted-foreground">
                Mejora el rendimiento de tus campañas publicitarias
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <Card className="col-span-2">
            <CardHeader>
              <CardTitle>Herramientas de Marketing</CardTitle>
              <CardDescription>
                Accede a todas las herramientas de marketing impulsadas por IA
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="flex items-center space-x-4 rounded-md border p-4">
                  <Megaphone className="h-5 w-5" />
                  <div className="flex-1 space-y-1">
                    <p className="text-sm font-medium leading-none">Generador de Posts</p>
                    <p className="text-sm text-muted-foreground">
                      Crea posts optimizados para redes sociales
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-4 rounded-md border p-4">
                  <Target className="h-5 w-5" />
                  <div className="flex-1 space-y-1">
                    <p className="text-sm font-medium leading-none">Análisis de Audiencia</p>
                    <p className="text-sm text-muted-foreground">
                      Conoce mejor a tu audiencia objetivo
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
      {/* Featured Apps Carousel */}
      <div className="mt-12">
        <h2 className="text-2xl font-bold mb-6 px-2">Featured Apps</h2>
        <div className="relative -mx-2">
          {/* Left Arrow */}
          <button
            type="button"
            aria-label="Scroll left"
            onClick={handleCarouselScrollLeft}
            className="absolute top-1/2 left-2 -translate-y-1/2 z-10 bg-white/60 border border-violet-200 shadow-lg rounded-full p-3 backdrop-blur-md hover:bg-violet-100 hover:scale-110 hover:shadow-violet-200 transition-all duration-200"
            style={{ boxShadow: "0 2px 8px rgba(80,80,120,0.10)" }}
          >
            <ArrowLeft className="w-6 h-6 text-violet-600" />
          </button>
          {/* Right Arrow */}
          <button
            type="button"
            aria-label="Scroll right"
            onClick={handleCarouselScroll}
            className="absolute top-1/2 right-6 -translate-y-1/2 z-10 bg-white/60 border border-violet-200 shadow-lg rounded-full p-3 backdrop-blur-md hover:bg-violet-100 hover:scale-110 hover:shadow-violet-200 transition-all duration-200"
            style={{ boxShadow: "0 2px 8px rgba(80,80,120,0.10)" }}
          >
            <ArrowRight className="w-6 h-6 text-violet-600" />
          </button>
          <div
            className="flex space-x-6 overflow-x-auto pb-4 pl-2 pr-10 snap-x snap-mandatory scrollbar-hide"
            ref={carouselRef}
            id="featured-apps-carousel"
            style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
          >
            {featuredApps.map((app, idx) => (
              <button
                key={app.id || idx}
                className={`min-w-[320px] max-w-xs bg-white/60 backdrop-blur-md border border-violet-100 p-6 flex-shrink-0 snap-start shadow-lg flex flex-col gap-3 hover:shadow-xl transition${idx === featuredApps.length - 1 ? ' mr-[-60px]' : ''} rounded-2xl`}
                onClick={() => openAppModal(app)}
                style={{ WebkitBackdropFilter: 'blur(12px)', backdropFilter: 'blur(12px)' }}
              >
                <span className="flex items-center justify-center rounded-full bg-violet-100 w-10 h-10 mb-2">
                  {app.icon}
                </span>
                <div className="flex gap-2 mb-1">
                  {app.badges.map((badge, i) => (
                    <span key={i} className={`${badge.color} text-xs font-semibold px-2 py-0.5 rounded`}>{badge.label}</span>
                  ))}
                </div>
                <div className="font-bold text-lg text-gray-900">{app.title}</div>
                <div className="text-gray-500 text-sm">{app.description}</div>
              </button>
            ))}
          </div>
          {/* Floating arrow button y gradient... */}
          <div className="pointer-events-none absolute top-0 right-0 h-full w-24 bg-gradient-to-l from-white to-transparent" />
        </div>
        {/* Hide scrollbar for Webkit browsers */}
        <style jsx global>{`
          #featured-apps-carousel::-webkit-scrollbar {
            display: none;
          }
        `}</style>
      </div>
      {/* Modal for app details */}
      <Dialog open={modalOpen} onOpenChange={setModalOpen}>
        <DialogContent className="max-w-4xl w-full p-0 overflow-hidden">
          <DialogTitle className="sr-only">{selectedApp?.title || "App Details"}</DialogTitle>
          {selectedApp && !showTool && (
            <div className="flex flex-col md:flex-row h-full">
              {/* Left: Main content */}
              <div className="flex-1 p-8">
                {/* Header: Icon, Title, Badges, Add to favorites */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-4">
                    <span className="flex items-center justify-center rounded-full bg-violet-100 w-14 h-14">
                      {selectedApp.icon}
                    </span>
                    <div>
                      <h2 className="text-2xl font-bold mb-1">{selectedApp.title}</h2>
                      <div className="flex gap-2 flex-wrap">
                        {selectedApp.details.map((badge, i) => (
                          <span key={i} className={`${badge.color} text-xs font-semibold px-2 py-0.5 rounded`}>{badge.label}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                  <button className="ml-4 flex items-center gap-1 text-xs text-gray-500 hover:text-violet-600 border border-gray-200 rounded-lg px-3 py-1 font-medium transition">
                    ☆ Add to favorites
                  </button>
                </div>
                {/* Overview */}
                <h3 className="font-semibold mb-1 mt-4">Overview</h3>
                <p className="mb-4 text-gray-700 text-base">{selectedApp.overview}</p>
                {/* Why use this app? */}
                <h3 className="font-semibold mb-1">Why use this app?</h3>
                <ul className="list-disc pl-5 text-gray-700 mb-4 space-y-1">
                  {selectedApp.why.map((reason, i) => {
                    // Bold the first part before ':' if present
                    const [bold, ...rest] = reason.split(":");
                    return (
                      <li key={i}>
                        <span className="font-semibold">{bold}{rest.length ? ":" : ""}</span>{rest.length ? <span> {rest.join(":")}</span> : null}
                      </li>
                    );
                  })}
                </ul>
                {/* Related Apps placeholder */}
                <div className="mt-8">
                  <h4 className="font-semibold mb-2">Related Apps</h4>
                  <div className="flex gap-2 text-gray-400 text-sm">(Coming soon)</div>
                </div>
              </div>
              {/* Right: Details and actions */}
              <div className="w-full md:w-80 bg-gray-50 border-l border-gray-100 p-8 flex flex-col gap-6">
                <div>
                  <h4 className="font-semibold mb-2">Details</h4>
                  <div className="flex gap-2 flex-wrap mb-4">
                    {selectedApp.details.map((badge, i) => (
                      <span key={i} className={`${badge.color} text-xs font-semibold px-2 py-0.5 rounded`}>{badge.label}</span>
                    ))}
                  </div>
                  <h4 className="font-semibold mb-2">Categories</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedApp.categories.map((cat, i) => (
                      <span key={i} className="bg-white border border-gray-200 text-gray-700 text-xs px-2 py-0.5 rounded flex items-center gap-1">
                        {cat}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="flex-1 flex items-end justify-end">
                  <button
                    type="button"
                    className="bg-violet-600 hover:bg-violet-700 text-white font-semibold px-6 py-2 rounded-lg transition w-full"
                    onClick={() => {
                      if (selectedApp.toolPath) {
                        window.open(selectedApp.toolPath, "_blank");
                      }
                    }}
                  >
                    Use App
                  </button>
                </div>
              </div>
              <DialogClose asChild>
                <button
                  aria-label="Close"
                  className="absolute top-4 right-4 text-5xl text-violet-600 bg-white border-2 border-violet-200 rounded-full w-16 h-16 flex items-center justify-center shadow-2xl cursor-pointer transition-all duration-300 hover:bg-violet-600 hover:text-white hover:rotate-180 hover:scale-125 hover:shadow-violet-300 focus:outline-none focus:ring-4 focus:ring-violet-400"
                >
                  <LucideX className="w-10 h-10" />
                </button>
              </DialogClose>
            </div>
          )}
          {selectedApp && showTool && selectedApp.title === "Background Remover" ? (
            <div className="flex flex-col md:flex-row h-[70vh] w-full">
              {/* Sidebar/tool */}
              <div className="w-full md:w-[420px] bg-white/80 border-r border-gray-100 p-8 flex flex-col">
                <button
                  className="mb-6 flex items-center gap-2 text-sm text-violet-600 hover:underline"
                  onClick={() => setShowTool(false)}
                >
                  ← Back
                </button>
                <h2 className="text-xl font-bold mb-1">Background Remover</h2>
                <p className="text-gray-500 mb-6">Effortlessly remove backgrounds from any image</p>
                <div className="mb-6">
                  <label className="block text-sm font-medium mb-2">Image</label>
                  <div className="flex flex-col items-center justify-center border-2 border-dashed border-violet-200 rounded-xl bg-violet-50/60 py-10 px-4 text-center cursor-pointer hover:border-violet-400 transition">
                    <span className="mb-2 text-violet-400">
                      <svg xmlns='http://www.w3.org/2000/svg' className='mx-auto' width='32' height='32' fill='none' viewBox='0 0 24 24' stroke='currentColor'><path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M12 16v-4m0 0V8m0 4h4m-4 0H8m12 4v1a3 3 0 01-3 3H7a3 3 0 01-3-3v-1a9 9 0 0118 0z' /></svg>
                    </span>
                    <span className="text-gray-400 text-sm">Drag & drop or <span className="text-violet-600 underline cursor-pointer">browse</span></span>
                    <span className="mt-4 text-xs text-gray-400">IMAGE</span>
                  </div>
                </div>
                <button className="mt-auto bg-violet-600 hover:bg-violet-700 text-white font-semibold w-full py-3 rounded-xl transition">Generate now</button>
              </div>
              {/* Document/editor area */}
              <div className="flex-1 bg-white/60 p-8 flex flex-col">
                <div className="flex items-center gap-4 mb-6">
                  <h3 className="text-lg font-bold">Untitled Document</h3>
                  <span className="bg-gray-100 text-gray-500 text-xs px-2 py-0.5 rounded">Normal text</span>
                </div>
                <div className="flex-1 border border-dashed border-gray-200 rounded-xl bg-white/80 flex items-center justify-center text-gray-400">
                  Press <span className="mx-1 font-mono text-violet-600">/</span> to tell Jasper what to write
                </div>
              </div>
            </div>
          ) : (
            selectedApp && (
              <div className="flex-1 flex items-end justify-end">
                <button
                  type="button"
                  className="bg-violet-600 hover:bg-violet-700 text-white font-semibold px-6 py-2 rounded-lg transition w-full"
                  onClick={() => {
                    if (selectedApp.toolPath) {
                      window.open(selectedApp.toolPath, "_blank");
                    }
                  }}
                >
                  Use App
                </button>
              </div>
            )
          )}
        </DialogContent>
      </Dialog>
      {/* Popular Apps Carousel */}
      <div className="mt-16">
        <h2 className="text-2xl font-bold mb-6 px-2">Popular Apps</h2>
        <div className="relative -mx-2">
          <div
            className="flex space-x-6 overflow-x-auto pb-4 pl-2 pr-10 snap-x snap-mandatory scrollbar-hide"
            style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
            id="popular-apps-carousel"
          >
            {popularApps.map((app) => (
              <button
                key={app.id}
                className="min-w-[320px] max-w-xs bg-white rounded-2xl border border-gray-200 p-6 flex-shrink-0 snap-start shadow-sm flex flex-col gap-3 hover:shadow-md transition"
                onClick={() => openAppModal(app)}
              >
                <span className="flex items-center justify-center rounded-full bg-violet-100 w-10 h-10 mb-2">
                  {app.icon}
                </span>
                <div className="flex gap-2 mb-1">
                  {app.badges.map((badge, i) => (
                    <span key={i} className={`${badge.color} text-xs font-semibold px-2 py-0.5 rounded`}>{badge.label}</span>
                  ))}
                </div>
                <div className="font-bold text-lg text-gray-900">{app.title}</div>
                <div className="text-gray-500 text-sm">{app.description}</div>
              </button>
            ))}
          </div>
          {/* Hide scrollbar for Webkit browsers */}
          <style jsx global>{`
            #popular-apps-carousel::-webkit-scrollbar {
              display: none;
            }
          `}</style>
          <div className="pointer-events-none absolute top-0 right-0 h-full w-24 bg-gradient-to-l from-white to-transparent" />
        </div>
      </div>
    </>
  );
} 