"use client";
import { useState } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { X as LucideX, Copy, Maximize2, Download, Grid, Hand, Clock, Minus, Plus, Grid as GridIcon } from "lucide-react";
import WordEditor from "../WordEditor";
import PodcastScriptPanel from "./apps/PodcastScriptPanel";
import BackgroundRemoverPanel from "./apps/BackgroundRemoverPanel";
import BlogPostPanel from "./apps/BlogPostPanel";
import EmailSequencePanel from "./apps/EmailSequencePanel";
import FacebookPostPanel from "./apps/FacebookPostPanel";
import InstagramCaptionPanel from "./apps/InstagramCaptionPanel";
import LandingPagePanel from "./apps/LandingPagePanel";
import ProductDescriptionPanel from "./apps/ProductDescriptionPanel";

const APP_PANELS = {
  'podcast-script': PodcastScriptPanel,
  'background-remover': BackgroundRemoverPanel,
  'blog-post': BlogPostPanel,
  'email-sequence': EmailSequencePanel,
  'facebook-post': FacebookPostPanel,
  'instagram-caption': InstagramCaptionPanel,
  'landing-page': LandingPagePanel,
  'product-description': ProductDescriptionPanel,
};

function AppCard({ title, desc, icon, badge, onClick }: { title: any, desc: any, icon: any, badge?: any, onClick?: () => void }) {
  return (
    <div
      className="bg-white border border-gray-200 rounded-2xl p-5 min-w-[180px] flex flex-col relative mr-2 cursor-pointer shadow-sm hover:shadow-lg hover:border-blue-400 transition-all duration-200 group"
      onClick={onClick}
    >
      {badge && (
        <span className="absolute top-3 right-3 bg-pink-100 text-pink-600 text-xs font-bold px-2 py-0.5 rounded">
          {badge}
        </span>
      )}
      <div className="flex items-center justify-center mb-3">
        <span className="bg-blue-50 group-hover:bg-blue-100 text-2xl rounded-full p-3 border border-blue-100 transition">{icon}</span>
      </div>
      <div className="font-bold text-lg mb-1 text-gray-800">{title}</div>
      <div className="text-sm text-gray-500">{desc}</div>
    </div>
  );
}

function AppModal({ onClose, onSelectApp }: { onClose: () => void, onSelectApp: (app: string) => void }) {
  const apps = [
    {
      key: 'podcast-script',
      title: 'Podcast Script',
      desc: 'Craft well-structured podcast scripts...',
      icon: '🎤',
      badge: null,
      info: 'Create podcast scripts that keep listeners engaged from start to finish.',
      overview: 'Create podcast scripts that keep listeners engaged from start to finish.',
      bullets: [
        'Structure your episodes for maximum impact.',
        'Save time with ready-to-use templates.',
        'Ensure consistency in your podcast voice.'
      ],
      categories: ['Content Marketing', 'Audio', 'Podcasting'],
      details: ['POPULAR']
    },
    {
      key: 'background-remover',
      title: 'Background Remover',
      desc: 'Effortlessly remove backgrounds from any image',
      icon: '🔄',
      badge: <><span className="bg-green-200 text-green-800 text-xs font-bold px-2 py-0.5 rounded">NEW</span> <span className="bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span></>,
      info: 'Remove backgrounds from images with a single click.',
      overview: 'Effortlessly remove backgrounds from any image.',
      bullets: [
        'Quickly isolate subjects for design or presentations.',
        'No design skills required.',
        'Supports multiple image formats.'
      ],
      categories: ['Image Editing', 'Productivity'],
      details: ['NEW', 'POPULAR']
    },
    {
      key: 'blog-post',
      title: 'Blog Post',
      desc: 'Write long-form content that provides value, drives traffic, and enhances SEO',
      icon: '🖊️',
      badge: <span className="bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span>,
      info: 'Generate SEO-friendly blog posts quickly.',
      overview: 'Write long-form content that provides value, drives traffic, and enhances SEO.',
      bullets: [
        'SEO-optimized content in minutes.',
        'Customizable tone and length.',
        "Increase your site's authority."
      ],
      categories: ['Content Marketing', 'SEO'],
      details: ['POPULAR']
    },
    {
      key: 'email-sequence',
      title: 'Email Sequence',
      desc: 'Guide customer journeys and boost conversions with a tailored email sequence',
      icon: '✉️',
      badge: <span className="bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span>,
      info: 'Create effective email sequences for your campaigns.',
      overview: 'Guide customer journeys and boost conversions with a tailored email sequence.',
      bullets: [
        'Automate your email marketing.',
        'Personalize each message.',
        'Track open and click rates.'
      ],
      categories: ['Email Marketing', 'Automation'],
      details: ['POPULAR']
    },
    {
      key: 'facebook-post',
      title: 'Facebook Post',
      desc: 'Foster engagement and amplify reach using engaging Facebook updates',
      icon: '👍',
      badge: <span className="bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span>,
      info: 'Write posts that boost engagement on Facebook.',
      overview: 'Foster engagement and amplify reach using engaging Facebook updates.',
      bullets: [
        "Increase your page's visibility.",
        'Engage your audience with creative posts.',
        'Schedule posts in advance.'
      ],
      categories: ['Social Media', 'Brand Marketing'],
      details: ['POPULAR']
    },
    {
      key: 'instagram-caption',
      title: 'Instagram Caption',
      desc: 'Boost engagement with captions that perfectly accompany your Instagram images',
      icon: '📷',
      badge: <span className="bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span>,
      info: 'Generate catchy captions for Instagram.',
      overview: "Enhance your Instagram strategy with captivating captions that complement your visuals and drive audience engagement. This app assists in crafting unique, attention-grabbing text that aligns with your brand's voice, deepens audience connection, and encourages interactions.",
      bullets: [
        'Increase Engagement: Generate captions that inspire likes, comments, and shares, boosting interaction rates and brand visibility.',
        "Strengthen Brand Voice: Ensure that every caption consistently reflects your brand's personality and messaging, maintaining a cohesive online presence.",
        'Streamline Content Management: Accelerate content production with quick, creative captions, enabling a faster response to social media trends and events.'
      ],
      categories: ['Social Media Marketing', 'Brand Marketing', 'Awareness', 'Creating Content'],
      details: ['POPULAR']
    },
    {
      key: 'landing-page',
      title: 'Landing Page',
      desc: 'Transform site traffic into valuable leads through engaging landing pages',
      icon: '🌀',
      badge: <span className="bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span>,
      info: 'Create high-converting landing pages.',
      overview: 'Transform site traffic into valuable leads through engaging landing pages.',
      bullets: [
        'Increase conversion rates.',
        'Easy-to-use drag and drop builder.',
        'Mobile responsive designs.'
      ],
      categories: ['Web', 'Lead Generation'],
      details: ['POPULAR']
    },
    {
      key: 'product-description',
      title: 'Product Description',
      desc: 'Compose detailed descriptions that highlight the benefits and features of a product',
      icon: '🧊',
      badge: <span className="bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">POPULAR</span>,
      info: 'Write compelling product descriptions.',
      overview: 'Compose detailed descriptions that highlight the benefits and features of a product.',
      bullets: [
        'Highlight product benefits and features.',
        'SEO-friendly descriptions.',
        'Increase conversion rates.'
      ],
      categories: ['E-commerce', 'Copywriting'],
      details: ['POPULAR']
    },
  ];
  const [carouselIndex, setCarouselIndex] = useState(0);
  const [showInfo, setShowInfo] = useState<null | number>(null);
  const visibleCount = 2;
  const canPrev = carouselIndex > 0;
  const canNext = carouselIndex < apps.length - visibleCount;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div className="bg-white/60 backdrop-blur-md border border-blue-200 rounded-2xl shadow-2xl w-full max-w-xl p-6 relative flex flex-col">
        {/* Encabezado sticky */}
        <div className="flex items-center justify-between px-4 py-3 border-b sticky top-0 bg-white/70 backdrop-blur-md z-10">
          <div className="flex items-center gap-6">
            <button className="font-semibold text-blue-700 border-b-2 border-blue-700 pb-1">Recently used <span className="ml-1 bg-gray-100 text-gray-500 text-xs px-2 py-0.5 rounded">{apps.length}</span></button>
            <button className="text-gray-400">Workspace Apps</button>
            <button className="text-gray-400">My Apps</button>
            <button className="text-gray-400">Favorites</button>
          </div>
          <div className="flex items-center gap-2">
            <button className="border border-gray-200 rounded-lg px-4 py-1 text-gray-800 font-medium hover:bg-gray-50">Browse All</button>
            <button className="text-gray-400 hover:text-gray-600 ml-2" onClick={onClose}><LucideX /></button>
          </div>
        </div>
        {/* Carrusel con flechas */}
        <div className="relative flex items-center justify-center px-4 py-4 border-b bg-white">
          <button
            className={`absolute left-0 z-10 bg-white rounded-full shadow p-1 border ${!canPrev ? 'opacity-30 cursor-not-allowed' : ''}`}
            onClick={() => canPrev && setCarouselIndex(i => i - 1)}
            disabled={!canPrev}
            aria-label="Previous"
            style={{top: '50%', transform: 'translateY(-50%)'}}
          >
            <span className="text-2xl">&#8592;</span>
          </button>
          <div className="flex gap-4 px-2 py-2 overflow-hidden w-[340px] md:w-[400px]">
            {apps.slice(carouselIndex, carouselIndex + visibleCount).map((app, idx) => (
              <div key={app.key} className="transition-transform duration-300" style={{ minWidth: 140, maxWidth: 200, flex: 1 }}>
                <AppCard
                  title={app.title}
                  desc={app.desc}
                  icon={app.icon}
                  badge={app.badge}
                  onClick={() => setShowInfo(carouselIndex + idx)}
                />
              </div>
            ))}
          </div>
          <button
            className={`absolute right-0 z-10 bg-white rounded-full shadow p-1 border ${!canNext ? 'opacity-30 cursor-not-allowed' : ''}`}
            onClick={() => canNext && setCarouselIndex(i => i + 1)}
            disabled={!canNext}
            aria-label="Next"
            style={{top: '50%', transform: 'translateY(-50%)'}}
          >
            <span className="text-2xl">&#8594;</span>
          </button>
          <div className="pointer-events-none absolute left-0 top-0 h-full w-8 bg-gradient-to-r from-white/90 to-transparent" />
          <div className="pointer-events-none absolute right-0 top-0 h-full w-8 bg-gradient-to-l from-white/90 to-transparent" />
        </div>
        {/* Modal explicativo tipo ficha */}
        {showInfo !== null && apps[showInfo] && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
            <div className="bg-white/60 backdrop-blur-md rounded-xl shadow-2xl p-0 max-w-3xl w-full relative animate-fade-in flex flex-col md:flex-row overflow-hidden border border-blue-200">
              {/* Columna izquierda: info principal */}
              <div className="flex-1 p-8 min-w-[320px] max-h-[80vh] overflow-y-auto">
                <button className="mb-4 text-gray-500 hover:text-gray-700 flex items-center gap-1" onClick={() => setShowInfo(null)}>
                  <span className="text-lg">&#8592;</span> <span>Back</span>
                </button>
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-4xl bg-gray-100 rounded-full p-2 border border-gray-200">{apps[showInfo].icon}</span>
                  <span className="font-bold text-2xl">{apps[showInfo].title}</span>
                  {apps[showInfo].details && apps[showInfo].details.map((d, i) => (
                    <span key={i} className="ml-2 bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">{d}</span>
                  ))}
                </div>
                <div className="text-gray-700 font-semibold mb-2 mt-4">Overview</div>
                <div className="text-gray-600 mb-4 text-base leading-relaxed">{apps[showInfo].overview}</div>
                <div className="text-gray-700 font-semibold mb-2">Why use this app?</div>
                <ul className="list-disc pl-5 text-gray-600 text-base mb-4 space-y-1">
                  {apps[showInfo].bullets.map((b, i) => <li key={i}>{b}</li>)}
                </ul>
                <div className="text-gray-700 font-semibold mb-2">Related Apps</div>
                <div className="text-xs text-gray-400">(Coming soon)</div>
              </div>
              {/* Columna derecha: detalles y acción */}
              <div className="w-full md:w-80 bg-gray-50 border-l p-8 flex flex-col gap-6 justify-between max-h-[80vh] overflow-y-auto">
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <span className="font-semibold text-gray-700">Details</span>
                    {apps[showInfo].details && apps[showInfo].details.map((d, i) => (
                      <span key={i} className="bg-pink-200 text-pink-800 text-xs font-bold px-2 py-0.5 rounded">{d}</span>
                    ))}
                  </div>
                  <div className="mb-6">
                    <span className="font-semibold text-gray-700">Categories</span>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {apps[showInfo].categories.map((cat, i) => (
                        <span key={i} className="bg-gray-200 text-gray-700 text-xs px-3 py-1 rounded-full font-medium">{cat}</span>
                      ))}
                    </div>
                  </div>
                </div>
                <button
                  className="w-full bg-blue-600 text-white rounded-lg py-3 font-semibold text-lg shadow hover:bg-blue-700 transition"
                  onClick={() => { onSelectApp(apps[showInfo].key); setShowInfo(null); onClose(); }}
                >Use App</button>
                <button className="w-full border border-gray-300 rounded-lg py-3 font-semibold text-gray-700 hover:bg-gray-100 transition">Add to favorites</button>
              </div>
              <button className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 text-2xl" onClick={() => setShowInfo(null)}><LucideX /></button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function MktIaCanvasPage() {
  const [showAppModal, setShowAppModal] = useState(false);
  const [selectedApp, setSelectedApp] = useState<string>('podcast-script');
  const SelectedPanel = APP_PANELS[selectedApp] || (() => <div />);

  return (
    <div className="relative min-h-screen bg-gray-50 flex flex-col">
      <div className="flex gap-8 px-8 py-12">
        {/* Panel izquierdo fijo y dinámico */}
        <div className="bg-white rounded-xl border shadow p-6 w-[420px] flex flex-col">
          <SelectedPanel />
        </div>
        {/* Panel derecho drag/zoom */}
        <div className="flex-1 flex flex-col">
      <TransformWrapper
        initialScale={1}
        minScale={0.1}
        maxScale={2}
        wheel={{ step: 0.1 }}
        doubleClick={{ disabled: true }}
      >
            {({ zoomIn, zoomOut, resetTransform, instance }) => (
          <>
            {/* Barra flotante de controles */}
            <div className="fixed bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 bg-white border border-gray-200 rounded-full shadow-lg px-3 py-2 z-50">
              <button className="p-2 rounded-full bg-blue-600 text-white border border-blue-600"><Hand className="w-5 h-5" /></button>
                  <button className="p-2 rounded-full hover:bg-gray-100" onClick={() => resetTransform()}><Clock className="w-5 h-5" /></button>
              <div className="flex items-center bg-white border border-gray-200 rounded-full px-2 py-1 mx-1">
                    <button className="p-2 rounded-full hover:bg-gray-100" onClick={() => zoomOut()}><Minus className="w-5 h-5" /></button>
                    <span className="text-gray-700 font-medium px-2 select-none" style={{minWidth: 48}}>{Math.round(instance.transformState.scale * 100)}%</span>
                    <button className="p-2 rounded-full hover:bg-gray-100" onClick={() => zoomIn()}><Plus className="w-5 h-5" /></button>
              </div>
              <button className="p-2 rounded-full hover:bg-gray-100"><GridIcon className="w-5 h-5" /></button>
            </div>
            {/* Canvas escalable y draggable */}
                <TransformComponent wrapperClass="flex-1 flex justify-center items-start overflow-x-auto">
                <div className="bg-white rounded-xl border shadow p-6 w-[520px] flex flex-col">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold">Writing Tips LinkedIn Post</span>
                    <button className="p-1 rounded hover:bg-gray-100"><Maximize2 className="w-5 h-5" /></button>
                  </div>
                  <div className="flex-1 min-h-[300px]">
                    <WordEditor />
                </div>
              </div>
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
        </div>
      </div>
      {/* Modal flotante de apps */}
      {showAppModal && <AppModal onClose={() => setShowAppModal(false)} onSelectApp={setSelectedApp} />}
      {/* Botón flotante para abrir el modal */}
      <button
        className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50 bg-white border border-gray-200 shadow-lg rounded-full p-4 flex items-center gap-2 hover:bg-gray-50"
        onClick={() => setShowAppModal(true)}
        aria-label="Open App Selector"
      >
        <Grid className="w-6 h-6 text-blue-600" />
      </button>
    </div>
  );
} 