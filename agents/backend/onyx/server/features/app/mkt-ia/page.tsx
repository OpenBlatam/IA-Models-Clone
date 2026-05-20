import Link from 'next/link'
import Image from 'next/image'

export default function MKTIA() {
  return (
    <main className="min-h-screen bg-white">
      {/* Hero Section */}
      <section className="relative w-full min-h-[80vh] flex items-center justify-center px-8 py-24">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-7xl md:text-8xl font-serif font-extrabold italic text-black tracking-tight leading-tight uppercase mb-8">
            Marketing con IA
          </h1>
          <p className="text-2xl md:text-3xl text-neutral-700 font-light max-w-3xl mx-auto mb-12">
            Transforma tu estrategia de marketing con inteligencia artificial
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link 
              href="/contact"
              className="px-8 py-4 bg-gradient-to-r from-[#FFD6E3] to-[#E3D6FF] text-black font-bold rounded-full hover:shadow-lg transition-all duration-300 text-lg"
            >
              Comienza Ahora
            </Link>
            <Link 
              href="#features"
              className="px-8 py-4 bg-white border-2 border-black text-black font-bold rounded-full hover:bg-black hover:text-white transition-all duration-300 text-lg"
            >
              Descubre Más
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="w-full py-24 px-8">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-5xl font-serif italic font-extrabold mb-16 text-center text-black">Características Principales</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            <div className="p-8 rounded-2xl border-2 border-[#E3D6FF] hover:shadow-lg transition-all duration-300">
              <div className="w-16 h-16 rounded-full bg-[#F3F7FF] flex items-center justify-center mb-6">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#FFD700" strokeWidth="2">
                  <path d="M12 2L15.09 8.26L22 9.27L17 12.91L18.18 21L12 17.27L5.82 21L7 12.91L2 9.27L8.91 8.26L12 2Z"/>
                </svg>
              </div>
              <h3 className="text-2xl font-bold mb-4 text-black">Análisis Predictivo</h3>
              <p className="text-neutral-700">Anticipa tendencias y comportamientos de tu audiencia con IA avanzada.</p>
            </div>
            <div className="p-8 rounded-2xl border-2 border-[#E3D6FF] hover:shadow-lg transition-all duration-300">
              <div className="w-16 h-16 rounded-full bg-[#F3F7FF] flex items-center justify-center mb-6">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#FFD700" strokeWidth="2">
                  <path d="M17 21V19C17 17.9391 16.5786 16.9217 15.8284 16.1716C15.0783 15.4214 14.0609 15 13 15H5C3.93913 15 2.92172 15.4214 2.17157 16.1716C1.42143 16.9217 1 17.9391 1 19V21"/>
                </svg>
              </div>
              <h3 className="text-2xl font-bold mb-4 text-black">Personalización</h3>
              <p className="text-neutral-700">Crea experiencias únicas para cada cliente con IA generativa.</p>
            </div>
            <div className="p-8 rounded-2xl border-2 border-[#E3D6FF] hover:shadow-lg transition-all duration-300">
              <div className="w-16 h-16 rounded-full bg-[#F3F7FF] flex items-center justify-center mb-6">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#FFD700" strokeWidth="2">
                  <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z"/>
                </svg>
              </div>
              <h3 className="text-2xl font-bold mb-4 text-black">Automatización</h3>
              <p className="text-neutral-700">Optimiza tus campañas de marketing con IA en tiempo real.</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="w-full bg-gradient-to-r from-[#FFD6E3] to-[#E3D6FF] py-24 px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-5xl font-serif italic font-extrabold mb-8 text-black">¿Listo para Transformar tu Marketing?</h2>
          <p className="text-xl text-neutral-700 mb-12">Únete a cientos de empresas que ya están revolucionando su estrategia con IA.</p>
          <Link 
            href="/contact"
            className="px-8 py-4 bg-black text-white font-bold rounded-full hover:bg-white hover:text-black transition-all duration-300 text-lg inline-block"
          >
            Agenda una Demo
          </Link>
        </div>
      </section>
    </main>
  )
} 