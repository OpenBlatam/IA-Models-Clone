"use client";

import { Button } from "@/components/ui/button";
import { signIn } from "next-auth/react";
import { EB_Garamond } from "next/font/google";
import Image from "next/image";
import { Apple, Facebook } from "lucide-react";

const ebGaramond = EB_Garamond({ subsets: ['latin'] });

export default function LoginPage() {
  return (
    <div className={`min-h-screen w-screen flex flex-col items-center justify-start bg-white ${ebGaramond.className}`}
      style={{ minHeight: '100dvh' }}>
      {/* Logo grande arriba */}
      <div className="w-full flex flex-col items-center pt-8 pb-4">
        <Image
          src="/b_logo.png"
          alt="B Logo"
          width={70}
          height={70}
          className="mb-2"
          priority
        />
      </div>
      {/* Imagen editorial tipo banner */}
      <div className="w-full flex justify-center items-center mb-6">
        <div className="w-full max-w-md h-[160px] rounded-md overflow-hidden flex items-center justify-center">
          <Image
            src="https://images.unsplash.com/photo-1517841905240-472988babdf9?auto=format&fit=crop&w=800&q=80"
            alt="Editorial Fashion"
            width={600}
            height={160}
            className="object-cover w-full h-full"
            priority
          />
        </div>
      </div>
      {/* Formulario y copy */}
      <div className="w-full max-w-md flex flex-col items-center px-4 flex-1">
        <h1 className="text-2xl font-bold tracking-wide text-center mb-2 uppercase">Inicia sesión o crea tu cuenta</h1>
        <p className="text-base text-gray-600 text-center mb-4 font-light">Accede a la comunidad creativa más exclusiva y conecta con mentes brillantes.</p>
        <p className="text-xs text-gray-400 text-center mb-6">Al continuar, aceptas nuestros <a href="#" className="underline">Términos</a> y <a href="#" className="underline">Política de Privacidad</a>.</p>
        <input
          type="email"
          placeholder="Correo electrónico"
          className="w-full border border-gray-300 rounded px-4 py-3 text-base font-light focus:outline-none focus:border-black focus:ring-0 transition-colors placeholder-gray-400 mb-4"
        />
        <Button className="w-full h-12 text-base font-semibold bg-black text-white hover:bg-gray-900 rounded-none tracking-widest uppercase mb-4">
          Continuar con email
        </Button>
        <div className="flex items-center w-full my-2">
          <div className="flex-1 h-px bg-gray-200" />
          <span className="mx-3 text-gray-400 uppercase text-xs tracking-widest">o</span>
          <div className="flex-1 h-px bg-gray-200" />
        </div>
        <div className="grid grid-cols-1 gap-3 w-full mb-2">
          <Button
            variant="outline"
            className="w-full flex items-center justify-center h-12 border-gray-300 bg-white hover:bg-gray-50 rounded-none text-base font-medium gap-2"
            onClick={() => signIn("google", { callbackUrl: "/dashboard" })}
          >
            <svg className="h-5 w-5" viewBox="0 0 24 24">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
            </svg>
            Google
          </Button>
          <Button
            variant="outline"
            className="w-full flex items-center justify-center h-12 border-gray-300 bg-white hover:bg-gray-50 rounded-none text-base font-medium gap-2"
            onClick={() => {}}
          >
            <Apple className="h-5 w-5 text-black" />
            Apple
          </Button>
          <Button
            variant="outline"
            className="w-full flex items-center justify-center h-12 border-gray-300 bg-white hover:bg-gray-50 rounded-none text-base font-medium gap-2"
            onClick={() => {}}
          >
            <Facebook className="h-5 w-5 text-[#1877F2]" />
            Facebook
          </Button>
        </div>
      </div>
    </div>
  );
}
