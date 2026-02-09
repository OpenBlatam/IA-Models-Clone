'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Share2, X, Copy, Check, Link, Mail, Twitter, Facebook } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { useCopyToClipboard } from '@/lib/hooks/useCopyToClipboard'
import * as QRCode from 'qrcode'
import { Model } from '@/store/modelStore'

interface ShareModalProps {
  model: Model
  isOpen: boolean
  onClose: () => void
}

export default function ShareModal({ model, isOpen, onClose }: ShareModalProps) {
  const [qrCodeUrl, setQrCodeUrl] = useState<string>('')
  const { copy, copied } = useCopyToClipboard()
  const shareUrl = model.githubUrl || (typeof window !== 'undefined' ? `${window.location.origin}/model/${model.id}` : '')

  useEffect(() => {
    if (isOpen && shareUrl && typeof window !== 'undefined') {
      import('qrcode').then((QRCode) => {
        QRCode.default.toDataURL(shareUrl, { width: 200 })
          .then((url: string) => setQrCodeUrl(url))
          .catch((err: any) => console.error('Error generating QR:', err))
      })
    }
  }, [isOpen, shareUrl])

  const handleShare = async (platform: string) => {
    const text = `Check out my TruthGPT model: ${model.name} - ${model.description}`
    
    switch (platform) {
      case 'twitter':
        window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(shareUrl)}`, '_blank')
        break
      case 'facebook':
        window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`, '_blank')
        break
      case 'email':
        window.location.href = `mailto:?subject=${encodeURIComponent(text)}&body=${encodeURIComponent(shareUrl)}`
        break
      case 'link':
        copy(shareUrl)
        break
    }
  }

  if (!isOpen) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="bg-slate-800 rounded-lg border border-slate-700 max-w-md w-full"
        >
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Share2 className="w-6 h-6 text-purple-400" />
                <h2 className="text-xl font-bold text-white">Compartir Modelo</h2>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>

            <div className="space-y-4">
              {/* Model Info */}
              <div className="p-4 bg-slate-700/30 rounded-lg border border-slate-600">
                <p className="text-sm font-medium text-white mb-1">{model.name}</p>
                <p className="text-xs text-slate-400">{model.description}</p>
              </div>

              {/* Share URL */}
              <div>
                <label className="text-sm font-medium text-slate-300 mb-2 block">
                  URL de Compartir
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={shareUrl}
                    readOnly
                    className="flex-1 px-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white text-sm"
                  />
                  <button
                    onClick={() => handleShare('link')}
                    className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                  >
                    {copied ? (
                      <Check className="w-5 h-5 text-green-400" />
                    ) : (
                      <Copy className="w-5 h-5 text-slate-300" />
                    )}
                  </button>
                </div>
              </div>

              {/* QR Code */}
              {qrCodeUrl && (
                <div className="flex justify-center">
                  <img src={qrCodeUrl} alt="QR Code" className="rounded-lg border border-slate-600" />
                </div>
              )}

              {/* Share Buttons */}
              <div className="grid grid-cols-3 gap-3">
                <button
                  onClick={() => handleShare('twitter')}
                  className="flex flex-col items-center gap-2 p-4 bg-slate-700/50 hover:bg-slate-600 rounded-lg transition-colors"
                >
                  <Twitter className="w-5 h-5 text-blue-400" />
                  <span className="text-xs text-slate-300">Twitter</span>
                </button>
                <button
                  onClick={() => handleShare('facebook')}
                  className="flex flex-col items-center gap-2 p-4 bg-slate-700/50 hover:bg-slate-600 rounded-lg transition-colors"
                >
                  <Facebook className="w-5 h-5 text-blue-600" />
                  <span className="text-xs text-slate-300">Facebook</span>
                </button>
                <button
                  onClick={() => handleShare('email')}
                  className="flex flex-col items-center gap-2 p-4 bg-slate-700/50 hover:bg-slate-600 rounded-lg transition-colors"
                >
                  <Mail className="w-5 h-5 text-purple-400" />
                  <span className="text-xs text-slate-300">Email</span>
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

