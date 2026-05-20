'use client';

import { useState } from 'react';
import { MessageSquare, Star, Send, X } from 'lucide-react';
import toast from 'react-hot-toast';

export function MusicFeedback() {
  const [isOpen, setIsOpen] = useState(false);
  const [rating, setRating] = useState(0);
  const [feedback, setFeedback] = useState('');

  const handleSubmit = () => {
    if (rating === 0) {
      toast.error('Por favor, selecciona una calificación');
      return;
    }

    // En producción, esto enviaría el feedback al backend
    console.log('Feedback:', { rating, feedback });
    toast.success('¡Gracias por tu feedback!');
    setRating(0);
    setFeedback('');
    setIsOpen(false);
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 left-6 p-4 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg transition-colors z-50"
        title="Enviar feedback"
      >
        <MessageSquare className="w-6 h-6" />
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 left-6 bg-gradient-to-br from-blue-900 to-purple-900 rounded-xl p-6 max-w-md w-full border border-white/20 shadow-2xl z-50">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Enviar Feedback</h3>
        <button
          onClick={() => setIsOpen(false)}
          className="p-1 hover:bg-white/10 rounded transition-colors"
        >
          <X className="w-4 h-4 text-white" />
        </button>
      </div>

      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-2">Calificación</label>
        <div className="flex gap-2">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              onClick={() => setRating(star)}
              className={`p-2 rounded transition-colors ${
                star <= rating
                  ? 'text-yellow-400 bg-yellow-500/20'
                  : 'text-gray-400 hover:text-yellow-400'
              }`}
            >
              <Star className="w-5 h-5 fill-current" />
            </button>
          ))}
        </div>
      </div>

      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-2">Comentarios</label>
        <textarea
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder="¿Qué te gustaría mejorar?"
          rows={4}
          className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>

      <button
        onClick={handleSubmit}
        className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
      >
        <Send className="w-4 h-4" />
        Enviar
      </button>
    </div>
  );
}


