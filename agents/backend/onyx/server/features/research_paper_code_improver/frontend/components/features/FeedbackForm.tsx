'use client'

import React, { useState } from 'react'
import { Star, Send } from 'lucide-react'
import { Card, Button, Textarea, Badge } from '../ui'
import toast from 'react-hot-toast'

interface FeedbackFormProps {
  onSubmit?: (feedback: { rating: number; comment: string }) => void
  onCancel?: () => void
}

const FeedbackForm: React.FC<FeedbackFormProps> = ({
  onSubmit,
  onCancel,
}) => {
  const [rating, setRating] = useState(0)
  const [hoveredRating, setHoveredRating] = useState(0)
  const [comment, setComment] = useState('')

  const handleSubmit = () => {
    if (rating === 0) {
      toast.error('Please select a rating')
      return
    }

    onSubmit?.({ rating, comment })
    toast.success('Thank you for your feedback!')
    setRating(0)
    setComment('')
  }

  return (
    <Card>
      <div className="space-y-4">
        <div>
          <h3 className="font-semibold text-gray-900 mb-2">Rate this improvement</h3>
          <div className="flex items-center gap-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                type="button"
                onClick={() => setRating(star)}
                onMouseEnter={() => setHoveredRating(star)}
                onMouseLeave={() => setHoveredRating(0)}
                className="p-1 transition-transform hover:scale-110"
                aria-label={`Rate ${star} stars`}
              >
                <Star
                  className={`w-6 h-6 ${
                    star <= (hoveredRating || rating)
                      ? 'fill-yellow-400 text-yellow-400'
                      : 'text-gray-300'
                  } transition-colors`}
                />
              </button>
            ))}
            {rating > 0 && (
              <Badge variant="success" size="sm" className="ml-2">
                {rating}/5
              </Badge>
            )}
          </div>
        </div>

        <Textarea
          label="Comments (optional)"
          placeholder="Tell us what you think..."
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          rows={4}
        />

        <div className="flex gap-3">
          {onCancel && (
            <Button variant="outline" onClick={onCancel} className="flex-1">
              Cancel
            </Button>
          )}
          <Button
            variant="primary"
            onClick={handleSubmit}
            disabled={rating === 0}
            className="flex-1"
          >
            <Send className="w-4 h-4 mr-2" />
            Submit Feedback
          </Button>
        </div>
      </div>
    </Card>
  )
}

export default FeedbackForm




