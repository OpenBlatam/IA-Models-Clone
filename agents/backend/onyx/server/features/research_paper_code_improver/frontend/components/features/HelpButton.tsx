'use client'

import React, { useState } from 'react'
import { HelpCircle, X } from 'lucide-react'
import { Button, Modal, Card } from '../ui'

interface HelpContent {
  title: string
  sections: Array<{
    heading: string
    content: string | React.ReactNode
  }>
}

interface HelpButtonProps {
  content: HelpContent
  variant?: 'primary' | 'outline' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
}

const HelpButton: React.FC<HelpButtonProps> = ({
  content,
  variant = 'ghost',
  size = 'sm',
}) => {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      <Button
        variant={variant}
        size={size}
        onClick={() => setIsOpen(true)}
        aria-label="Help"
      >
        <HelpCircle className="w-4 h-4" />
      </Button>

      <Modal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        title={content.title}
        size="lg"
      >
        <div className="space-y-6">
          {content.sections.map((section, index) => (
            <div key={index}>
              <h3 className="font-semibold text-gray-900 mb-2">
                {section.heading}
              </h3>
              <div className="text-sm text-gray-700 leading-relaxed">
                {typeof section.content === 'string' ? (
                  <p>{section.content}</p>
                ) : (
                  section.content
                )}
              </div>
            </div>
          ))}
        </div>
      </Modal>
    </>
  )
}

export default HelpButton




