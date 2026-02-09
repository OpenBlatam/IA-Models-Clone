/**
 * Validadores modulares para mensajes
 * Separación de lógica de validación
 */

import type {
  MessageAttachment,
  MessageLink,
  MessageNotification,
  MessageBookmark
} from '../types/message.types'

export interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings?: string[]
}

export class MessageValidator {
  /**
   * Valida un adjunto
   */
  static validateAttachment(attachment: MessageAttachment): ValidationResult {
    const errors: string[] = []

    if (!attachment.type) {
      errors.push('El tipo de adjunto es requerido')
    }

    if (!attachment.url) {
      errors.push('La URL del adjunto es requerida')
    } else if (!this.isValidUrl(attachment.url)) {
      errors.push('La URL del adjunto no es válida')
    }

    if (!attachment.name) {
      errors.push('El nombre del adjunto es requerido')
    }

    return {
      valid: errors.length === 0,
      errors
    }
  }

  /**
   * Valida un enlace
   */
  static validateLink(link: MessageLink): ValidationResult {
    const errors: string[] = []
    const warnings: string[] = []

    if (!link.url) {
      errors.push('La URL del enlace es requerida')
    } else if (!this.isValidUrl(link.url)) {
      errors.push('La URL del enlace no es válida')
    }

    if (!link.title) {
      warnings.push('Se recomienda agregar un título al enlace')
    }

    if (!link.description) {
      warnings.push('Se recomienda agregar una descripción al enlace')
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings: warnings.length > 0 ? warnings : undefined
    }
  }

  /**
   * Valida una notificación
   */
  static validateNotification(notification: MessageNotification): ValidationResult {
    const errors: string[] = []

    if (!notification.type) {
      errors.push('El tipo de notificación es requerido')
    }

    if (!notification.title) {
      errors.push('El título de la notificación es requerido')
    } else if (notification.title.length > 100) {
      errors.push('El título no puede exceder 100 caracteres')
    }

    if (!notification.body) {
      errors.push('El cuerpo de la notificación es requerido')
    } else if (notification.body.length > 500) {
      errors.push('El cuerpo no puede exceder 500 caracteres')
    }

    if (typeof notification.read !== 'boolean') {
      errors.push('El estado de lectura debe ser un booleano')
    }

    if (!notification.timestamp || notification.timestamp <= 0) {
      errors.push('El timestamp debe ser válido')
    }

    return {
      valid: errors.length === 0,
      errors
    }
  }

  /**
   * Valida un marcador
   */
  static validateBookmark(bookmark: MessageBookmark): ValidationResult {
    const errors: string[] = []

    if (!bookmark.name) {
      errors.push('El nombre del marcador es requerido')
    } else if (bookmark.name.length > 50) {
      errors.push('El nombre no puede exceder 50 caracteres')
    }

    if (!bookmark.category) {
      errors.push('La categoría del marcador es requerida')
    }

    if (!Array.isArray(bookmark.tags)) {
      errors.push('Los tags deben ser un array')
    } else if (bookmark.tags.length > 10) {
      errors.push('No se pueden tener más de 10 tags')
    }

    return {
      valid: errors.length === 0,
      errors
    }
  }

  /**
   * Valida múltiples adjuntos
   */
  static validateAttachments(attachments: MessageAttachment[]): ValidationResult {
    const errors: string[] = []

    if (attachments.length > 20) {
      errors.push('No se pueden tener más de 20 adjuntos por mensaje')
    }

    attachments.forEach((attachment, index) => {
      const result = this.validateAttachment(attachment)
      if (!result.valid) {
        errors.push(`Adjunto ${index + 1}: ${result.errors.join(', ')}`)
      }
    })

    return {
      valid: errors.length === 0,
      errors
    }
  }

  /**
   * Valida múltiples enlaces
   */
  static validateLinks(links: MessageLink[]): ValidationResult {
    const errors: string[] = []
    const warnings: string[] = []

    if (links.length > 50) {
      errors.push('No se pueden tener más de 50 enlaces por mensaje')
    }

    links.forEach((link, index) => {
      const result = this.validateLink(link)
      if (!result.valid) {
        errors.push(`Enlace ${index + 1}: ${result.errors.join(', ')}`)
      }
      if (result.warnings) {
        warnings.push(`Enlace ${index + 1}: ${result.warnings.join(', ')}`)
      }
    })

    return {
      valid: errors.length === 0,
      errors,
      warnings: warnings.length > 0 ? warnings : undefined
    }
  }

  /**
   * Valida una URL
   */
  private static isValidUrl(url: string): boolean {
    try {
      new URL(url)
      return true
    } catch {
      return false
    }
  }
}



