/**
 * Tests para Utilidades de Performance
 * =====================================
 */

import {
  memoize,
  memoizeWithTTL,
  debounce,
  throttle,
  batchProcess,
  measureTime,
  deepEqual,
  stableObject
} from '../performanceUtils'

describe('Performance Utils', () => {
  describe('memoize', () => {
    it('debe memoizar resultados', () => {
      let callCount = 0
      const fn = memoize((n: number) => {
        callCount++
        return n * 2
      })

      expect(fn(5)).toBe(10)
      expect(callCount).toBe(1)

      expect(fn(5)).toBe(10)
      expect(callCount).toBe(1) // No debe llamar de nuevo

      expect(fn(10)).toBe(20)
      expect(callCount).toBe(2)
    })

    it('debe respetar el límite de cache', () => {
      const fn = memoize((n: number) => n, 2)

      fn(1)
      fn(2)
      fn(3) // Debe eliminar 1 del cache

      // 1 ya no está en cache, debe recalcular
      fn(1)
    })
  })

  describe('memoizeWithTTL', () => {
    beforeEach(() => {
      jest.useFakeTimers()
    })

    afterEach(() => {
      jest.useRealTimers()
    })

    it('debe expirar el cache después de TTL', async () => {
      let callCount = 0
      const fn = memoizeWithTTL((n: number) => {
        callCount++
        return n * 2
      }, 1000)

      expect(fn(5)).toBe(10)
      expect(callCount).toBe(1)

      // Avanzar tiempo pero no suficiente para expirar
      jest.advanceTimersByTime(500)
      expect(fn(5)).toBe(10)
      expect(callCount).toBe(1)

      // Avanzar tiempo suficiente para expirar
      jest.advanceTimersByTime(600)
      expect(fn(5)).toBe(10)
      expect(callCount).toBe(2) // Debe recalcular
    })
  })

  describe('debounce', () => {
    beforeEach(() => {
      jest.useFakeTimers()
    })

    afterEach(() => {
      jest.useRealTimers()
    })

    it('debe debounce llamadas', () => {
      let callCount = 0
      const debounced = debounce(() => {
        callCount++
      }, 100)

      debounced()
      debounced()
      debounced()

      expect(callCount).toBe(0)

      jest.advanceTimersByTime(100)
      expect(callCount).toBe(1)
    })

    it('debe cancelar debounce', () => {
      let callCount = 0
      const debounced = debounce(() => {
        callCount++
      }, 100)

      debounced()
      debounced.cancel()

      jest.advanceTimersByTime(100)
      expect(callCount).toBe(0)
    })
  })

  describe('throttle', () => {
    beforeEach(() => {
      jest.useFakeTimers()
    })

    afterEach(() => {
      jest.useRealTimers()
    })

    it('debe throttlear llamadas', () => {
      let callCount = 0
      const throttled = throttle(() => {
        callCount++
      }, 100)

      throttled()
      throttled()
      throttled()

      expect(callCount).toBe(1)

      jest.advanceTimersByTime(100)
      throttled()
      expect(callCount).toBe(2)
    })
  })

  describe('batchProcess', () => {
    it('debe procesar items en lotes', async () => {
      const items = [1, 2, 3, 4, 5]
      const processor = async (batch: number[]) => batch.map(n => n * 2)

      const results = await batchProcess(items, processor, 2)
      expect(results).toEqual([2, 4, 6, 8, 10])
    })
  })

  describe('measureTime', () => {
    it('debe medir tiempo de ejecución', async () => {
      const fn = async () => {
        await new Promise(resolve => setTimeout(resolve, 100))
        return 'result'
      }

      const { result, time } = await measureTime(fn)
      expect(result).toBe('result')
      expect(time).toBeGreaterThan(0)
    })
  })

  describe('deepEqual', () => {
    it('debe comparar objetos profundamente', () => {
      expect(deepEqual({ a: 1, b: 2 }, { a: 1, b: 2 })).toBe(true)
      expect(deepEqual({ a: 1, b: 2 }, { a: 1, b: 3 })).toBe(false)
      expect(deepEqual({ a: { b: 1 } }, { a: { b: 1 } })).toBe(true)
      expect(deepEqual({ a: { b: 1 } }, { a: { b: 2 } })).toBe(false)
    })

    it('debe manejar arrays', () => {
      expect(deepEqual([1, 2, 3], [1, 2, 3])).toBe(true)
      expect(deepEqual([1, 2, 3], [1, 2, 4])).toBe(false)
    })
  })

  describe('stableObject', () => {
    it('debe crear objeto estable con keys ordenadas', () => {
      const obj = { c: 3, a: 1, b: 2 }
      const stable = stableObject(obj)

      expect(Object.keys(stable)).toEqual(['a', 'b', 'c'])
      expect(stable.a).toBe(1)
      expect(stable.b).toBe(2)
      expect(stable.c).toBe(3)
    })
  })
})







