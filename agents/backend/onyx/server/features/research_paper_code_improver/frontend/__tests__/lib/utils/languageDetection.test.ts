/**
 * Tests for languageDetection utility
 */

import { detectLanguage } from '@/lib/utils/languageDetection'

describe('detectLanguage', () => {
  it('detects JavaScript', () => {
    expect(detectLanguage('function test() { return true; }')).toBe('javascript')
    expect(detectLanguage('const x = () => {}')).toBe('javascript')
    expect(detectLanguage('let arr = [1, 2, 3]')).toBe('javascript')
  })

  it('detects TypeScript', () => {
    expect(detectLanguage('const x: string = "test"')).toBe('typescript')
    expect(detectLanguage('interface Test { }')).toBe('typescript')
  })

  it('detects Python', () => {
    expect(detectLanguage('def test(): return True')).toBe('python')
    expect(detectLanguage('import os')).toBe('python')
    expect(detectLanguage('if __name__ == "__main__":')).toBe('python')
  })

  it('detects Java', () => {
    expect(detectLanguage('public class Test { }')).toBe('java')
    expect(detectLanguage('public static void main')).toBe('java')
  })

  it('detects C++', () => {
    expect(detectLanguage('#include <iostream>')).toBe('cpp')
    expect(detectLanguage('using namespace std;')).toBe('cpp')
  })

  it('detects C', () => {
    expect(detectLanguage('#include <stdio.h>')).toBe('c')
  })

  it('detects Go', () => {
    expect(detectLanguage('package main')).toBe('go')
    expect(detectLanguage('func main() { }')).toBe('go')
  })

  it('detects Rust', () => {
    expect(detectLanguage('fn main() { }')).toBe('rust')
    expect(detectLanguage('let x: i32 = 5;')).toBe('rust')
  })

  it('detects HTML', () => {
    expect(detectLanguage('<html><body></body></html>')).toBe('html')
    expect(detectLanguage('<!DOCTYPE html>')).toBe('html')
  })

  it('detects CSS', () => {
    expect(detectLanguage('.class { color: red; }')).toBe('css')
    expect(detectLanguage('@media screen { }')).toBe('css')
  })

  it('returns unknown for unrecognized code', () => {
    expect(detectLanguage('some random text')).toBe('unknown')
    expect(detectLanguage('')).toBe('unknown')
  })

  it('is case insensitive', () => {
    expect(detectLanguage('FUNCTION TEST() {}')).toBe('javascript')
    expect(detectLanguage('DEF TEST(): PASS')).toBe('python')
  })
})




