import './styles.scss'

import { EditorContent, useEditor } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Ai from '@tiptap-pro/extension-ai'
import React from 'react'

import { variables } from '../../../variables'

export default function EditorWithAI() {
  const editor = useEditor({
    extensions: [
      StarterKit,
      Ai.configure({
        appId: 'APP_ID_HERE',
        token: 'TOKEN_HERE',
        // WARNING: Remove this line in your code. It only works in the demo environment.
        baseUrl: variables.tiptapAiBaseUrl,
        autocompletion: true,
      }),
    ],
    content: `
      <p>As an editor suite, Tiptap offers extensive&nbsp;</p>
      <p></p>
    `,
  })

  if (!editor) {
    return null
  }

  return (
    <>
      <div className="control-group">
        <div className="hint">💡 Press <kbd>TAB ⇥</kbd> at the end of a text to autocomplete, twice to confirm</div>
      </div>
      <EditorContent editor={editor} />
    </>
  )
} 