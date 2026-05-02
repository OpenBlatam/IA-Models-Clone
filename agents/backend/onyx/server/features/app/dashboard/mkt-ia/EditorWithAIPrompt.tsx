import './styles.scss'

import Placeholder from '@tiptap/extension-placeholder'
import { EditorContent, useEditor } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Ai from '@tiptap-pro/extension-ai'
import React, { useState } from 'react'

import { variables } from '../../../variables'

export default function EditorWithAIPrompt() {
  const [prompt, setPrompt] = useState(
    "Write something about Tiptap Editor and include a list of it's core features",
  )
  const [shouldFormatResponse, setShouldFormatResponse] = useState(true)
  const editor = useEditor({
    extensions: [
      StarterKit,
      Ai.configure({
        appId: 'APP_ID_HERE',
        token: 'TOKEN_HERE',
        // WARNING: Remove this line in your code. It only works in the demo environment.
        baseUrl: variables.tiptapAiBaseUrl,
      }),
      Placeholder.configure({
        placeholder: 'Use the input above to generate content...',
      }),
    ],
  })

  if (!editor) {
    return null
  }

  /**
   * @type {import('@tiptap-pro/extension-ai').AiStorage}
   */
  const editorStorage = editor.storage.ai || editor.storage.aiAdvanced

  return (
    <>
      <div className="control-group">
        <div className="flex-row">
          <div className="input-group">
            <label className="label-large">Prompt</label>
            <textarea
              placeholder="Write your prompt for creating content here"
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
            ></textarea>
          </div>
          <div className="switch-group">
            <label>
              <input
                type="radio"
                name="option-switch"
                checked={shouldFormatResponse}
                onChange={() => setShouldFormatResponse(true)}
              />
              Formatted
            </label>
            <label>
              <input
                type="radio"
                name="option-switch"
                checked={!shouldFormatResponse}
                onChange={() => setShouldFormatResponse(false)}
              />
              Unformatted
            </label>
          </div>
          <button
            disabled={editorStorage.state === 'loading' || !prompt}
            className="primary"
            onClick={() => {
              editor.commands.aiTextPrompt({
                text: prompt,
                format: shouldFormatResponse ? 'rich-text' : 'plain-text',
                stream: true,
              })
            }}
          >
            Generate
          </button>
        </div>
        {editorStorage.state === 'loading' && (
          <div className="hint purple-spinner">AI is generating</div>
        )}
        {editorStorage.state === 'error' && (
          <div className="hint error">{editorStorage.error.message}</div>
        )}
      </div>
      <EditorContent editor={editor} />
    </>
  )
} 