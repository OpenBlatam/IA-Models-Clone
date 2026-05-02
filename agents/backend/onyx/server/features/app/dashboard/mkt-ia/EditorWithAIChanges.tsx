import './styles.scss'
import './extension-styles.css'

import { Decoration } from '@tiptap/pm/view'
import { EditorContent, useEditor } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Ai from '@tiptap-pro/extension-ai'
import AiChanges from '@tiptap-pro/extension-ai-changes'
import React, { useEffect, useRef, useState } from 'react'

import { variables } from '../../../variables'
import { ReviewChangePopover } from './components/ReviewChangePopover'

export default function EditorWithAIChanges() {
  const [state, setState] = useState({
    isLoading: false,
    errorMessage: null,
    response: null,
  })
  const [streamMode, setStreamMode] = useState(true)
  const [tooltipElement, setTooltipElement] = useState(null)

  const editor = useEditor({
    immediatelyRender: true,
    shouldRerenderOnTransaction: true,
    extensions: [
      StarterKit,
      Ai.configure({
        appId: 'APP_ID_HERE',
        token: 'TOKEN_HERE',
        // ATTENTION: This is for demo purposes only
        baseUrl: variables.tiptapAiBaseUrl,
        autocompletion: true,
        onLoading: () => {
          setState({
            isLoading: true,
            errorMessage: null,
            response: '',
          })
        },
        onChunk: ({ response }) => {
          setState({
            isLoading: true,
            errorMessage: null,
            response,
          })
        },
        onSuccess: context => {
          setState({
            isLoading: false,
            errorMessage: null,
            response: context.response,
          })
          context.editor.commands.setShowAiChanges(true)
        },
        onError: (error, context) => {
          console.error(error)
          setState({
            isLoading: false,
            errorMessage: error.message,
            response: null,
          })
          context.editor.commands.stopTrackingAiChanges()
        },
      }),
      AiChanges.configure({
        getCustomDecorations({ change, isSelected, getDefaultDecorations }) {
          const decorations = getDefaultDecorations()

          if (isSelected) {
            decorations.push(
              Decoration.widget(change.newRange.to, () => {
                const element = document.createElement('span')

                setTooltipElement(element)
                return element
              }),
            )
          }
          return decorations
        },
      }),
    ],
    content: `
      <p>
        Rocking like a mobile?
      </p>
      <p>
        Did you hear about the mobile phone that joined a rock band? Yeah, it was a real smartTONE!
        It rocked the stage with its gigabytes of rhythm and had everyone calling for an encore, but
        it couldn't resist the temptation to drop a few bars and left the audience in absolute silence.
        Turns out, it wasn't quite cut out for the music industry.
      </p>
      <p>
        They say it's now pursuing a career in ringtone composition. Who knows, maybe one day it'll top
        the charts with its catchy melodies!
      </p>
      <p></p>
    `,
  })

  if (!editor) {
    return null
  }

  const { empty: selectionIsEmpty, from: selectionFrom, to: selectionTo } = editor.state.selection
  const selectionContainsText = editor.state.doc.textBetween(selectionFrom, selectionTo, ' ')
  const disableAiGenerationButtons = selectionIsEmpty || !selectionContainsText

  const isTrackingAiChanges = editor.extensionStorage.aiChanges.getIsTrackingAiChanges()
  const changes = editor.extensionStorage.aiChanges.getChanges()

  const previousChangesRef = useRef([])

  useEffect(() => {
    if (isTrackingAiChanges && changes.length === 0 && previousChangesRef.current.length > 0) {
      editor.commands.stopTrackingAiChanges()
    }
    previousChangesRef.current = changes
  }, [isTrackingAiChanges, changes, editor])

  return (
    <>
      <div className="control-group">
        {!isTrackingAiChanges && (
          <div className="flex-row">
            <div className="switch-group">
              <label>
                <input
                  type="radio"
                  name="option-switch"
                  onChange={() => setStreamMode(false)}
                  checked={streamMode === false}
                />
                Non-streaming
              </label>
              <label>
                <input
                  type="radio"
                  name="option-switch"
                  onChange={() => setStreamMode(true)}
                  checked={streamMode === true}
                />
                Streaming
              </label>
            </div>
            <div className="hint">💡 Select text to improve</div>
          </div>
        )}
        {!isTrackingAiChanges && (
          <div className="button-group">
            <button
              onClick={() => editor
                .chain()
                .focus()
                .startTrackingAiChanges()
                .setShowAiChanges(false)
                .aiSummarize({ stream: streamMode })
                .run()
              }
              disabled={disableAiGenerationButtons}
            >
              Summarize
            </button>
            <button
              onClick={() => editor
                .chain()
                .focus()
                .startTrackingAiChanges()
                .setShowAiChanges(false)
                .aiComplete({ append: true, stream: streamMode })
                .run()
              }
              disabled={disableAiGenerationButtons}
            >
              Continue writing
            </button>
            <button
              onClick={() => editor
                .chain()
                .focus()
                .startTrackingAiChanges()
                .setShowAiChanges(false)
                .aiEmojify({ stream: streamMode })
                .run()
              }
              disabled={disableAiGenerationButtons}
            >
              Enrich with 🙂
            </button>
            <button
              onClick={() => editor
                .chain()
                .focus()
                .startTrackingAiChanges()
                .setShowAiChanges(false)
                .aiDeEmojify({ stream: streamMode })
                .run()
              }
              disabled={disableAiGenerationButtons}
            >
              De-Emojify
            </button>
            <button
              onClick={() => editor
                .chain()
                .focus()
                .startTrackingAiChanges()
                .setShowAiChanges(false)
                .aiTranslate('de', { stream: streamMode })
                .run()
              }
              disabled={disableAiGenerationButtons}
            >
              Translate to&nbsp;<i>German</i>
            </button>
          </div>
        )}
        {isTrackingAiChanges && (
          <>
            <div className="label-small">Reviewing AI changes</div>
            <div className="button-group">
              <button onClick={() => editor.commands.stopTrackingAiChanges()}>
                Accept all
              </button>
              <button className='destructive' onClick={() => editor.chain().rejectAllAiChanges().stopTrackingAiChanges().run()}>
                Reject all
              </button>
            </div>
          </>
        )}

        {state.errorMessage && <div className="hint error">{state.errorMessage}</div>}
        {state.isLoading && <div className="hint purple-spinner">AI is generating</div>}
      </div>

      <EditorContent editor={editor} />
      <ReviewChangePopover editor={editor} element={tooltipElement} />
    </>
  )
} 