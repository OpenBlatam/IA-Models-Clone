import {
  autoUpdate, offset, shift, useFloating,
} from '@floating-ui/react'
import { createPortal } from 'react-dom'

/**
 * Shows a popover on the selected change. Lets the user accept or reject the
 * change made by the AI.
 */
export function ReviewChangePopover({
  element,
  editor,
}) {
  const selectedChange = editor.extensionStorage.aiChanges.getSelectedChange()
  const isOpen = Boolean(element && selectedChange)

  const { refs, floatingStyles } = useFloating({
    open: Boolean(element),
    middleware: [offset(8), shift({ padding: 8 })],
    whileElementsMounted: autoUpdate,
  })

  return (
    <>
      {element && createPortal(<span ref={refs.setReference}></span>, element)}
      {isOpen && (
        <div
          className="popover-parent"
          ref={refs.setFloating}
          style={floatingStyles}
        >
          <div className="popover">
            <div className="button-group">
              <button
                type="button"
                onClick={() => {
                  editor
                    .chain()
                    .acceptAiChange(selectedChange.id)
                    .focus()
                    .run()
                }}
              >
                Accept
              </button>
              <button
                type="button"
                className="destructive"
                onClick={() => {
                  editor
                    .chain()
                    .rejectAiChange(selectedChange.id)
                    .focus()
                    .run()
                }}
              >
                Reject
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
} 