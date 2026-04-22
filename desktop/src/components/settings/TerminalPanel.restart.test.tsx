import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'

import { useSettingsStore } from '../../stores/settingsStore'

const terminalMocks = vi.hoisted(() => ({
  startDesktopTerminal: vi.fn(),
  closeDesktopTerminal: vi.fn(),
  resizeDesktopTerminal: vi.fn().mockResolvedValue(undefined),
  writeDesktopTerminal: vi.fn().mockResolvedValue(undefined),
  listenDesktopTerminalEvents: vi.fn().mockResolvedValue(() => {}),
}))

vi.mock('../../lib/desktopRuntime', () => ({
  isTauriRuntime: () => true,
}))

vi.mock('../../lib/settingsTerminal', () => ({
  startDesktopTerminal: terminalMocks.startDesktopTerminal,
  closeDesktopTerminal: terminalMocks.closeDesktopTerminal,
  resizeDesktopTerminal: terminalMocks.resizeDesktopTerminal,
  writeDesktopTerminal: terminalMocks.writeDesktopTerminal,
  listenDesktopTerminalEvents: terminalMocks.listenDesktopTerminalEvents,
}))

class MockTerminal {
  cols = 100
  rows = 24

  loadAddon() {}

  open() {}

  focus() {}

  clear() {}

  write() {}

  onData() {
    return { dispose() {} }
  }

  dispose() {}
}

class MockFitAddon {
  fit() {}
}

vi.mock('@xterm/xterm', () => ({
  Terminal: MockTerminal,
}))

vi.mock('@xterm/addon-fit', () => ({
  FitAddon: MockFitAddon,
}))

import { TerminalPanel } from './TerminalPanel'

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((innerResolve) => {
    resolve = innerResolve
  })
  return { promise, resolve }
}

describe('TerminalPanel restart flow', () => {
  beforeEach(() => {
    useSettingsStore.setState({ locale: 'en' })

    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    )

    terminalMocks.startDesktopTerminal.mockReset()
    terminalMocks.closeDesktopTerminal.mockReset()
    terminalMocks.resizeDesktopTerminal.mockClear()
    terminalMocks.writeDesktopTerminal.mockClear()
    terminalMocks.listenDesktopTerminalEvents.mockClear()

    terminalMocks.startDesktopTerminal
      .mockResolvedValueOnce({
        sessionId: 'session-1',
        shell: 'bash',
        cwd: '/Users/nanmi',
        explicitCommandName: 'claude-haha',
        docsCommandName: 'claude',
      })
      .mockResolvedValueOnce({
        sessionId: 'session-2',
        shell: 'bash',
        cwd: '/tmp/restarted',
        explicitCommandName: 'claude-haha',
        docsCommandName: 'claude',
      })
  })

  it('switches to the new session without waiting for old close to finish', async () => {
    const oldClose = createDeferred<void>()
    terminalMocks.closeDesktopTerminal.mockImplementation((sessionId: string) => {
      if (sessionId === 'session-1') return oldClose.promise
      return Promise.resolve()
    })

    render(<TerminalPanel />)

    await screen.findByText('Shell: bash · Working directory: /Users/nanmi')

    fireEvent.click(screen.getByRole('button', { name: 'Restart shell' }))

    await waitFor(() =>
      expect(
        screen.getByText('Shell: bash · Working directory: /tmp/restarted'),
      ).toBeInTheDocument(),
    )
    expect(terminalMocks.closeDesktopTerminal).toHaveBeenCalledWith('session-1')

    oldClose.resolve()
  })
})
