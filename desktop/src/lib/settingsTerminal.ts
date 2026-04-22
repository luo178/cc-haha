import { isTauriRuntime } from './desktopRuntime'

export type DesktopTerminalStart = {
  sessionId: string
  shell: string
  cwd: string
  explicitCommandName: string
  docsCommandName: string
}

export type DesktopTerminalEvent =
  | { type: 'output'; sessionId: string; data: string }
  | { type: 'exit'; sessionId: string; exitCode?: number | null }

export async function startDesktopTerminal(cwd?: string) {
  if (!isTauriRuntime()) {
    throw new Error('Desktop terminal is only available in the Tauri runtime.')
  }

  const { invoke } = await import(/* @vite-ignore */ '@tauri-apps/api/core')
  return invoke<DesktopTerminalStart>('terminal_start_session', {
    cwd: cwd?.trim() || null,
  })
}

export async function writeDesktopTerminal(sessionId: string, data: string) {
  const { invoke } = await import(/* @vite-ignore */ '@tauri-apps/api/core')
  return invoke('terminal_write', { sessionId, data })
}

export async function resizeDesktopTerminal(
  sessionId: string,
  cols: number,
  rows: number,
) {
  const { invoke } = await import(/* @vite-ignore */ '@tauri-apps/api/core')
  return invoke('terminal_resize', {
    sessionId,
    cols,
    rows,
  })
}

export async function closeDesktopTerminal(sessionId: string) {
  const { invoke } = await import(/* @vite-ignore */ '@tauri-apps/api/core')
  return invoke('terminal_close', { sessionId })
}

export async function listenDesktopTerminalEvents(
  handler: (event: DesktopTerminalEvent) => void,
) {
  const { listen } = await import(/* @vite-ignore */ '@tauri-apps/api/event')
  return listen<DesktopTerminalEvent>('desktop-terminal-event', (event) => {
    handler(event.payload)
  })
}
