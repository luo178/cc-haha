import { useEffect, useMemo, useRef, useState } from 'react'
import '@xterm/xterm/css/xterm.css'
import { Button } from '../shared/Button'
import { useTranslation } from '../../i18n'
import { useUIStore } from '../../stores/uiStore'
import { isTauriRuntime } from '../../lib/desktopRuntime'
import {
  closeDesktopTerminal,
  listenDesktopTerminalEvents,
  resizeDesktopTerminal,
  startDesktopTerminal,
  writeDesktopTerminal,
  type DesktopTerminalStart,
} from '../../lib/settingsTerminal'

type TerminalLike = {
  clear(): void
  dispose(): void
  focus(): void
  write(data: string): void
  onData(listener: (data: string) => void): { dispose(): void }
  cols: number
  rows: number
}

type FitAddonLike = {
  fit(): void
}

function readThemeVar(name: string, fallback: string) {
  if (typeof window === 'undefined') return fallback
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim()
  return value || fallback
}

export function TerminalPanel() {
  const t = useTranslation()
  const addToast = useUIStore((s) => s.addToast)
  const [session, setSession] = useState<DesktopTerminalStart | null>(null)
  const [launchError, setLaunchError] = useState<string | null>(null)
  const [isLaunching, setIsLaunching] = useState(false)
  const [exitCode, setExitCode] = useState<number | null>(null)
  const [terminalReady, setTerminalReady] = useState(false)
  const [eventsReady, setEventsReady] = useState(false)
  const hostRef = useRef<HTMLDivElement | null>(null)
  const terminalRef = useRef<TerminalLike | null>(null)
  const fitRef = useRef<FitAddonLike | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const pendingOutputRef = useRef<string[]>([])
  const autoStartedRef = useRef(false)
  const isDesktop = isTauriRuntime()

  const focusTerminal = () => {
    terminalRef.current?.focus()
  }

  const closeSessionInBackground = (sessionId: string | null) => {
    if (!sessionId) return
    void closeDesktopTerminal(sessionId).catch((error) => {
      setLaunchError(error instanceof Error ? error.message : String(error))
    })
  }

  const flushBufferedOutput = () => {
    if (!terminalRef.current || pendingOutputRef.current.length === 0) return
    for (const chunk of pendingOutputRef.current) {
      terminalRef.current.write(chunk)
    }
    pendingOutputRef.current = []
  }

  useEffect(() => {
    if (!isDesktop || !hostRef.current || terminalRef.current) return

    let disposed = false
    let cleanup = () => {}

    void (async () => {
      const [{ Terminal }, { FitAddon }] = await Promise.all([
        import('@xterm/xterm'),
        import('@xterm/addon-fit'),
      ])
      if (disposed || !hostRef.current) return

      const terminal = new Terminal({
        cursorBlink: true,
        convertEol: true,
        fontFamily:
          '"SF Mono", "JetBrains Mono", "Fira Code", "Cascadia Mono", ui-monospace, monospace',
        fontSize: 12.5,
        lineHeight: 1.35,
        scrollback: 5000,
        theme: {
          background: readThemeVar('--color-terminal-bg', '#121212'),
          foreground: readThemeVar('--color-terminal-fg', '#d4d4d4'),
          cursor: readThemeVar('--color-brand', '#d97757'),
          cursorAccent: readThemeVar('--color-surface', '#ffffff'),
          selectionBackground: 'rgba(217, 119, 87, 0.22)',
          black: '#1f1f1f',
          red: '#ff6d67',
          green: '#7ef18a',
          yellow: '#f8c55f',
          blue: '#7aa2f7',
          magenta: '#c792ea',
          cyan: '#7fd1b9',
          white: '#d7d2d0',
          brightBlack: '#6c6763',
          brightRed: '#ff8e88',
          brightGreen: '#96f5a0',
          brightYellow: '#ffd67c',
          brightBlue: '#98b8ff',
          brightMagenta: '#ddb2ff',
          brightCyan: '#9be7d1',
          brightWhite: '#f7f2ef',
        },
      })
      const fitAddon = new FitAddon()
      terminal.loadAddon(fitAddon)
      terminal.open(hostRef.current)
      terminal.focus()
      terminalRef.current = terminal
      fitRef.current = fitAddon
      setTerminalReady(true)
      flushBufferedOutput()

      const scheduleResize = () => {
        if (!fitRef.current || !terminalRef.current || !sessionIdRef.current) return
        fitRef.current.fit()
        const cols = terminalRef.current.cols
        const rows = terminalRef.current.rows
        if (cols > 0 && rows > 0) {
          void resizeDesktopTerminal(sessionIdRef.current, cols, rows).catch(() => null)
        }
      }

      const dataDisposable = terminal.onData((data) => {
        const currentSessionId = sessionIdRef.current
        if (!currentSessionId) return
        void writeDesktopTerminal(currentSessionId, data).catch((error) => {
          setLaunchError(
            error instanceof Error ? error.message : String(error),
          )
        })
      })

      const resizeObserver =
        typeof ResizeObserver !== 'undefined'
          ? new ResizeObserver(() => scheduleResize())
          : null
      resizeObserver?.observe(hostRef.current)
      hostRef.current.addEventListener('mousedown', focusTerminal)

      setTimeout(scheduleResize, 0)

      cleanup = () => {
        dataDisposable.dispose()
        resizeObserver?.disconnect()
        hostRef.current?.removeEventListener('mousedown', focusTerminal)
        terminal.dispose()
        terminalRef.current = null
        fitRef.current = null
        setTerminalReady(false)
      }
    })()

    return () => {
      disposed = true
      cleanup()
    }
  }, [isDesktop])

  useEffect(() => {
    if (!isDesktop) return

    let cancelled = false
    let unlisten: (() => void) | undefined

    void listenDesktopTerminalEvents((event) => {
      if (cancelled || event.sessionId !== sessionIdRef.current) return
      if (event.type === 'output') {
        if (terminalRef.current) {
          terminalRef.current.write(event.data)
        } else {
          pendingOutputRef.current.push(event.data)
        }
        return
      }

      setExitCode(event.exitCode ?? null)
      if (event.exitCode !== null && event.exitCode !== undefined) {
        terminalRef.current?.write(
          `\r\n${t('settings.terminal.exitPrefix')} ${event.exitCode}\r\n`,
        )
      }
    }).then((dispose) => {
      unlisten = dispose
      setEventsReady(true)
    })

    return () => {
      cancelled = true
      setEventsReady(false)
      unlisten?.()
    }
  }, [isDesktop, t])

  const startSession = async () => {
    if (!isDesktop) return

    setIsLaunching(true)
    setLaunchError(null)
    setExitCode(null)

    const previousSessionId = sessionIdRef.current

    try {
      const started = await startDesktopTerminal()
      sessionIdRef.current = started.sessionId
      setSession(started)
      terminalRef.current?.clear()
      terminalRef.current?.focus()
      fitRef.current?.fit()
      if (terminalRef.current?.cols && terminalRef.current?.rows) {
        await resizeDesktopTerminal(
          started.sessionId,
          terminalRef.current.cols,
          terminalRef.current.rows,
        ).catch(() => null)
      }
      if (previousSessionId && previousSessionId !== started.sessionId) {
        closeSessionInBackground(previousSessionId)
      }
    } catch (error) {
      setLaunchError(error instanceof Error ? error.message : String(error))
    } finally {
      setIsLaunching(false)
    }
  }

  useEffect(() => {
    if (!isDesktop || !terminalReady || !eventsReady || autoStartedRef.current) return
    autoStartedRef.current = true
    void startSession()

    return () => {
      autoStartedRef.current = false
      const existingSessionId = sessionIdRef.current
      sessionIdRef.current = null
      if (existingSessionId) {
        closeSessionInBackground(existingSessionId)
      }
    }
  }, [eventsReady, isDesktop, terminalReady])

  const explicitCommand = session?.explicitCommandName || 'claude-haha'
  const docsCommand = session?.docsCommandName || 'claude'
  const exampleCommands = useMemo(
    () => [
      `${docsCommand} plugin install skill-creator@claude-plugins-official --scope user`,
      `${explicitCommand} mcp add docs --transport http https://example.com/mcp`,
    ],
    [docsCommand, explicitCommand],
  )

  const handleCopy = async (value: string) => {
    try {
      await navigator.clipboard.writeText(value)
      addToast({
        type: 'success',
        message: t('settings.terminal.copied'),
      })
    } catch (error) {
      addToast({
        type: 'error',
        message:
          error instanceof Error ? error.message : t('settings.terminal.copyFailed'),
      })
    }
  }

  return (
    <div className="w-full min-w-0 max-w-6xl">
      <section className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface-container-low)] p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--color-text-tertiary)]">
              {t('settings.terminal.eyebrow')}
            </div>
            <h2 className="mt-2 text-lg font-semibold text-[var(--color-text-primary)]">
              {t('settings.terminal.title')}
            </h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--color-text-secondary)]">
              {t('settings.terminal.description')}
            </p>
          </div>

          {isDesktop ? (
            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => void startSession()}
                loading={isLaunching}
              >
                {t('settings.terminal.restart')}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => terminalRef.current?.clear()}
              >
                {t('settings.terminal.clear')}
              </Button>
            </div>
          ) : null}
        </div>

        <div className="mt-4 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="text-sm font-semibold text-[var(--color-text-primary)]">
                {t('settings.terminal.docsTitle')}
              </div>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                {t('settings.terminal.docsBody', {
                  docsCommand,
                  explicitCommand,
                })}
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => void handleCopy(docsCommand)}
              >
                {t('settings.terminal.copyDocsCommand', { name: docsCommand })}
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => void handleCopy(explicitCommand)}
              >
                {t('settings.terminal.copyExplicitCommand', {
                  name: explicitCommand,
                })}
              </Button>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            {exampleCommands.map((command) => (
              <button
                key={command}
                type="button"
                onClick={() => void handleCopy(command)}
                className="rounded-full border border-[var(--color-border)] bg-[var(--color-surface-container-low)] px-3 py-1.5 font-[var(--font-mono)] text-[11px] text-[var(--color-text-secondary)] transition-colors hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text-primary)]"
              >
                {command}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="mt-6 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
        {!isDesktop ? (
          <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-container-low)] px-4 py-6 text-sm text-[var(--color-text-tertiary)]">
            {t('settings.terminal.runtimeOnly')}
          </div>
        ) : (
          <div className="min-w-0">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-[var(--color-text-primary)]">
                  {t('settings.terminal.sessionTitle')}
                </div>
                <div className="mt-1 text-xs text-[var(--color-text-tertiary)]">
                  {session
                    ? t('settings.terminal.sessionMeta', {
                        shell: session.shell,
                        cwd: session.cwd,
                      })
                    : t('settings.terminal.sessionPending')}
                </div>
              </div>
              {exitCode !== null ? (
                <span className="rounded-full border border-[var(--color-border)] px-3 py-1 text-xs text-[var(--color-text-tertiary)]">
                  {t('settings.terminal.exitBadge', { code: String(exitCode) })}
                </span>
              ) : null}
            </div>

            {launchError ? (
              <div className="mt-4 rounded-xl border border-[var(--color-error)]/25 bg-[var(--color-error)]/8 px-4 py-3 text-sm text-[var(--color-error)]">
                {launchError}
              </div>
            ) : null}

            <div className="mt-4 overflow-hidden rounded-2xl border border-[var(--color-terminal-border)] bg-[var(--color-terminal-bg)] shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
              <div className="flex items-center gap-2 border-b border-[var(--color-terminal-border)] bg-[var(--color-terminal-header)] px-4 py-2.5">
                <div className="h-2.5 w-2.5 rounded-full bg-[var(--color-terminal-danger)]" />
                <div className="h-2.5 w-2.5 rounded-full bg-[var(--color-terminal-warning)]" />
                <div className="h-2.5 w-2.5 rounded-full bg-[var(--color-terminal-accent)]" />
                <span className="ml-2 truncate font-[var(--font-mono)] text-[11px] text-[var(--color-terminal-muted)]">
                  {session?.cwd || t('settings.terminal.sessionPending')}
                </span>
              </div>
              <div
                ref={hostRef}
                className="h-[560px] min-h-[420px] w-full cursor-text px-2 py-2"
              />
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
