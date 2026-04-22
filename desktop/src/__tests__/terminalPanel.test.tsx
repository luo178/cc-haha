import { beforeEach, describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import '@testing-library/jest-dom'

import { TerminalPanel } from '../components/settings/TerminalPanel'
import { useSettingsStore } from '../stores/settingsStore'

describe('TerminalPanel', () => {
  beforeEach(() => {
    useSettingsStore.setState({ locale: 'en' })
  })

  it('shows bundled CLI guidance and browser-runtime fallback text', () => {
    render(<TerminalPanel />)

    expect(screen.getByText('Setup Terminal')).toBeInTheDocument()
    expect(
      screen.getByText(/if you already have the official Claude Code installed/i),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/available only inside the packaged Tauri desktop runtime/i),
    ).toBeInTheDocument()
    expect(screen.getByText('Copy `claude-haha`')).toBeInTheDocument()
    expect(screen.queryByText('Working directory')).not.toBeInTheDocument()
    expect(screen.queryByText('Restart shell')).not.toBeInTheDocument()
  })
})
