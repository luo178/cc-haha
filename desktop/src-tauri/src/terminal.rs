use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Context, Result};
use portable_pty::{native_pty_system, CommandBuilder, MasterPty, PtySize};
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

const TERMINAL_EVENT: &str = "desktop-terminal-event";

pub struct TerminalSession {
    master: Box<dyn MasterPty + Send>,
    writer: Box<dyn Write + Send>,
    child: Box<dyn portable_pty::Child + Send>,
    cleanup_paths: Vec<PathBuf>,
}

#[derive(Default)]
pub struct TerminalState(pub Mutex<HashMap<String, TerminalSession>>);

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalStartResponse {
    pub session_id: String,
    pub shell: String,
    pub cwd: String,
    pub explicit_command_name: String,
    pub docs_command_name: String,
}

#[derive(Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TerminalEventPayload {
    Output { session_id: String, data: String },
    Exit { session_id: String, exit_code: Option<i32> },
}

#[tauri::command]
pub fn terminal_start_session(
    app: AppHandle,
    state: State<'_, TerminalState>,
    cwd: Option<String>,
) -> Result<TerminalStartResponse, String> {
    start_session(&app, &state, cwd).map_err(|err| err.to_string())
}

#[tauri::command]
pub fn terminal_write(
    state: State<'_, TerminalState>,
    session_id: String,
    data: String,
) -> Result<(), String> {
    let mut guard = state
        .0
        .lock()
        .map_err(|_| "terminal state is unavailable".to_string())?;
    let session = guard
        .get_mut(&session_id)
        .ok_or_else(|| format!("terminal session not found: {session_id}"))?;

    session
        .writer
        .write_all(data.as_bytes())
        .and_then(|_| session.writer.flush())
        .map_err(|err| format!("write terminal input: {err}"))
}

#[tauri::command]
pub fn terminal_resize(
    state: State<'_, TerminalState>,
    session_id: String,
    cols: u16,
    rows: u16,
) -> Result<(), String> {
    if cols == 0 || rows == 0 {
        return Ok(());
    }

    let mut guard = state
        .0
        .lock()
        .map_err(|_| "terminal state is unavailable".to_string())?;
    let session = guard
        .get_mut(&session_id)
        .ok_or_else(|| format!("terminal session not found: {session_id}"))?;

    session
        .master
        .resize(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|err| format!("resize terminal: {err}"))
}

#[tauri::command]
pub fn terminal_close(
    state: State<'_, TerminalState>,
    session_id: String,
) -> Result<(), String> {
    let session = {
        let mut guard = state
            .0
            .lock()
            .map_err(|_| "terminal state is unavailable".to_string())?;
        guard.remove(&session_id)
    };

    if let Some(session) = session {
        thread::spawn(move || terminate_terminal_session(session));
    }

    Ok(())
}

pub fn stop_all_sessions(state: &State<'_, TerminalState>) {
    let sessions = if let Ok(mut guard) = state.0.lock() {
        guard.drain().map(|(_, session)| session).collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    for session in sessions {
        terminate_terminal_session(session);
    }
}

fn cleanup_terminal_session(session: &mut TerminalSession) {
    for path in session.cleanup_paths.drain(..) {
        let _ = fs::remove_file(path);
    }
}

fn terminate_terminal_session(mut session: TerminalSession) {
    let _ = session.child.kill();
    let _ = session.child.wait();
    cleanup_terminal_session(&mut session);
}

fn start_session(
    app: &AppHandle,
    state: &State<'_, TerminalState>,
    cwd: Option<String>,
) -> Result<TerminalStartResponse> {
    let requested_cwd = resolve_terminal_cwd(cwd)?;
    let session_id = build_terminal_session_id();
    let cli_binary = resolve_sidecar_binary_path(app)?;
    let app_root = resolve_app_root()?;

    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize {
            rows: 24,
            cols: 100,
            pixel_width: 0,
            pixel_height: 0,
        })
        .context("open pty")?;

    let shell_bootstrap = build_shell_bootstrap(
        &session_id,
        &requested_cwd,
        &cli_binary,
        &app_root,
    )?;

    let mut cmd = CommandBuilder::new(&shell_bootstrap.program);
    cmd.cwd(&requested_cwd);
    for arg in &shell_bootstrap.args {
        cmd.arg(arg);
    }
    for (key, value) in &shell_bootstrap.env {
        cmd.env(key, value);
    }

    let reader = pair
        .master
        .try_clone_reader()
        .context("clone pty reader")?;
    let writer = pair.master.take_writer().context("take pty writer")?;
    let child = pair
        .slave
        .spawn_command(cmd)
        .context("spawn shell in pty")?;

    let output_session_id = session_id.clone();
    let output_app = app.clone();
    thread::spawn(move || stream_terminal_output(output_app, output_session_id, reader));

    let mut guard = state
        .0
        .lock()
        .map_err(|_| anyhow!("terminal state is unavailable"))?;
    guard.insert(
        session_id.clone(),
        TerminalSession {
            master: pair.master,
            writer,
            child,
            cleanup_paths: shell_bootstrap.cleanup_paths,
        },
    );

    Ok(TerminalStartResponse {
        session_id,
        shell: shell_bootstrap.display_name,
        cwd: requested_cwd.to_string_lossy().to_string(),
        explicit_command_name: "claude-haha".to_string(),
        docs_command_name: "claude".to_string(),
    })
}

fn stream_terminal_output(
    app: AppHandle,
    session_id: String,
    mut reader: Box<dyn Read + Send>,
) {
    let mut buffer = [0_u8; 8192];

    loop {
        match reader.read(&mut buffer) {
            Ok(0) => {
                let _ = app.emit(
                    TERMINAL_EVENT,
                    TerminalEventPayload::Exit {
                        session_id: session_id.clone(),
                        exit_code: None,
                    },
                );
                break;
            }
            Ok(read_bytes) => {
                let data = String::from_utf8_lossy(&buffer[..read_bytes]).to_string();
                let _ = app.emit(
                    TERMINAL_EVENT,
                    TerminalEventPayload::Output {
                        session_id: session_id.clone(),
                        data,
                    },
                );
            }
            Err(_) => {
                let _ = app.emit(
                    TERMINAL_EVENT,
                    TerminalEventPayload::Exit {
                        session_id: session_id.clone(),
                        exit_code: None,
                    },
                );
                break;
            }
        }
    }
}

struct ShellBootstrap {
    program: String,
    args: Vec<String>,
    env: Vec<(String, String)>,
    cleanup_paths: Vec<PathBuf>,
    display_name: String,
}

fn build_shell_bootstrap(
    session_id: &str,
    cwd: &Path,
    cli_binary: &Path,
    app_root: &Path,
) -> Result<ShellBootstrap> {
    #[cfg(target_os = "windows")]
    {
        let bootstrap = build_powershell_bootstrap(cwd, cli_binary, app_root);
        return Ok(ShellBootstrap {
            program: "powershell.exe".to_string(),
            args: vec![
                "-NoLogo".to_string(),
                "-NoExit".to_string(),
                "-NoProfile".to_string(),
                "-Command".to_string(),
                bootstrap,
            ],
            env: vec![
                ("TERM".to_string(), "xterm-256color".to_string()),
                ("CLAUDE_APP_ROOT".to_string(), app_root.to_string_lossy().to_string()),
            ],
            cleanup_paths: Vec::new(),
            display_name: "PowerShell".to_string(),
        });
    }

    #[cfg(not(target_os = "windows"))]
    {
        let shell_path = resolve_unix_shell()?;
        let rc_path = write_bash_rc_file(session_id, cwd, cli_binary, app_root)?;
        return Ok(ShellBootstrap {
            program: shell_path.to_string_lossy().to_string(),
            args: vec![
                "--noprofile".to_string(),
                "--rcfile".to_string(),
                rc_path.to_string_lossy().to_string(),
                "-i".to_string(),
            ],
            env: vec![
                ("TERM".to_string(), "xterm-256color".to_string()),
                ("COLORTERM".to_string(), "truecolor".to_string()),
                ("CLAUDE_APP_ROOT".to_string(), app_root.to_string_lossy().to_string()),
            ],
            cleanup_paths: vec![rc_path],
            display_name: "bash".to_string(),
        })
    }
}

#[cfg(not(target_os = "windows"))]
fn resolve_unix_shell() -> Result<PathBuf> {
    for candidate in ["/bin/bash", "/usr/bin/bash"] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Ok(path);
        }
    }

    Err(anyhow!("bash is not available on this machine"))
}

#[cfg(not(target_os = "windows"))]
fn write_bash_rc_file(
    session_id: &str,
    cwd: &Path,
    cli_binary: &Path,
    app_root: &Path,
) -> Result<PathBuf> {
    let rc_path = std::env::temp_dir().join(format!("cc-haha-terminal-{session_id}.bashrc"));
    let home_rc = std::env::var("HOME")
        .ok()
        .map(PathBuf::from)
        .map(|home| home.join(".bashrc"));

    let source_user_rc = if let Some(path) = home_rc.filter(|candidate| candidate.exists()) {
        format!("source {} >/dev/null 2>&1 || true\n", quote_for_bash(&path.to_string_lossy()))
    } else {
        String::new()
    };

    let cli_path = quote_for_bash(&cli_binary.to_string_lossy());
    let app_root_path = quote_for_bash(&app_root.to_string_lossy());
    let cwd_path = quote_for_bash(&cwd.to_string_lossy());

    let rc_content = format!(
        "{source_user_rc}export CLAUDE_APP_ROOT={app_root_path}\n\
export PS1='\\w $ '\n\
function claude-haha() {{\n  {cli_path} cli --app-root \"$CLAUDE_APP_ROOT\" \"$@\"\n}}\n\
function claude() {{\n  claude-haha \"$@\"\n}}\n\
cd {cwd_path}\n"
    );

    fs::write(&rc_path, rc_content).context("write terminal rc file")?;
    Ok(rc_path)
}

#[cfg(target_os = "windows")]
fn build_powershell_bootstrap(cwd: &Path, cli_binary: &Path, app_root: &Path) -> String {
    let cli = quote_for_powershell(&cli_binary.to_string_lossy());
    let root = quote_for_powershell(&app_root.to_string_lossy());
    let cwd_value = quote_for_powershell(&cwd.to_string_lossy());

    format!(
        "$env:CLAUDE_APP_ROOT = '{root}'; \
function global:claude-haha {{ & '{cli}' cli --app-root $env:CLAUDE_APP_ROOT @Args }}; \
function global:claude {{ claude-haha @Args }}; \
Set-Location -LiteralPath '{cwd_value}'"
    )
}

fn resolve_terminal_cwd(requested: Option<String>) -> Result<PathBuf> {
    let path = requested
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(default_home_dir);

    if !path.exists() {
        return Err(anyhow!(
            "terminal working directory does not exist: {}",
            path.to_string_lossy()
        ));
    }

    if !path.is_dir() {
        return Err(anyhow!(
            "terminal working directory is not a directory: {}",
            path.to_string_lossy()
        ));
    }

    Ok(path)
}

fn default_home_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("USERPROFILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("C:\\"))
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/"))
    }
}

fn resolve_app_root() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("resolve current exe path")?;
    let dir = exe
        .parent()
        .ok_or_else(|| anyhow!("current exe has no parent dir"))?;
    Ok(dir.to_path_buf())
}

fn resolve_sidecar_binary_path(_app: &AppHandle) -> Result<PathBuf> {
    let current_exe = std::env::current_exe().context("resolve current exe path")?;
    let current_dir = current_exe
        .parent()
        .ok_or_else(|| anyhow!("current exe has no parent dir"))?;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let binaries_dir = manifest_dir.join("binaries");

    for candidate in sidecar_name_candidates() {
        let sibling = current_dir.join(&candidate);
        if sibling.exists() {
            return Ok(sibling);
        }

        let dev_binary = binaries_dir.join(&candidate);
        if dev_binary.exists() {
            return Ok(dev_binary);
        }
    }

    Err(anyhow!(
        "could not locate bundled sidecar near {} or {}",
        current_dir.to_string_lossy(),
        binaries_dir.to_string_lossy()
    ))
}

fn sidecar_name_candidates() -> Vec<String> {
    let mut candidates = vec!["claude-sidecar".to_string()];
    let triple_name = format!("claude-sidecar-{}", current_target_triple());
    candidates.push(triple_name);

    #[cfg(target_os = "windows")]
    {
        let with_exe = candidates
            .iter()
            .map(|name| format!("{name}.exe"))
            .collect::<Vec<_>>();
        candidates.extend(with_exe);
    }

    candidates
}

fn current_target_triple() -> &'static str {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "aarch64-apple-darwin"
    }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "x86_64-apple-darwin"
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "x86_64-pc-windows-msvc"
    }

    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        "aarch64-pc-windows-msvc"
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "x86_64-unknown-linux-gnu"
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "aarch64-unknown-linux-gnu"
    }
}

fn build_terminal_session_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    format!("terminal-{millis}")
}

fn quote_for_bash(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(target_os = "windows")]
fn quote_for_powershell(value: &str) -> String {
    value.replace('\'', "''")
}
