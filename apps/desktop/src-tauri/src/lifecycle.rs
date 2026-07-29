use std::fmt;
use std::future::Future;
use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use serde::Serialize;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout, timeout_at, Instant};
use zeroize::{Zeroize, Zeroizing};

use crate::backend::BackendCommand;
use crate::handshake::{parse_handshake, HandshakeError, MAX_HANDSHAKE_BYTES};
use crate::process_tree::ProcessTree;

const HEALTH_RETRY_DELAY: Duration = Duration::from_millis(100);
const HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_millis(750);
const SHUTDOWN_HTTP_TIMEOUT: Duration = Duration::from_secs(2);
const SHUTDOWN_PROCESS_TIMEOUT: Duration = Duration::from_secs(7);
const STDERR_TAIL_BYTES: usize = 8192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StartupPhase {
    Handshake,
    Health,
}

impl fmt::Display for StartupPhase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handshake => formatter.write_str("startup handshake"),
            Self::Health => formatter.write_str("health check"),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionConfig {
    pub origin: String,
    pub token: String,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("failed to start the local service")]
    Spawn(#[source] std::io::Error),
    #[error("failed to own the local service process tree")]
    ProcessTree(#[source] std::io::Error),
    #[error("local service startup was cancelled")]
    Cancelled,
    #[error("the local service did not provide a startup handshake{diagnostic}")]
    MissingHandshake { diagnostic: String },
    #[error("the local service returned an invalid startup handshake: {source}{diagnostic}")]
    Handshake {
        #[source]
        source: HandshakeError,
        diagnostic: String,
    },
    #[error("the local service exited during {phase}{diagnostic}")]
    ExitedDuringStartup {
        phase: StartupPhase,
        diagnostic: String,
    },
    #[error("the local service timed out during {phase}{diagnostic}")]
    StartupTimeout {
        phase: StartupPhase,
        diagnostic: String,
    },
    #[error("failed to build the local health client{diagnostic}")]
    HttpClient {
        #[source]
        source: reqwest::Error,
        diagnostic: String,
    },
    #[error("failed while communicating with the local service{diagnostic}")]
    Io {
        #[source]
        source: std::io::Error,
        diagnostic: String,
    },
    #[error("forced local service cleanup failed")]
    Cleanup(#[source] std::io::Error),
}

enum StartupFailure {
    Cancelled,
    MissingHandshake,
    Handshake(HandshakeError),
    Exited(StartupPhase),
    Timeout(StartupPhase),
    HttpClient(reqwest::Error),
    Io(std::io::Error),
}

impl StartupFailure {
    fn into_backend(self, diagnostic: String) -> BackendError {
        match self {
            Self::Cancelled => BackendError::Cancelled,
            Self::MissingHandshake => BackendError::MissingHandshake { diagnostic },
            Self::Handshake(source) => BackendError::Handshake { source, diagnostic },
            Self::Exited(phase) => BackendError::ExitedDuringStartup { phase, diagnostic },
            Self::Timeout(phase) => BackendError::StartupTimeout { phase, diagnostic },
            Self::HttpClient(source) => BackendError::HttpClient { source, diagnostic },
            Self::Io(source) => BackendError::Io { source, diagnostic },
        }
    }
}

pub struct RunningBackend {
    child: Child,
    process_tree: ProcessTree,
    original_pid: u32,
    client: reqwest::Client,
    origin: String,
    token: Zeroizing<String>,
    stdout_task: JoinHandle<()>,
    stderr_task: JoinHandle<Vec<u8>>,
}

impl RunningBackend {
    pub async fn start(
        command: BackendCommand,
        total_timeout: Duration,
    ) -> Result<Self, BackendError> {
        let (_cancel, receiver) = watch::channel(false);
        Self::start_cancellable(command, total_timeout, receiver).await
    }

    pub async fn start_cancellable(
        command: BackendCommand,
        total_timeout: Duration,
        mut cancel: watch::Receiver<bool>,
    ) -> Result<Self, BackendError> {
        let deadline = Instant::now() + total_timeout;
        let mut token_bytes = [0_u8; 32];
        getrandom::fill(&mut token_bytes).map_err(|error| BackendError::Io {
            source: std::io::Error::other(format!("secure token generation failed: {error}")),
            diagnostic: String::new(),
        })?;
        let token = Zeroizing::new(hex::encode(token_bytes));
        token_bytes.zeroize();

        let mut process = Command::new(&command.program);
        process
            .args(&command.args)
            .current_dir(&command.current_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(windows)]
        process.creation_flags(0x0800_0000 | 0x0000_0200);
        let mut child = process.spawn().map_err(BackendError::Spawn)?;
        let expected_pid = child.id().ok_or_else(|| BackendError::Io {
            source: std::io::Error::other("local service process ID is unavailable"),
            diagnostic: String::new(),
        })?;
        let process_tree = match ProcessTree::attach(expected_pid) {
            Ok(process_tree) => process_tree,
            Err(error) => {
                force_unowned_process_tree(expected_pid, &mut child).await;
                return Err(BackendError::ProcessTree(error));
            }
        };
        let mut stdin = child.stdin.take().ok_or_else(|| BackendError::Io {
            source: std::io::Error::other("local service stdin is unavailable"),
            diagnostic: String::new(),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| BackendError::Io {
            source: std::io::Error::other("local service stdout is unavailable"),
            diagnostic: String::new(),
        })?;
        let stderr = child.stderr.take().ok_or_else(|| BackendError::Io {
            source: std::io::Error::other("local service stderr is unavailable"),
            diagnostic: String::new(),
        })?;
        let stderr_task = tokio::spawn(drain_stderr_tail(stderr, token.len()));
        let mut stdout = BufReader::new(stdout);
        let mut stdout_task = None;

        let startup_result = async {
            wait_for_phase(
                deadline,
                &mut cancel,
                StartupPhase::Handshake,
                stdin.write_all(format!("{}\n", token.as_str()).as_bytes()),
            )
            .await?
            .map_err(StartupFailure::Io)?;
            wait_for_phase(
                deadline,
                &mut cancel,
                StartupPhase::Handshake,
                stdin.shutdown(),
            )
            .await?
            .map_err(StartupFailure::Io)?;
            drop(stdin);

            let line = wait_for_phase(
                deadline,
                &mut cancel,
                StartupPhase::Handshake,
                read_handshake_line(&mut stdout),
            )
            .await??;
            let handshake =
                parse_handshake(&line, expected_pid).map_err(StartupFailure::Handshake)?;
            stdout_task = Some(tokio::spawn(async move {
                let _ = tokio::io::copy(&mut stdout, &mut tokio::io::sink()).await;
            }));
            let origin = format!("http://127.0.0.1:{}", handshake.port);
            let client = reqwest::Client::builder()
                .no_proxy()
                .timeout(HEALTH_REQUEST_TIMEOUT)
                .build()
                .map_err(StartupFailure::HttpClient)?;
            loop {
                if child.try_wait().map_err(StartupFailure::Io)?.is_some() {
                    return Err(StartupFailure::Exited(StartupPhase::Health));
                }
                let request = client
                    .get(format!("{origin}/healthz"))
                    .bearer_auth(token.as_str())
                    .send();
                match wait_for_phase(deadline, &mut cancel, StartupPhase::Health, request).await? {
                    Ok(response) if response.status() == StatusCode::OK => break,
                    _ => {
                        wait_for_phase(
                            deadline,
                            &mut cancel,
                            StartupPhase::Health,
                            sleep(HEALTH_RETRY_DELAY),
                        )
                        .await?;
                    }
                }
            }
            Ok::<_, StartupFailure>((origin, client))
        }
        .await;

        let (origin, client) = match startup_result {
            Ok(result) => result,
            Err(failure) => {
                let diagnostic = cleanup_failed_start(
                    &mut child,
                    &process_tree,
                    stdout_task.take(),
                    stderr_task,
                    token.as_str(),
                )
                .await;
                if matches!(&failure, StartupFailure::Cancelled) && !diagnostic.is_empty() {
                    eprintln!("Local service startup cancellation{diagnostic}");
                }
                return Err(failure.into_backend(diagnostic));
            }
        };
        Ok(Self {
            child,
            process_tree,
            original_pid: expected_pid,
            client,
            origin,
            token,
            stdout_task: stdout_task.expect("stdout draining starts before health checks"),
            stderr_task,
        })
    }

    pub fn session(&self) -> SessionConfig {
        SessionConfig {
            origin: self.origin.clone(),
            token: self.token.as_str().to_owned(),
        }
    }

    pub fn process_id(&self) -> Option<u32> {
        Some(self.original_pid)
    }

    pub fn has_exited(&mut self) -> bool {
        !matches!(self.child.try_wait(), Ok(None))
    }

    pub async fn shutdown(mut self) -> Result<(), BackendError> {
        let exited_before_shutdown = self.child.try_wait().ok().flatten().is_some();
        let request = self
            .client
            .post(format!("{}/api/v1/shutdown", self.origin))
            .bearer_auth(self.token.as_str())
            .send();
        let _ = timeout(SHUTDOWN_HTTP_TIMEOUT, request).await;

        let graceful = timeout(SHUTDOWN_PROCESS_TIMEOUT, self.child.wait()).await;
        let cleanup_error = if matches!(graceful, Ok(Ok(_))) {
            None
        } else {
            let cleanup_error = self.process_tree.terminate().err();
            let _ = self.child.kill().await;
            let _ = self.child.wait().await;
            cleanup_error
        };
        let _ = timeout(Duration::from_secs(1), self.stdout_task).await;
        let stderr = timeout(Duration::from_secs(1), self.stderr_task)
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_default();
        if exited_before_shutdown {
            let diagnostic = sanitized_diagnostic(&stderr, self.token.as_str());
            if !diagnostic.is_empty() {
                eprintln!("Local service exited unexpectedly{diagnostic}");
            }
        }
        if let Some(error) = cleanup_error {
            return Err(BackendError::Cleanup(error));
        }
        Ok(())
    }
}

async fn wait_for_phase<T, F>(
    deadline: Instant,
    cancel: &mut watch::Receiver<bool>,
    phase: StartupPhase,
    future: F,
) -> Result<T, StartupFailure>
where
    F: Future<Output = T>,
{
    tokio::select! {
        biased;
        _ = cancel.changed() => Err(StartupFailure::Cancelled),
        result = timeout_at(deadline, future) => {
            result.map_err(|_| StartupFailure::Timeout(phase))
        }
    }
}

async fn read_handshake_line<R>(reader: &mut R) -> Result<Vec<u8>, StartupFailure>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut total = 0_usize;
    loop {
        let mut line = Vec::new();
        let remaining = MAX_HANDSHAKE_BYTES + 1 - total;
        let read = reader
            .take(remaining as u64)
            .read_until(b'\n', &mut line)
            .await
            .map_err(StartupFailure::Io)?;
        if read == 0 {
            return Err(StartupFailure::MissingHandshake);
        }
        total += read;
        if total > MAX_HANDSHAKE_BYTES {
            return Err(StartupFailure::Handshake(HandshakeError::InvalidLength));
        }
        if !line.ends_with(b"\n") {
            return Err(StartupFailure::MissingHandshake);
        }
        if !line.iter().all(u8::is_ascii_whitespace) {
            return Ok(line);
        }
    }
}

async fn cleanup_failed_start(
    child: &mut Child,
    process_tree: &ProcessTree,
    stdout_task: Option<JoinHandle<()>>,
    stderr_task: JoinHandle<Vec<u8>>,
    token: &str,
) -> String {
    let cleanup_error = process_tree.terminate().err();
    let _ = child.kill().await;
    let _ = child.wait().await;
    if let Some(task) = stdout_task {
        let _ = timeout(Duration::from_secs(1), task).await;
    }
    let stderr = timeout(Duration::from_secs(1), stderr_task)
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or_default();
    let mut diagnostic = sanitized_diagnostic(&stderr, token);
    if let Some(error) = cleanup_error {
        diagnostic.push_str(&format!("; process-tree cleanup reported: {error}"));
    }
    diagnostic
}

fn sanitized_diagnostic(stderr: &[u8], token: &str) -> String {
    let text = String::from_utf8_lossy(stderr).replace(token, "[REDACTED]");
    let text = text
        .chars()
        .map(|character| {
            if character == '\n'
                || character == '\r'
                || character == '\t'
                || !character.is_control()
            {
                character
            } else {
                '\u{fffd}'
            }
        })
        .collect::<String>();
    let text = text.trim();
    let text = if text.chars().count() > STDERR_TAIL_BYTES {
        text.chars()
            .rev()
            .take(STDERR_TAIL_BYTES)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    } else {
        text.to_owned()
    };
    if text.is_empty() {
        String::new()
    } else {
        format!("; stderr tail: {text}")
    }
}

async fn drain_stderr_tail<R>(mut reader: R, secret_length: usize) -> Vec<u8>
where
    R: AsyncRead + Unpin,
{
    let retained_bytes = STDERR_TAIL_BYTES + secret_length.saturating_sub(1);
    let mut tail = Vec::new();
    let mut block = [0_u8; 1024];
    loop {
        let count = match reader.read(&mut block).await {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        tail.extend_from_slice(&block[..count]);
        if tail.len() > retained_bytes {
            tail.drain(..tail.len() - retained_bytes);
        }
    }
    tail
}

#[cfg(windows)]
async fn force_unowned_process_tree(pid: u32, child: &mut Child) {
    let mut command = Command::new("taskkill.exe");
    command
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(0x0800_0000);
    let _ = timeout(Duration::from_secs(3), command.status()).await;
    let _ = child.kill().await;
    let _ = child.wait().await;
}

#[cfg(not(windows))]
async fn force_unowned_process_tree(_pid: u32, child: &mut Child) {
    let _ = child.kill().await;
    let _ = child.wait().await;
}

pub enum StopAction {
    Done,
    Wait,
    Shutdown(RunningBackend),
}

enum LifecycleState {
    Idle,
    Starting { cancel: watch::Sender<bool> },
    Ready(RunningBackend),
    Stopping,
    Exited,
}

pub struct LifecycleManager {
    state: LifecycleState,
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self {
            state: LifecycleState::Idle,
        }
    }
}

impl LifecycleManager {
    pub fn begin_start(&mut self) -> Result<watch::Receiver<bool>, &'static str> {
        if !matches!(self.state, LifecycleState::Idle) {
            return Err("local service startup has already begun");
        }
        let (cancel, receiver) = watch::channel(false);
        self.state = LifecycleState::Starting { cancel };
        Ok(receiver)
    }

    pub fn accept_started(&mut self, backend: RunningBackend) -> Result<(), RunningBackend> {
        if matches!(self.state, LifecycleState::Starting { .. }) {
            self.state = LifecycleState::Ready(backend);
            Ok(())
        } else {
            Err(backend)
        }
    }

    pub fn finish_start_failure(&mut self) {
        if matches!(
            self.state,
            LifecycleState::Starting { .. } | LifecycleState::Stopping
        ) {
            self.state = LifecycleState::Exited;
        }
    }

    pub fn request_stop(&mut self) -> StopAction {
        match std::mem::replace(&mut self.state, LifecycleState::Stopping) {
            LifecycleState::Idle | LifecycleState::Exited => {
                self.state = LifecycleState::Exited;
                StopAction::Done
            }
            LifecycleState::Starting { cancel } => {
                let _ = cancel.send(true);
                StopAction::Wait
            }
            LifecycleState::Ready(backend) => StopAction::Shutdown(backend),
            LifecycleState::Stopping => StopAction::Wait,
        }
    }

    pub fn finish_stop(&mut self) {
        self.state = LifecycleState::Exited;
    }

    pub fn take_exited_backend(&mut self) -> Option<RunningBackend> {
        let exited = match &mut self.state {
            LifecycleState::Ready(backend) => backend.has_exited(),
            _ => false,
        };
        if !exited {
            return None;
        }
        match std::mem::replace(&mut self.state, LifecycleState::Stopping) {
            LifecycleState::Ready(backend) => Some(backend),
            state => {
                self.state = state;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starting_lifecycle_is_cancelled_and_can_finish_stopping() {
        let mut manager = LifecycleManager::default();
        let receiver = manager.begin_start().unwrap();

        assert!(matches!(manager.request_stop(), StopAction::Wait));
        assert!(*receiver.borrow());

        manager.finish_start_failure();
        assert!(matches!(manager.request_stop(), StopAction::Done));
    }

    #[test]
    fn duplicate_stop_requests_are_idempotent() {
        let mut manager = LifecycleManager::default();

        assert!(matches!(manager.request_stop(), StopAction::Done));
        assert!(matches!(manager.request_stop(), StopAction::Done));
    }

    #[test]
    fn diagnostics_redact_the_session_token() {
        let diagnostic = sanitized_diagnostic(b"failed token-123", "token-123");

        assert_eq!(diagnostic, "; stderr tail: failed [REDACTED]");
    }

    #[tokio::test]
    async fn diagnostics_redact_a_token_straddling_the_tail_boundary() {
        let token = "a".repeat(64);
        let mut stderr = vec![b'x'; 100];
        stderr.extend_from_slice(token.as_bytes());
        stderr.extend(std::iter::repeat(b'y').take(STDERR_TAIL_BYTES - 32));
        let (mut writer, reader) = tokio::io::duplex(stderr.len());
        let write = tokio::spawn(async move {
            writer.write_all(&stderr).await.unwrap();
        });

        let retained = drain_stderr_tail(reader, token.len()).await;
        write.await.unwrap();
        let diagnostic = sanitized_diagnostic(&retained, &token);

        assert!(diagnostic.contains("[REDACTED]"));
        assert!(!diagnostic.contains(&"a".repeat(16)));
    }
}
