use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use serde::Serialize;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout, Instant};
use zeroize::Zeroize;

use crate::backend::BackendCommand;
use crate::handshake::{parse_handshake, HandshakeError, MAX_HANDSHAKE_BYTES};

const HEALTH_RETRY_DELAY: Duration = Duration::from_millis(100);
const HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_millis(750);
const SHUTDOWN_HTTP_TIMEOUT: Duration = Duration::from_secs(2);
const SHUTDOWN_PROCESS_TIMEOUT: Duration = Duration::from_secs(7);
const STDERR_TAIL_BYTES: usize = 8192;

#[derive(Clone, Debug, Serialize)]
pub struct SessionConfig {
    pub origin: String,
    pub token: String,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("failed to start the local service")]
    Spawn(#[source] std::io::Error),
    #[error("the local service did not provide a startup handshake")]
    MissingHandshake,
    #[error(transparent)]
    Handshake(#[from] HandshakeError),
    #[error("the local service exited during startup")]
    ExitedDuringStartup,
    #[error("the local service did not become healthy before the startup deadline")]
    StartupTimeout,
    #[error("failed to build the local health client")]
    HttpClient(#[source] reqwest::Error),
    #[error("failed while communicating with the local service")]
    Io(#[source] std::io::Error),
}

pub struct RunningBackend {
    child: Child,
    client: reqwest::Client,
    origin: String,
    token: String,
    stdout_task: JoinHandle<()>,
    stderr_task: JoinHandle<Vec<u8>>,
}

impl RunningBackend {
    pub async fn start(
        command: BackendCommand,
        total_timeout: Duration,
    ) -> Result<Self, BackendError> {
        match timeout(total_timeout, Self::start_inner(command, total_timeout)).await {
            Ok(result) => result,
            Err(_) => Err(BackendError::StartupTimeout),
        }
    }

    async fn start_inner(
        command: BackendCommand,
        total_timeout: Duration,
    ) -> Result<Self, BackendError> {
        let mut token_bytes = [0_u8; 32];
        getrandom::fill(&mut token_bytes).map_err(|error| {
            BackendError::Io(std::io::Error::other(format!(
                "secure token generation failed: {error}"
            )))
        })?;
        let token = hex::encode(token_bytes);
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
        let expected_pid = child.id().ok_or(BackendError::ExitedDuringStartup)?;
        let mut stdin = child.stdin.take().ok_or(BackendError::MissingHandshake)?;
        stdin
            .write_all(format!("{token}\n").as_bytes())
            .await
            .map_err(BackendError::Io)?;
        stdin.shutdown().await.map_err(BackendError::Io)?;
        drop(stdin);

        let stdout = child.stdout.take().ok_or(BackendError::MissingHandshake)?;
        let stderr = child.stderr.take().ok_or(BackendError::MissingHandshake)?;
        let stderr_task = tokio::spawn(drain_stderr_tail(stderr));
        let mut stdout = BufReader::new(stdout);
        let mut line = Vec::new();
        let read = (&mut stdout)
            .take((MAX_HANDSHAKE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
            .await
            .map_err(BackendError::Io)?;
        if read == 0 || !line.ends_with(b"\n") {
            return Err(BackendError::MissingHandshake);
        }
        let handshake = parse_handshake(&line, expected_pid)?;
        let stdout_task = tokio::spawn(async move {
            let _ = tokio::io::copy(&mut stdout, &mut tokio::io::sink()).await;
        });
        let origin = format!("http://127.0.0.1:{}", handshake.port);
        let client = reqwest::Client::builder()
            .timeout(HEALTH_REQUEST_TIMEOUT)
            .build()
            .map_err(BackendError::HttpClient)?;
        let deadline = Instant::now() + total_timeout;
        loop {
            if let Some(_status) = child.try_wait().map_err(BackendError::Io)? {
                return Err(BackendError::ExitedDuringStartup);
            }
            match client
                .get(format!("{origin}/healthz"))
                .bearer_auth(&token)
                .send()
                .await
            {
                Ok(response) if response.status() == StatusCode::OK => break,
                _ if Instant::now() >= deadline => return Err(BackendError::StartupTimeout),
                _ => sleep(HEALTH_RETRY_DELAY).await,
            }
        }
        Ok(Self {
            child,
            client,
            origin,
            token,
            stdout_task,
            stderr_task,
        })
    }

    pub fn session(&self) -> SessionConfig {
        SessionConfig {
            origin: self.origin.clone(),
            token: self.token.clone(),
        }
    }

    pub fn process_id(&self) -> Option<u32> {
        self.child.id()
    }

    pub fn has_exited(&mut self) -> bool {
        !matches!(self.child.try_wait(), Ok(None))
    }

    pub async fn shutdown(mut self) -> Result<(), BackendError> {
        let pid = self.child.id();
        let request = self
            .client
            .post(format!("{}/api/v1/shutdown", self.origin))
            .bearer_auth(&self.token)
            .send();
        let _ = timeout(SHUTDOWN_HTTP_TIMEOUT, request).await;

        let graceful = timeout(SHUTDOWN_PROCESS_TIMEOUT, self.child.wait()).await;
        if !matches!(graceful, Ok(Ok(_))) {
            if let Some(pid) = pid {
                force_process_tree(pid).await;
            }
            let _ = self.child.kill().await;
            let _ = self.child.wait().await;
        }
        let _ = timeout(Duration::from_secs(1), self.stdout_task).await;
        let _ = timeout(Duration::from_secs(1), self.stderr_task).await;
        self.token.zeroize();
        Ok(())
    }
}

async fn drain_stderr_tail<R>(mut reader: R) -> Vec<u8>
where
    R: AsyncRead + Unpin,
{
    let mut tail = Vec::new();
    let mut block = [0_u8; 1024];
    loop {
        let count = match reader.read(&mut block).await {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        tail.extend_from_slice(&block[..count]);
        if tail.len() > STDERR_TAIL_BYTES {
            tail.drain(..tail.len() - STDERR_TAIL_BYTES);
        }
    }
    tail
}

#[cfg(windows)]
async fn force_process_tree(pid: u32) {
    let mut command = Command::new("taskkill.exe");
    command
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(0x0800_0000);
    let _ = timeout(Duration::from_secs(3), command.status()).await;
}

#[cfg(not(windows))]
async fn force_process_tree(_pid: u32) {}

#[derive(Default)]
pub struct LifecycleManager {
    backend: Option<RunningBackend>,
}

impl LifecycleManager {
    pub fn attach(&mut self, backend: RunningBackend) {
        self.backend = Some(backend);
    }

    pub async fn stop(&mut self) {
        if let Some(backend) = self.backend.take() {
            let _ = backend.shutdown().await;
        }
    }

    pub fn backend_exited(&mut self) -> bool {
        self.backend
            .as_mut()
            .is_some_and(RunningBackend::has_exited)
    }
}
