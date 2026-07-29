use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use gs_video_desktop::backend::BackendCommand;
use gs_video_desktop::lifecycle::{BackendError, RunningBackend};

fn fake_command(args: &[&str]) -> BackendCommand {
    BackendCommand {
        program: PathBuf::from(env!("CARGO_BIN_EXE_fake_sidecar")),
        args: args.iter().map(OsString::from).collect(),
        current_dir: std::env::current_dir().unwrap(),
    }
}

#[tokio::test]
async fn starts_health_checks_and_gracefully_stops_fake_sidecar() {
    let backend = RunningBackend::start(fake_command(&[]), Duration::from_secs(5))
        .await
        .unwrap();
    let session = backend.session();

    assert!(session.origin.starts_with("http://127.0.0.1:"));
    assert_eq!(session.token.len(), 64);
    assert!(backend.process_id().is_some());

    backend.shutdown().await.unwrap();
}

#[tokio::test]
async fn startup_timeout_does_not_wait_for_hung_sidecar() {
    let started = std::time::Instant::now();
    let result = RunningBackend::start(
        fake_command(&["--no-handshake"]),
        Duration::from_millis(250),
    )
    .await;

    assert!(matches!(result, Err(BackendError::StartupTimeout)));
    assert!(started.elapsed() < Duration::from_secs(3));
}
