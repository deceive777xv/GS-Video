use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use gs_video_desktop::backend::BackendCommand;
use gs_video_desktop::lifecycle::{BackendError, RunningBackend, StartupPhase};
use tokio::sync::watch;

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
async fn starts_a_sidecar_through_a_pid_redirector() {
    let backend =
        RunningBackend::start(fake_command(&["--redirect-child"]), Duration::from_secs(5))
            .await
            .unwrap();

    backend.shutdown().await.unwrap();
}

#[test]
fn loopback_health_ignores_process_proxy_environment() {
    const PROBE_CHILD: &str = "GS_VIDEO_PROXY_PROBE_CHILD";

    if std::env::var_os(PROBE_CHILD).is_some() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let backend = RunningBackend::start(fake_command(&[]), Duration::from_secs(2))
                .await
                .unwrap();
            backend.shutdown().await.unwrap();
        });
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "loopback_health_ignores_process_proxy_environment",
            "--nocapture",
        ])
        .env(PROBE_CHILD, "1")
        .env("HTTP_PROXY", "http://127.0.0.1:9")
        .env("HTTPS_PROXY", "http://127.0.0.1:9")
        .env("ALL_PROXY", "http://127.0.0.1:9")
        .env_remove("NO_PROXY")
        .env_remove("no_proxy")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "proxy-isolation probe failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[tokio::test]
async fn startup_timeout_does_not_wait_for_hung_sidecar() {
    let started = std::time::Instant::now();
    let result = RunningBackend::start(
        fake_command(&["--no-handshake"]),
        Duration::from_millis(250),
    )
    .await;

    assert!(matches!(result, Err(BackendError::StartupTimeout { .. })));
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[tokio::test]
async fn accepts_a_bounded_blank_line_before_the_handshake() {
    let backend = RunningBackend::start(
        fake_command(&["--blank-before-handshake"]),
        Duration::from_secs(5),
    )
    .await
    .unwrap();

    backend.shutdown().await.unwrap();
}

#[tokio::test]
async fn distinguishes_health_timeout_from_handshake_timeout() {
    let result =
        RunningBackend::start(fake_command(&["--unhealthy"]), Duration::from_millis(350)).await;

    assert!(matches!(
        result,
        Err(BackendError::StartupTimeout {
            phase: StartupPhase::Health,
            ..
        })
    ));
}

#[tokio::test]
async fn reports_post_handshake_exit_and_redacts_the_token() {
    let result = RunningBackend::start(
        fake_command(&["--stderr-token-and-exit"]),
        Duration::from_secs(2),
    )
    .await;
    let error = match result {
        Err(error) => error.to_string(),
        Ok(backend) => {
            backend.shutdown().await.unwrap();
            panic!("sidecar unexpectedly became ready");
        }
    };

    assert!(error.contains("exited during health check"));
    assert!(error.contains("secret-from-stdin=[REDACTED]"));
    assert!(!error
        .split(|character: char| !character.is_ascii_hexdigit())
        .any(|value| value.len() == 64));
}

#[tokio::test]
async fn startup_can_be_cancelled_without_waiting_for_the_deadline() {
    let (cancel, receiver) = watch::channel(false);
    let startup = tokio::spawn(RunningBackend::start_cancellable(
        fake_command(&["--no-handshake"]),
        Duration::from_secs(20),
        receiver,
    ));
    tokio::time::sleep(Duration::from_millis(100)).await;

    cancel.send(true).unwrap();
    let result = tokio::time::timeout(Duration::from_secs(3), startup)
        .await
        .unwrap()
        .unwrap();

    assert!(matches!(result, Err(BackendError::Cancelled)));
}

#[cfg(windows)]
#[tokio::test]
async fn startup_failure_reclaims_a_spawned_grandchild() {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    fn process_exists(pid: u32) -> bool {
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if handle.is_null() {
            return false;
        }
        unsafe {
            CloseHandle(handle);
        }
        true
    }

    let temp = tempfile::tempdir().unwrap();
    let pid_path = temp.path().join("grandchild.pid");
    let command = BackendCommand {
        program: PathBuf::from(env!("CARGO_BIN_EXE_fake_sidecar")),
        args: vec![
            "--spawn-grandchild".into(),
            pid_path.clone().into_os_string(),
            "--no-handshake".into(),
        ],
        current_dir: std::env::current_dir().unwrap(),
    };

    let result = RunningBackend::start(command, Duration::from_millis(500)).await;
    assert!(matches!(
        result,
        Err(BackendError::StartupTimeout {
            phase: StartupPhase::Handshake,
            ..
        })
    ));
    let pid = std::fs::read_to_string(&pid_path)
        .unwrap()
        .parse::<u32>()
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while process_exists(pid) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(25));
    }
    assert!(
        !process_exists(pid),
        "grandchild process {pid} survived cleanup"
    );
}
