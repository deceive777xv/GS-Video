pub mod backend;
pub mod handshake;
pub mod lifecycle;
pub mod process_tree;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use backend::{BackendCommand, DEV_BROWSER_ORIGIN};
use lifecycle::{BackendError, LifecycleManager, RunningBackend, SessionConfig, StopAction};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};
use tokio::sync::{oneshot, watch, Notify};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Default)]
struct DesktopState {
    lifecycle: Mutex<LifecycleManager>,
    lifecycle_changed: Notify,
    exit_started: AtomicBool,
}

pub fn bootstrap_script(session: &SessionConfig) -> Result<String, serde_json::Error> {
    let allowed_origin = serde_json::to_string(DEV_BROWSER_ORIGIN)?;
    Ok(format!(
        "if(window.top===window&&window.location.origin==={allowed_origin}){{window.__GS_VIDEO_SESSION__={};}}",
        serde_json::to_string(session)?
    ))
}

fn is_allowed_app_navigation(url: &tauri::Url) -> bool {
    url.origin().ascii_serialization() == DEV_BROWSER_ORIGIN
}

enum StartOutcome {
    Ready,
    Cancelled,
}

async fn start_and_show(
    app: tauri::AppHandle,
    cancel: watch::Receiver<bool>,
) -> Result<StartOutcome, String> {
    let command = match BackendCommand::development_from_manifest() {
        Ok(command) => command,
        Err(error) => {
            let state = app.state::<DesktopState>();
            state
                .lifecycle
                .lock()
                .expect("desktop lifecycle mutex poisoned")
                .finish_start_failure();
            state.lifecycle_changed.notify_one();
            return Err(format!(
                "{error}. Run `npm run prepare:desktop` from the repository root."
            ));
        }
    };
    let backend = match RunningBackend::start_cancellable(command, STARTUP_TIMEOUT, cancel).await {
        Ok(backend) => backend,
        Err(BackendError::Cancelled) => {
            let state = app.state::<DesktopState>();
            state
                .lifecycle
                .lock()
                .expect("desktop lifecycle mutex poisoned")
                .finish_start_failure();
            state.lifecycle_changed.notify_one();
            return Ok(StartOutcome::Cancelled);
        }
        Err(error) => {
            let state = app.state::<DesktopState>();
            state
                .lifecycle
                .lock()
                .expect("desktop lifecycle mutex poisoned")
                .finish_start_failure();
            state.lifecycle_changed.notify_one();
            return Err(format!(
                "{error}. Run `npm run prepare:desktop`, then retry and review the terminal diagnostics."
            ));
        }
    };
    let session = backend.session();
    let rejected = {
        let state = app.state::<DesktopState>();
        let result = state
            .lifecycle
            .lock()
            .expect("desktop lifecycle mutex poisoned")
            .accept_started(backend);
        state.lifecycle_changed.notify_one();
        result.err()
    };
    if let Some(backend) = rejected {
        let _ = backend.shutdown().await;
        let state = app.state::<DesktopState>();
        state
            .lifecycle
            .lock()
            .expect("desktop lifecycle mutex poisoned")
            .finish_stop();
        state.lifecycle_changed.notify_one();
        return Ok(StartOutcome::Cancelled);
    }
    let script = bootstrap_script(&session).map_err(|error| error.to_string())?;
    let (sender, receiver) = oneshot::channel();
    let window_app = app.clone();
    app.run_on_main_thread(move || {
        let result =
            WebviewWindowBuilder::new(&window_app, "main", WebviewUrl::App("index.html".into()))
                .title("GS Video")
                .inner_size(1280.0, 800.0)
                .min_inner_size(960.0, 640.0)
                .visible(false)
                .initialization_script(&script)
                .on_navigation(is_allowed_app_navigation)
                .build()
                .and_then(|window| {
                    window.show()?;
                    Ok(())
                })
                .map_err(|error| error.to_string());
        let _ = sender.send(result);
    })
    .map_err(|error| error.to_string())?;
    receiver
        .await
        .map_err(|_| "main window creation was interrupted".to_string())??;
    Ok(StartOutcome::Ready)
}

enum FinalAction {
    Exit(i32),
    Restart,
}

fn request_shutdown(app: &tauri::AppHandle, final_action: FinalAction) {
    let state = app.state::<DesktopState>();
    if state.exit_started.swap(true, Ordering::SeqCst) {
        return;
    }
    let exit_app = app.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            let state = exit_app.state::<DesktopState>();
            let notified = state.lifecycle_changed.notified();
            let action = state
                .lifecycle
                .lock()
                .expect("desktop lifecycle mutex poisoned")
                .request_stop();
            match action {
                StopAction::Done => break,
                StopAction::Wait => notified.await,
                StopAction::Shutdown(backend) => {
                    if let Err(error) = backend.shutdown().await {
                        eprintln!("GS Video desktop cleanup error: {error}");
                    }
                    state
                        .lifecycle
                        .lock()
                        .expect("desktop lifecycle mutex poisoned")
                        .finish_stop();
                    state.lifecycle_changed.notify_one();
                    break;
                }
            }
        }
        match final_action {
            FinalAction::Exit(code) => exit_app.exit(code),
            FinalAction::Restart => exit_app.request_restart(),
        }
    });
}

fn request_exit(app: &tauri::AppHandle, code: i32) {
    request_shutdown(app, FinalAction::Exit(code));
}

#[tauri::command]
fn restart_app(app: tauri::AppHandle) {
    request_shutdown(&app, FinalAction::Restart);
}

fn show_fatal_error(app: &tauri::AppHandle, message: String) {
    if app
        .state::<DesktopState>()
        .exit_started
        .load(Ordering::SeqCst)
    {
        return;
    }
    eprintln!("GS Video desktop error: {message}");
    let exit_app = app.clone();
    app.dialog()
        .message(message)
        .title("GS Video")
        .kind(MessageDialogKind::Error)
        .show(move |_| request_exit(&exit_app, 1));
}

async fn monitor_backend(app: tauri::AppHandle) {
    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let state = app.state::<DesktopState>();
        if state.exit_started.load(Ordering::SeqCst) {
            return;
        }
        let exited_backend = state
            .lifecycle
            .lock()
            .expect("desktop lifecycle mutex poisoned")
            .take_exited_backend();
        if let Some(backend) = exited_backend {
            if let Err(error) = backend.shutdown().await {
                eprintln!("GS Video desktop cleanup error: {error}");
            }
            state
                .lifecycle
                .lock()
                .expect("desktop lifecycle mutex poisoned")
                .finish_stop();
            state.lifecycle_changed.notify_one();
            show_fatal_error(
                &app,
                "The local Python service exited unexpectedly. Review the terminal diagnostics, then restart GS Video."
                    .to_string(),
            );
            return;
        }
    }
}

pub fn run() {
    let application = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![restart_app])
        .manage(DesktopState::default())
        .setup(|app| {
            let app = app.handle().clone();
            let cancel = app
                .state::<DesktopState>()
                .lifecycle
                .lock()
                .expect("desktop lifecycle mutex poisoned")
                .begin_start()
                .map_err(std::io::Error::other)?;
            tauri::async_runtime::spawn(async move {
                match start_and_show(app.clone(), cancel).await {
                    Ok(StartOutcome::Ready) => monitor_backend(app).await,
                    Ok(StartOutcome::Cancelled) => {}
                    Err(error) => show_fatal_error(&app, error),
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                request_exit(window.app_handle(), 0);
            }
        })
        .build(tauri::generate_context!())
        .expect("failed to build GS Video desktop host");

    application.run(|app, event| {
        if let RunEvent::ExitRequested { api, .. } = event {
            let state = app.state::<DesktopState>();
            if !state.exit_started.load(Ordering::SeqCst) {
                api.prevent_exit();
                request_exit(app, 0);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_script_uses_json_serialization() {
        let script = bootstrap_script(&SessionConfig {
            origin: "http://127.0.0.1:49152".to_string(),
            token: "quote-\"-newline-\n".to_string(),
        })
        .unwrap();
        assert!(script.starts_with(
            "if(window.top===window&&window.location.origin===\"http://127.0.0.1:1420\")"
        ));
        assert!(script.contains("quote-\\\"-newline-\\n"));
        assert!(!script.contains("quote-\"-newline-\n"));
    }

    #[test]
    fn navigation_is_limited_to_the_fixed_application_origin() {
        assert!(is_allowed_app_navigation(
            &"http://127.0.0.1:1420/projects/current".parse().unwrap()
        ));
        assert!(!is_allowed_app_navigation(
            &"http://127.0.0.1:1421/".parse().unwrap()
        ));
        assert!(!is_allowed_app_navigation(
            &"https://example.com/".parse().unwrap()
        ));
    }
}
