pub mod backend;
pub mod handshake;
pub mod lifecycle;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use backend::BackendCommand;
use lifecycle::{LifecycleManager, RunningBackend, SessionConfig};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};
use tokio::sync::{oneshot, Mutex};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Default)]
struct DesktopState {
    lifecycle: Mutex<LifecycleManager>,
    exit_started: AtomicBool,
}

pub fn bootstrap_script(session: &SessionConfig) -> Result<String, serde_json::Error> {
    Ok(format!(
        "window.__GS_VIDEO_SESSION__={};",
        serde_json::to_string(session)?
    ))
}

async fn start_and_show(app: tauri::AppHandle) -> Result<(), String> {
    let command = BackendCommand::development_from_manifest().map_err(|error| error.to_string())?;
    let backend = RunningBackend::start(command, STARTUP_TIMEOUT)
        .await
        .map_err(|error| error.to_string())?;
    let session = backend.session();
    {
        let state = app.state::<DesktopState>();
        state.lifecycle.lock().await.attach(backend);
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
        .map_err(|_| "main window creation was interrupted".to_string())?
}

fn request_exit(app: &tauri::AppHandle, code: i32) {
    let state = app.state::<DesktopState>();
    if state.exit_started.swap(true, Ordering::SeqCst) {
        return;
    }
    let exit_app = app.clone();
    tauri::async_runtime::spawn(async move {
        {
            let state = exit_app.state::<DesktopState>();
            state.lifecycle.lock().await.stop().await;
        }
        exit_app.exit(code);
    });
}

fn show_fatal_error(app: &tauri::AppHandle, message: String) {
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
        let exited = state.lifecycle.lock().await.backend_exited();
        if exited {
            show_fatal_error(
                &app,
                "The local Python service exited unexpectedly. Check the project logs for details."
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
        .manage(DesktopState::default())
        .setup(|app| {
            let app = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                match start_and_show(app.clone()).await {
                    Ok(()) => monitor_backend(app).await,
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
        assert!(script.starts_with("window.__GS_VIDEO_SESSION__={"));
        assert!(script.contains("quote-\\\"-newline-\\n"));
        assert!(!script.contains("quote-\"-newline-\n"));
    }
}
