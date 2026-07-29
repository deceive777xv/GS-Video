use std::ffi::OsString;
use std::path::{Path, PathBuf};

use thiserror::Error;

pub const DEV_BROWSER_ORIGIN: &str = "http://127.0.0.1:1420";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendCommand {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub current_dir: PathBuf,
}

#[derive(Debug, Error)]
pub enum BackendCommandError {
    #[error("repository root is unavailable")]
    RepositoryUnavailable,
    #[error("project Python is missing at {0}")]
    PythonMissing(PathBuf),
    #[error("desktop runtime configuration is missing at {0}")]
    RuntimeMissing(PathBuf),
}

impl BackendCommand {
    pub fn development(repo_root: &Path) -> Result<Self, BackendCommandError> {
        let root = dunce::canonicalize(repo_root)
            .map_err(|_| BackendCommandError::RepositoryUnavailable)?;
        let program = root.join(".venv").join("Scripts").join("python.exe");
        if !program.is_file() {
            return Err(BackendCommandError::PythonMissing(program));
        }
        let runtime = root.join(".runtime").join("desktop-runtime.json");
        if !runtime.is_file() {
            return Err(BackendCommandError::RuntimeMissing(runtime));
        }
        Ok(Self {
            program,
            args: vec![
                "-m".into(),
                "gs_video".into(),
                "--serve".into(),
                "--runtime-config".into(),
                runtime.into_os_string(),
                "--session-token-stdin".into(),
                "--browser-origin".into(),
                DEV_BROWSER_ORIGIN.into(),
                "--startup-handshake".into(),
            ],
            current_dir: root,
        })
    }

    pub fn development_from_manifest() -> Result<Self, BackendCommandError> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root = manifest
            .ancestors()
            .nth(3)
            .ok_or(BackendCommandError::RepositoryUnavailable)?;
        Self::development(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn development_command_uses_only_fixed_non_secret_arguments() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let python = root.join(".venv").join("Scripts").join("python.exe");
        let runtime = root.join(".runtime").join("desktop-runtime.json");
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::create_dir_all(runtime.parent().unwrap()).unwrap();
        std::fs::write(&python, b"python").unwrap();
        std::fs::write(&runtime, b"{}").unwrap();

        let command = BackendCommand::development(root).unwrap();
        let args = command
            .args
            .iter()
            .map(|value| value.to_string_lossy())
            .collect::<Vec<_>>();

        assert!(args.contains(&"--session-token-stdin".into()));
        assert!(args.contains(&"--startup-handshake".into()));
        assert!(args.contains(&DEV_BROWSER_ORIGIN.into()));
        assert!(!args.iter().any(|value| value.contains("token=")));
    }

    #[cfg(windows)]
    #[test]
    fn manifest_development_command_avoids_verbatim_windows_paths() {
        let command = BackendCommand::development_from_manifest().unwrap();
        let runtime = command
            .args
            .windows(2)
            .find(|pair| pair[0] == "--runtime-config")
            .map(|pair| pair[1].to_string_lossy())
            .unwrap();

        assert!(!command.program.to_string_lossy().starts_with(r"\\?\"));
        assert!(!command.current_dir.to_string_lossy().starts_with(r"\\?\"));
        assert!(!runtime.starts_with(r"\\?\"));
    }
}
