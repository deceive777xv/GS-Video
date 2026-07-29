use serde::Deserialize;
use thiserror::Error;

pub const API_VERSION: &str = "v1";
pub const MAX_HANDSHAKE_BYTES: usize = 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StartupHandshake {
    pub port: u16,
    pub api_version: String,
    pub pid: u32,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum HandshakeError {
    #[error("startup handshake is empty or too large")]
    InvalidLength,
    #[error("startup handshake is not valid UTF-8 JSON")]
    InvalidJson,
    #[error("startup handshake returned an unsupported API version")]
    UnsupportedVersion,
    #[error("startup handshake returned the wrong process ID")]
    WrongProcess,
    #[error("startup handshake returned an invalid port")]
    InvalidPort,
}

pub fn parse_handshake(
    bytes: &[u8],
    expected_pid: u32,
) -> Result<StartupHandshake, HandshakeError> {
    if bytes.is_empty() || bytes.len() > MAX_HANDSHAKE_BYTES {
        return Err(HandshakeError::InvalidLength);
    }
    let line = std::str::from_utf8(bytes).map_err(|_| HandshakeError::InvalidJson)?;
    if line
        .chars()
        .any(|character| character.is_control() && !matches!(character, '\r' | '\n' | '\t'))
    {
        return Err(HandshakeError::InvalidJson);
    }
    let handshake: StartupHandshake =
        serde_json::from_str(line.trim()).map_err(|_| HandshakeError::InvalidJson)?;
    if handshake.port == 0 {
        return Err(HandshakeError::InvalidPort);
    }
    if handshake.api_version != API_VERSION {
        return Err(HandshakeError::UnsupportedVersion);
    }
    if handshake.pid != expected_pid {
        return Err(HandshakeError::WrongProcess);
    }
    Ok(handshake)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strict_matching_handshake() {
        let parsed = parse_handshake(br#"{"port":49152,"apiVersion":"v1","pid":42}"#, 42).unwrap();
        assert_eq!(parsed.port, 49152);
    }

    #[test]
    fn rejects_unknown_fields_version_pid_and_oversized_lines() {
        assert_eq!(
            parse_handshake(
                br#"{"port":49152,"apiVersion":"v1","pid":42,"token":"leak"}"#,
                42,
            ),
            Err(HandshakeError::InvalidJson)
        );
        assert_eq!(
            parse_handshake(br#"{"port":49152,"apiVersion":"v2","pid":42}"#, 42),
            Err(HandshakeError::UnsupportedVersion)
        );
        assert_eq!(
            parse_handshake(br#"{"port":49152,"apiVersion":"v1","pid":41}"#, 42),
            Err(HandshakeError::WrongProcess)
        );
        assert_eq!(
            parse_handshake(&vec![b'x'; MAX_HANDSHAKE_BYTES + 1], 42),
            Err(HandshakeError::InvalidLength)
        );
    }
}
