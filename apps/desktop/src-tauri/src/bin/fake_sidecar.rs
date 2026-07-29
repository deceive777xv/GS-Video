use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

fn main() {
    let mut token = String::new();
    std::io::stdin().lock().read_line(&mut token).unwrap();
    let token = token.trim_end_matches(['\r', '\n']).to_string();
    if std::env::args().any(|argument| argument == "--no-handshake") {
        std::thread::sleep(Duration::from_secs(30));
        return;
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    println!(
        "{{\"port\":{port},\"apiVersion\":\"v1\",\"pid\":{}}}",
        std::process::id()
    );
    std::io::stdout().flush().unwrap();
    for incoming in listener.incoming() {
        let mut stream = incoming.unwrap();
        if handle_request(&mut stream, &token) {
            break;
        }
    }
}

fn handle_request(stream: &mut TcpStream, token: &str) -> bool {
    let mut request = Vec::new();
    let mut block = [0_u8; 1024];
    while request.len() < 16 * 1024 {
        let count = stream.read(&mut block).unwrap();
        if count == 0 {
            break;
        }
        request.extend_from_slice(&block[..count]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
    let text = String::from_utf8_lossy(&request);
    let authorized = text
        .lines()
        .any(|line| line.eq_ignore_ascii_case(&format!("Authorization: Bearer {token}")));
    let first = text.lines().next().unwrap_or_default();
    let shutdown = first.starts_with("POST /api/v1/shutdown ");
    let health = first.starts_with("GET /healthz ");
    let (status, body) = if !authorized {
        ("401 Unauthorized", "{}")
    } else if health {
        ("200 OK", "{\"status\":\"ok\"}")
    } else if shutdown {
        ("202 Accepted", "")
    } else {
        ("404 Not Found", "{}")
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    stream.flush().unwrap();
    authorized && shutdown
}
