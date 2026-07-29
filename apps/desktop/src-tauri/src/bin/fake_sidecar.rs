use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::time::Duration;

fn main() {
    let mut arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if let Some(index) = arguments
        .iter()
        .position(|argument| argument == "--redirect-child")
    {
        arguments.remove(index);
        arguments.push("--reported-parent-pid".into());
        arguments.push(std::process::id().to_string());
        let status = Command::new(std::env::current_exe().unwrap())
            .args(arguments)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .unwrap();
        std::process::exit(status.code().unwrap_or(1));
    }
    let reported_parent_pid = arguments
        .windows(2)
        .find(|pair| pair[0] == "--reported-parent-pid")
        .and_then(|pair| pair[1].parse::<u32>().ok())
        .unwrap_or_else(std::process::id);
    if arguments
        .iter()
        .any(|argument| argument == "--grandchild-process")
    {
        std::thread::sleep(Duration::from_secs(30));
        return;
    }
    let mut token = String::new();
    std::io::stdin().lock().read_line(&mut token).unwrap();
    let token = token.trim_end_matches(['\r', '\n']).to_string();
    if let Some(index) = arguments
        .iter()
        .position(|argument| argument == "--spawn-grandchild")
    {
        let pid_path = arguments.get(index + 1).expect("grandchild PID path");
        let child = Command::new(std::env::current_exe().unwrap())
            .arg("--grandchild-process")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        std::fs::write(pid_path, child.id().to_string()).unwrap();
    }
    if arguments
        .iter()
        .any(|argument| argument == "--no-handshake")
    {
        std::thread::sleep(Duration::from_secs(30));
        return;
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    if arguments
        .iter()
        .any(|argument| argument == "--blank-before-handshake")
    {
        println!();
    }
    println!(
        "{{\"port\":{port},\"apiVersion\":\"v1\",\"pid\":{},\"parentPid\":{reported_parent_pid}}}",
        std::process::id(),
    );
    std::io::stdout().flush().unwrap();
    if arguments
        .iter()
        .any(|argument| argument == "--stderr-token-and-exit")
    {
        eprintln!("secret-from-stdin={token}");
        return;
    }
    let unhealthy = arguments.iter().any(|argument| argument == "--unhealthy");
    for incoming in listener.incoming() {
        let mut stream = incoming.unwrap();
        if handle_request(&mut stream, &token, unhealthy) {
            break;
        }
    }
}

fn handle_request(stream: &mut TcpStream, token: &str, unhealthy: bool) -> bool {
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
    } else if health && unhealthy {
        ("503 Service Unavailable", "{\"status\":\"starting\"}")
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
