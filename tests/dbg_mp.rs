use st_gateway::routes::__test::parse_multipart;

#[tokio::test]
async fn dbg1() {
    let b = "----B";
    let crlf = "\r\n";
    let mut p = String::new();
    p.push_str(&format!("--{b}{crlf}"));
    p.push_str(&format!("Content-Disposition: form-data; name=\"ch_name\"{crlf}{crlf}"));
    p.push_str(&format!("test{crlf}"));
    p.push_str(&format!("--{b}--{crlf}"));
    let ct = format!("multipart/form-data; boundary={b}");
    let r = parse_multipart(&ct, Body::from(p.clone().into_bytes())).await;
    println!("fields-only: {:?}", r.as_ref().map(|m| m.fields.len()).map_err(|e| e.to_string()));
}

#[tokio::test]
async fn dbg2() {
    let b = "----B";
    let crlf = "\r\n";
    let mut p = String::new();
    p.push_str(&format!("--{b}{crlf}"));
    p.push_str(&format!("Content-Disposition: form-data; name=\"ch_name\"{crlf}{crlf}"));
    p.push_str(&format!("test{crlf}"));
    p.push_str(&format!("--{b}{crlf}"));
    p.push_str(&format!("Content-Disposition: form-data; name=\"avatar\"; filename=\"\"{crlf}"));
    p.push_str(&format!("Content-Type: application/octet-stream{crlf}{crlf}"));
    p.push_str(&format!("{crlf}--{b}--{crlf}"));
    let ct = format!("multipart/form-data; boundary={b}");
    let r = parse_multipart(&ct, Body::from(p.into_bytes())).await;
    println!("with-empty-avatar: {:?}", r.as_ref().map(|m| (m.fields.len(), m.file_bytes.is_none())).map_err(|e| e.to_string()));
}

use axum::body::Body;
