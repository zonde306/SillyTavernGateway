cargo zigbuild --release --target x86_64-unknown-linux-musl
copy /Y target\x86_64-unknown-linux-musl\release\st-gateway .\st-gateway
pause