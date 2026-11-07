import subprocess, time, socket, http.client, os, shutil, json, base64

def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p
GW = free_port()

shutil.rmtree('data/_sb', ignore_errors=True)
os.makedirs('data/_sb/data/default-user/characters', exist_ok=True)
os.makedirs('data/_sb/data/default-user/chats', exist_ok=True)
os.makedirs('data/_sb/data/default-user/thumbnails/avatar', exist_ok=True)

# Build a REAL session cookie the way ST does: session=<base64(json)>.sig
secret = 'testsecret'
session = {'handle': 'default-user', 'csrfToken': 'tok123'}
import hashlib, hmac, binascii
# keygrip-style HMAC-SHA1 base64url
def keygrip_sign(data: str, secret: str) -> str:
    h = hmac.new(secret.encode(), data.encode(), hashlib.sha1).digest()
    return base64.urlsafe_b64encode(h).decode().rstrip('=')
sess_b64 = base64.b64encode(json.dumps(session).encode()).decode()
sig = keygrip_sign(f'session={sess_b64}', secret)
# cookies: session and session.sig
cookie = f'session={sess_b64}; session.sig={sig}'
print('cookie:', cookie[:80], '...')

# a real character file
card = json.dumps({'spec': 'chara_card_v2', 'spec_version': '2.0', 'name': 'test',
                   'description': 'd', 'data': {'name': 'test', 'description': 'd'}})
b64 = base64.b64encode(card.encode()).decode()
import struct, zlib
def chunk(t, d):
    return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 6, 0, 0, 0)) + chunk(b'tEXt', b'chara\x00' + b64.encode()) + chunk(b'IEND', b'')
open('data/_sb/data/default-user/characters/test.png', 'wb').write(png)

cfg = f'''dataRoot: "{os.path.abspath('data/_sb/data').replace(chr(92), '/')}"
node:
  port: 1
gateway:
  listen_addr: "127.0.0.1:{GW}"
  node_addr: "127.0.0.1:1"
'''
open('_edit_cfg.yaml', 'w').write(cfg)
gw = subprocess.Popen([r'target/debug/st-gateway.exe', '--config', '_edit_cfg.yaml'],
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(1.5)

boundary = '----WebKitFormBoundaryMnu7FQjIMbsvNyAs'
def part(name, value):
    return (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode()
body = part('ch_name', 'test') + part('avatar_url', 'test.png') + part('description', 'hello updated') + f'--{boundary}--\r\n'.encode()

conn = http.client.HTTPConnection('127.0.0.1', GW, timeout=10)
try:
    conn.request('POST', '/api/characters/edit', body=body, headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Cookie': cookie,
        'X-CSRF-Token': 'tok123',
    })
    resp = conn.getresponse()
    print('STATUS:', resp.status, resp.read()[:300])
except Exception as e:
    print('EXC:', type(e).__name__, e)
gw.kill(); gw.wait()
out = gw.stdout.read()
print('--- gateway log tail ---')
print(out.decode('utf-8', 'replace')[-1500:])
