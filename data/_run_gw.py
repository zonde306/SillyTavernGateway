import subprocess, time, socket, http.client, os, shutil, json, base64, struct, zlib
import hashlib, hmac, traceback

def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p

ROOT = os.path.abspath('data/_ren')
GW = free_port()
shutil.rmtree(ROOT, ignore_errors=True)
CH = os.path.join(ROOT, 'default-user', 'characters')
for d in [CH, os.path.join(ROOT,'default-user','chats'), os.path.join(ROOT,'default-user','thumbnails','avatar')]:
    os.makedirs(d, exist_ok=True)

secret = 'testsecret'
session = {'handle': 'default-user', 'csrfToken': 'tok123'}
def kg(data):
    h = hmac.new(secret.encode(), data.encode(), hashlib.sha1).digest()
    return base64.urlsafe_b64encode(h).decode().rstrip('=')
sess_b64 = base64.b64encode(json.dumps(session).encode()).decode()
cookie = f'session={sess_b64}; session.sig={kg("session=" + sess_b64)}'

def chunk(t, d):
    return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
def make_png(name):
    card = json.dumps({'spec':'chara_card_v2','spec_version':'2.0','name':name,'description':'d',
        'data':{'name':name,'description':'d','alternate_greetings':['alt1','alt2']}})
    b64 = base64.b64encode(card.encode()).decode()
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB',1,1,8,6,0,0,0))
            + chunk(b'tEXt', b'chara\x00'+b64.encode()) + chunk(b'IDAT', zlib.compress(b'\x00\x00\x00\x00\x00'))
            + chunk(b'IEND', b''))
open(os.path.join(CH,'test.png'),'wb').write(make_png('test'))

open('_ren_cfg.yaml','w').write(f'''dataRoot: "{ROOT.replace(chr(92),"/")}"
node:
  port: 1
gateway:
  listen_addr: "127.0.0.1:{GW}"
  node_addr: "127.0.0.1:1"
''')
gw = subprocess.Popen([r'target/debug/st-gateway.exe','--config','_ren_cfg.yaml'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(1.5)

def post(path, body):
    conn = http.client.HTTPConnection('127.0.0.1', GW, timeout=20)
    try:
        conn.request('POST', path, body=body, headers={'Content-Type':'application/json',
            'Cookie':cookie,'X-CSRF-Token':'tok123'})
        r = conn.getresponse()
        return r.status, r.read()[:400]
    except Exception as e:
        return 'EXC', repr(e)

for label, path, body in [
    ('rename','/api/characters/rename',{'avatar_url':'test.png','new_name':'renamed'}),
    ('get','/api/characters/get',{'avatar_url':'renamed.png'}),
]:
    st, body = post(path, json.dumps(body))
    print(label, '->', st, body)
print('files:', sorted(os.listdir(CH)))

gw.kill(); gw.wait()
print('--- log ---')
print(gw.stdout.read().decode('utf-8','replace')[-3000:])
