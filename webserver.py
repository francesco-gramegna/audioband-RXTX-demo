#code made by ai

#!/usr/bin/env python3
import http.server
import socketserver
import urllib.parse
import subprocess
import os
import socket

PORT = 8000
PYTHON_SCRIPT = "TransmitterScript.py"

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

class WavServerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text to WAV</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 10px;
            font-size: 16px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 15px 32px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        #status {
            margin-top: 20px;
            padding: 10px;
            text-align: center;
        }
        #player {
            width: 100%;
            margin-top: 20px;
        }
    </style>
</head>
<body>
<audio id="wakelockAudio" loop playsinline>
    <source src="data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=" type="audio/wav">
</audio>
    <h1>Text to WAV Generator</h1>
    <textarea id="textInput" placeholder="Enter your text here..."></textarea>
    <button onclick="generateWav()">Generate WAV</button>
    <button onclick="generateFrequencySpan()">Generate Frequency Span</button>
    <div id="status"></div>
    <audio id="player" controls style="display:none;"></audio>
    <script>
let wakelockActive = false;

async function enableWakeLock() {
    if (wakelockActive) return;

    const audio = document.getElementById("wakelockAudio");

    try {
        await audio.play();
        wakelockActive = true;
        console.log("iOS screen wake-lock ACTIVE.");
    } catch (err) {
        console.log("Wake-lock pending user interaction:", err);
    }
}

// Safari requires a user gesture first
document.addEventListener("click", enableWakeLock);
document.addEventListener("touchstart", enableWakeLock);

// Try once immediately (may fail until interaction)
enableWakeLock();


        async function generateWav() {
            const text = document.getElementById('textInput').value;
            if (!text.trim()) {
                alert('Please enter some text');
                return;
            }
            await generateAudio('/generate', 'text=' + encodeURIComponent(text));
        }

        async function generateFrequencySpan() {
            await generateAudio('/generate', 'frequency_span=true');
        }

        async function generateAudio(endpoint, bodyData) {
            const buttonList = document.querySelectorAll('button');
            const status = document.getElementById('status');
            const player = document.getElementById('player');
            
            buttonList.forEach(btn => btn.disabled = true);
            status.textContent = 'Generating...';
            player.style.display = 'none';

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: bodyData
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);

                    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: 48000
                    });
                    const arrayBuffer = await blob.arrayBuffer();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                    console.log('Audio sample rate:', audioBuffer.sampleRate);
                    console.log('Context sample rate:', audioContext.sampleRate);

                    player.src = url;
                    player.style.display = 'block';
                    status.textContent = `Success! Play your audio below (${audioBuffer.sampleRate}Hz, ${audioBuffer.numberOfChannels}ch)`;
                    status.style.color = 'green';
                } else {
                    const error = await response.text();
                    status.textContent = 'Error: ' + error;
                    status.style.color = 'red';
                }
            } catch (error) {
                status.textContent = 'Error: ' + error.message;
                status.style.color = 'red';
            } finally {
                buttonList.forEach(btn => btn.disabled = false);
            }
        }
    </script>
</body>
</html>
            '''
            self.wfile.write(html.encode())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            params = urllib.parse.parse_qs(post_data)

            # Check if this is a frequency span request
            is_frequency_span = params.get('frequency_span', [''])[0] == 'true'
            text = params.get('text', [''])[0] if not is_frequency_span else None

            try:
                cmd = ['python3', PYTHON_SCRIPT]
                if text:
                    cmd.append(text)

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                if result.returncode != 0:
                    self.send_error(500, f"Script error: {result.stderr}")
                    return

                wav_file = 'output.wav'
                if not os.path.exists(wav_file):
                    self.send_error(500, "WAV file not generated")
                    return

                with open(wav_file, 'rb') as f:
                    wav_data = f.read()

                self.send_response(200)
                self.send_header('Content-type', 'audio/wav')
                self.send_header('Content-Length', len(wav_data))
                self.end_headers()
                self.wfile.write(wav_data)
                os.remove(wav_file)

            except subprocess.TimeoutExpired:
                self.send_error(500, "Script timeout")
            except Exception as e:
                self.send_error(500, str(e))

if __name__ == '__main__':
    local_ip = get_local_ip()
    with socketserver.TCPServer(("", PORT), WavServerHandler) as httpd:
        print(f"Server running!")
        print(f"Local access: http://localhost:{PORT}")
        print(f"Phone access: http://{local_ip}:{PORT}")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

