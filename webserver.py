#code made by ai
#!/usr/bin/env python3
import http.server
import socketserver
import urllib.parse
import subprocess
import os
import socket

PORT = 8000

# Replace this with your actual Python script path and command
PYTHON_SCRIPT = "TransmitterScript.py"

def get_local_ip():
    """Get the local IP address of this machine"""
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
    <h1>Text to WAV Generator</h1>
    <textarea id="textInput" placeholder="Enter your text here..."></textarea>
    <button onclick="generateWav()">Generate WAV</button>
    <div id="status"></div>
    <audio id="player" controls style="display:none;"></audio>

    <script>
        async function generateWav() {
            const text = document.getElementById('textInput').value;
            if (!text.trim()) {
                alert('Please enter some text');
                return;
            }

            const button = document.querySelector('button');
            const status = document.getElementById('status');
            const player = document.getElementById('player');
            
            button.disabled = true;
            status.textContent = 'Generating...';
            player.style.display = 'none';

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: 'text=' + encodeURIComponent(text)
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    
                    // Use Web Audio API for bit-perfect playback
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: 48000  // Force 48kHz sample rate
                    });
                    
                    const arrayBuffer = await blob.arrayBuffer();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    
                    // Verify the sample rate
                    console.log('Audio sample rate:', audioBuffer.sampleRate);
                    console.log('Context sample rate:', audioContext.sampleRate);
                    
                    // Also set for regular player as fallback
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
                button.disabled = false;
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
            text = params.get('text', [''])[0]

            if not text:
                self.send_error(400, "No text provided")
                return

            try:
                # Run your Python script with the text as an argument
                # The script should output a WAV file named 'output.wav'
                result = subprocess.run(
                    ['python3', PYTHON_SCRIPT, text],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    self.send_error(500, f"Script error: {result.stderr}")
                    return

                # Read the generated WAV file
                wav_file = 'output.wav'  # Adjust this to match your script's output
                if not os.path.exists(wav_file):
                    self.send_error(500, "WAV file not generated")
                    return

                with open(wav_file, 'rb') as f:
                    wav_data = f.read()

                # Send the WAV file back
                self.send_response(200)
                self.send_header('Content-type', 'audio/wav')
                self.send_header('Content-Length', len(wav_data))
                self.end_headers()
                self.wfile.write(wav_data)

                # Clean up
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
