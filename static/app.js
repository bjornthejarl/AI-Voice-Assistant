/**
 * Sarah Voice Assistant - Toggle Mode with Smart Listening
 * Click to start/stop recording, auto-stops on silence
 */

class SarahAssistant {
    constructor() {
        this.orb = document.getElementById('orb');
        this.status = document.getElementById('status');
        this.response = document.getElementById('response');
        this.hint = document.getElementById('hint');
        this.canvas = document.getElementById('waveform');
        this.ctx = this.canvas.getContext('2d');

        this.mediaRecorder = null;
        this.audioChunks = [];
        this.analyser = null;
        this.audioContext = null;
        this.isRecording = false;
        this.animationFrame = null;

        // Smart listening settings - higher thresholds to filter keyboard noise
        this.silenceThreshold = 25; // Audio level below this = silence (was 15)
        this.silenceDuration = 2500; // ms of silence before auto-stop
        this.lastSpeechTime = null;
        this.silenceCheckInterval = null;

        this.init();
    }

    init() {
        // Toggle on click
        document.body.addEventListener('click', (e) => {
            if (this.orb.classList.contains('speaking') || this.orb.classList.contains('thinking')) return;
            this.toggleRecording();
        });

        // Space bar toggle
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !e.repeat) {
                e.preventDefault();
                if (this.orb.classList.contains('speaking') || this.orb.classList.contains('thinking')) return;
                this.toggleRecording();
            }
        });

        this.idleAnimation();
    }

    toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }

    async startRecording() {
        if (this.isRecording) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);

            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (e) => this.audioChunks.push(e.data);
            this.mediaRecorder.onstop = () => this.processRecording();

            this.mediaRecorder.start();
            this.isRecording = true;
            this.lastSpeechTime = Date.now();

            document.body.classList.add('active');
            this.orb.classList.add('listening');
            this.orb.classList.remove('speaking', 'thinking');
            this.setStatus('listening');
            this.response.classList.remove('visible');
            this.hint.textContent = 'Click to stop or wait for silence';

            this.drawWaveform();
            this.startSilenceDetection();

        } catch (err) {
            console.error('Mic error:', err);
        }
    }

    stopRecording() {
        if (!this.isRecording) return;

        this.isRecording = false;
        document.body.classList.remove('active');
        this.hint.textContent = 'Click or press Space to talk';

        // Stop silence detection
        if (this.silenceCheckInterval) {
            clearInterval(this.silenceCheckInterval);
            this.silenceCheckInterval = null;
        }

        if (this.mediaRecorder?.state !== 'inactive') {
            this.mediaRecorder.stop();
        }

        this.mediaRecorder?.stream?.getTracks().forEach(t => t.stop());

        // Clean up AudioContext to avoid resource leaks
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().catch(() => { });
        }
        this.audioContext = null;
        this.analyser = null;

        if (this.animationFrame) cancelAnimationFrame(this.animationFrame);

        this.orb.classList.remove('listening');
        this.orb.classList.add('thinking');
        this.setStatus('thinking');
    }

    startSilenceDetection() {
        const dataArray = new Uint8Array(this.analyser.frequencyBinCount);

        this.silenceCheckInterval = setInterval(() => {
            if (!this.isRecording) return;

            this.analyser.getByteFrequencyData(dataArray);
            const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;

            if (avg > this.silenceThreshold) {
                // Speech detected
                this.lastSpeechTime = Date.now();
            } else {
                // Silence - check if exceeded duration
                const silentFor = Date.now() - this.lastSpeechTime;
                if (silentFor >= this.silenceDuration) {
                    console.log('Auto-stop: silence detected');
                    this.stopRecording();
                }
            }
        }, 100);
    }

    async processRecording() {
        const blob = new Blob(this.audioChunks, { type: 'audio/webm' });

        // Check if recording is too short (likely accidental)
        if (blob.size < 5000) {
            this.orb.classList.remove('thinking');
            this.setStatus('');
            this.idleAnimation();
            return;
        }

        const reader = new FileReader();
        reader.onloadend = async () => await this.sendToServer(reader.result);
        reader.readAsDataURL(blob);
    }

    async sendToServer(audioData) {
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ audio: audioData })
            });

            const data = await res.json();

            if (data.error) {
                this.setStatus('');
                this.orb.classList.remove('thinking');
                this.idleAnimation();
                return;
            }

            this.response.textContent = data.text;
            this.response.classList.add('visible');

            if (data.audio) {
                this.orb.classList.remove('thinking');
                this.orb.classList.add('speaking');
                this.setStatus('');

                this.currentAudio = new Audio(data.audio);
                this.currentAudio.onended = () => {
                    this.currentAudio = null;
                    this.orb.classList.remove('speaking');
                    this.stopInterruptListener();
                    // Start follow-up listening after Sarah speaks
                    this.startFollowUpListening();
                };
                this.currentAudio.play();
                this.speakingAnimation();

                // Start listening for interrupts while speaking
                this.startInterruptListener();
            } else {
                this.orb.classList.remove('thinking');
                this.idleAnimation();
            }

        } catch (err) {
            console.error('Error:', err);
            this.orb.classList.remove('thinking');
            this.idleAnimation();
        }
    }

    async startInterruptListener() {
        // Listen for speech that could interrupt Sarah
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.interruptContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.interruptContext.createMediaStreamSource(stream);
            this.interruptAnalyser = this.interruptContext.createAnalyser();
            this.interruptAnalyser.fftSize = 256;
            source.connect(this.interruptAnalyser);

            this.interruptStream = stream;

            const dataArray = new Uint8Array(this.interruptAnalyser.frequencyBinCount);
            let speechFrames = 0;
            const requiredFrames = 6; // ~600ms of speech (2+ words)

            this.interruptInterval = setInterval(() => {
                if (!this.currentAudio) return;

                this.interruptAnalyser.getByteFrequencyData(dataArray);
                const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;

                if (avg > 20) {
                    speechFrames++;
                    if (speechFrames >= requiredFrames) {
                        console.log('Interrupt detected!');
                        this.handleInterrupt();
                    }
                } else {
                    speechFrames = Math.max(0, speechFrames - 1);
                }
            }, 100);

        } catch (err) {
            console.error('Interrupt listener error:', err);
        }
    }

    stopInterruptListener() {
        if (this.interruptInterval) {
            clearInterval(this.interruptInterval);
            this.interruptInterval = null;
        }
        if (this.interruptStream) {
            this.interruptStream.getTracks().forEach(t => t.stop());
            this.interruptStream = null;
        }
        if (this.interruptContext && this.interruptContext.state !== 'closed') {
            this.interruptContext.close().catch(() => { });
        }
        this.interruptContext = null;
        this.interruptAnalyser = null;
    }

    handleInterrupt() {
        // Stop Sarah speaking
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio = null;
        }
        this.stopInterruptListener();
        this.orb.classList.remove('speaking');

        // Send interrupt signal and start recording
        fetch('/api/interrupt', { method: 'POST' });

        this.hint.textContent = 'Interrupted! Listening...';
        this.startRecording();
    }

    async startFollowUpListening() {
        // Wait for follow-up speech for up to 10 seconds
        const followUpTimeout = 10000;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);

            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (e) => this.audioChunks.push(e.data);
            this.mediaRecorder.onstop = () => this.processRecording();

            this.mediaRecorder.start();
            this.isRecording = true;
            this.lastSpeechTime = null; // Not started speaking yet

            this.orb.classList.add('listening');
            this.setStatus('listening');
            this.hint.textContent = 'Waiting for follow-up...';
            this.response.classList.remove('visible');

            this.drawWaveform();

            const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            const startTime = Date.now();
            let speechDetected = false;

            this.silenceCheckInterval = setInterval(() => {
                if (!this.isRecording) return;

                this.analyser.getByteFrequencyData(dataArray);
                const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;

                if (avg > this.silenceThreshold) {
                    // Speech detected
                    speechDetected = true;
                    this.lastSpeechTime = Date.now();
                } else if (speechDetected && this.lastSpeechTime) {
                    // Silence after speech - check if done speaking
                    const silentFor = Date.now() - this.lastSpeechTime;
                    if (silentFor >= this.silenceDuration) {
                        console.log('Follow-up complete');
                        this.stopRecording();
                        return;
                    }
                }

                // Timeout - no speech detected
                const elapsed = Date.now() - startTime;
                if (!speechDetected && elapsed >= followUpTimeout) {
                    console.log('No follow-up, returning to idle');
                    this.cancelRecording();
                }
            }, 100);

        } catch (err) {
            console.error('Follow-up mic error:', err);
            this.idleAnimation();
        }
    }

    cancelRecording() {
        // Stop without processing (user didn't say anything)
        this.isRecording = false;
        document.body.classList.remove('active');
        this.hint.textContent = 'Click or press Space to talk';

        if (this.silenceCheckInterval) {
            clearInterval(this.silenceCheckInterval);
            this.silenceCheckInterval = null;
        }

        if (this.mediaRecorder?.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        this.mediaRecorder?.stream?.getTracks().forEach(t => t.stop());
        this.audioChunks = []; // Clear chunks so processRecording does nothing

        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().catch(() => { });
        }
        this.audioContext = null;
        this.analyser = null;

        if (this.animationFrame) cancelAnimationFrame(this.animationFrame);

        this.orb.classList.remove('listening', 'thinking');
        this.setStatus('');
        this.idleAnimation();
    }

    setStatus(text) {
        this.status.textContent = text;
        this.status.classList.toggle('active', text !== '');
    }

    drawWaveform() {
        if (!this.isRecording || !this.analyser) return;

        const data = new Uint8Array(this.analyser.frequencyBinCount);
        this.analyser.getByteFrequencyData(data);

        this.ctx.clearRect(0, 0, 200, 200);

        const cx = 100, cy = 100, base = 35;

        this.ctx.beginPath();
        this.ctx.strokeStyle = 'rgba(0, 255, 150, 0.6)';
        this.ctx.lineWidth = 2;

        for (let i = 0; i < data.length; i++) {
            const angle = (i / data.length) * 2 * Math.PI;
            const amp = data[i] / 255 * 25;
            const r = base + amp;
            const x = cx + Math.cos(angle) * r;
            const y = cy + Math.sin(angle) * r;
            i === 0 ? this.ctx.moveTo(x, y) : this.ctx.lineTo(x, y);
        }

        this.ctx.closePath();
        this.ctx.stroke();

        this.animationFrame = requestAnimationFrame(() => this.drawWaveform());
    }

    idleAnimation() {
        let phase = 0;
        const draw = () => {
            if (this.isRecording || this.orb.classList.contains('speaking')) return;

            this.ctx.clearRect(0, 0, 200, 200);
            const r = 35 + Math.sin(phase) * 3;

            this.ctx.beginPath();
            this.ctx.strokeStyle = 'rgba(0, 150, 255, 0.3)';
            this.ctx.lineWidth = 1;
            this.ctx.arc(100, 100, r, 0, 2 * Math.PI);
            this.ctx.stroke();

            phase += 0.03;
            this.animationFrame = requestAnimationFrame(draw);
        };
        draw();
    }

    speakingAnimation() {
        let phase = 0;
        const draw = () => {
            if (!this.orb.classList.contains('speaking')) {
                this.idleAnimation();
                return;
            }

            this.ctx.clearRect(0, 0, 200, 200);
            this.ctx.beginPath();
            this.ctx.strokeStyle = 'rgba(255, 200, 100, 0.6)';
            this.ctx.lineWidth = 2;

            for (let i = 0; i < 360; i++) {
                const angle = (i / 360) * 2 * Math.PI;
                const wave = Math.sin(i * 0.08 + phase) * 8;
                const r = 40 + wave;
                const x = 100 + Math.cos(angle) * r;
                const y = 100 + Math.sin(angle) * r;
                i === 0 ? this.ctx.moveTo(x, y) : this.ctx.lineTo(x, y);
            }

            this.ctx.closePath();
            this.ctx.stroke();

            phase += 0.15;
            this.animationFrame = requestAnimationFrame(draw);
        };
        draw();
    }
}

document.addEventListener('DOMContentLoaded', () => new SarahAssistant());
