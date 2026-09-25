import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import soundfile as sf
import io
import os
import tempfile
import base64
import traceback

st.set_page_config(page_title="Consola Multitrack de Stems", layout="wide")

st.title("🎛️ Consola Multitrack de Stems con Waveform Interactivo")
st.markdown("Separa tus pistas con IA y contrólalas con la interfaz de ondas interactivas para cada canal.")

uploaded_file = st.file_uploader("Sube tu archivo de audio (WAV o MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_filename = tmp_file.name

    try:
        data, samplerate = sf.read(temp_filename)
        st.success("¡Archivo cargado con éxito! Listo para separar.")
    except Exception as e:
        st.error(f"Error al leer el archivo de audio: {e}")

    if st.button("🚀 Separar Stems con Inteligencia Artificial"):
        progress_text = st.empty()
        bar = st.progress(0)
        
        progress_text.text("🔄 Inicializando motor de inteligencia artificial...")
        bar.progress(15)

        try:
            from audio_separator.separator import Separator
            
            progress_text.text("📥 Cargando modelo Demucs (htdemucs_6s)...")
            bar.progress(40)
            
            separator = Separator(output_dir=os.getcwd())
            separator.load_model('htdemucs_6s.yaml')
            
            progress_text.text("🎧 Separando pistas (esto puede tardar unos minutos)...")
            bar.progress(70)
            
            output_files = separator.separate(temp_filename)
            
            bar.progress(95)
            abs_output_files = [
                f if os.path.isabs(f) else os.path.join(os.getcwd(), f) 
                for f in output_files
            ]
            
            st.session_state["separated_stems"] = abs_output_files
            bar.progress(100)
            progress_text.text("✅ ¡Separación completada con éxito!")
            st.success("¡Separación completada con éxito!")
        except Exception as e:
            progress_text.empty()
            bar.empty()
            st.error("Detalle técnico del error:")
            st.code(traceback.format_exc())

    if "separated_stems" in st.session_state:
        st.markdown("---")
        
        if "stem_names" not in st.session_state:
            st.session_state["stem_names"] = {}
            for idx, filepath in enumerate(st.session_state["separated_stems"]):
                base_name = os.path.basename(filepath).replace(".wav", "").replace(".mp3", "")
                st.session_state["stem_names"][idx] = base_name

        stems_data = []
        max_len = 0
        valid_stems = []

        for idx, filepath in enumerate(st.session_state["separated_stems"]):
            stem_path = filepath if os.path.exists(filepath) else os.path.basename(filepath)
            try:
                s_data, s_sr = sf.read(stem_path)
                if len(s_data.shape) > 1:
                    s_data = np.mean(s_data, axis=1)
                stems_data.append((idx, s_data, s_sr))
                if len(s_data) > max_len:
                    max_len = len(s_data)
                valid_stems.append(idx)
            except Exception:
                continue

        normalized_stems = {}
        sample_rate = stems_data[0][2] if stems_data else 44100

        for idx, s_data, s_sr in stems_data:
            if len(s_data) < max_len:
                s_data = np.pad(s_data, (0, max_len - len(s_data)), 'constant')
            normalized_stems[idx] = s_data

        # ==========================================
        # SECCIÓN 1: REPRODUCTOR MAESTRO (PLAY ALL)
        # ==========================================
        st.markdown("### 🎛️ Reproductor Maestro (Play All Sincronizado)")
        st.markdown("Mezcla global de todos los stems activos aplicando volúmenes, Mutes y Solos.")

        solo_active = any(st.session_state.get(f"solo_{idx}", False) for idx in valid_stems)
        master_mix = np.zeros(max_len)

        for idx in valid_stems:
            vol = st.session_state.get(f"vol_{idx}", 0.8)
            is_solo = st.session_state.get(f"solo_{idx}", False)
            is_mute = st.session_state.get(f"mute_{idx}", False)

            if solo_active and not is_solo:
                continue
            if is_mute:
                continue

            master_mix += normalized_stems[idx] * vol

        max_val = np.max(np.abs(master_mix))
        if max_val > 1.0:
            master_mix = master_mix / max_val

        master_io = io.BytesIO()
        sf.write(master_io, master_mix, sample_rate, format='WAV')
        master_bytes = master_io.getvalue()

        st.audio(master_bytes, format='audio/wav')
        st.markdown("---")

        # ==========================================
        # SECCIÓN 2: CANALES INDIVIDUALES CON WAVEFORM INTERACTIVO
        # ==========================================
        st.subheader("🎚️ Canales Individuales de Stems")

        for idx in valid_stems:
            current_name = st.session_state["stem_names"][idx]
            stem_data = normalized_stems[idx]

            col_title, col_btn = st.columns([4, 1])
            with col_title:
                st.markdown(f"### 🎵 Pista: {current_name}")
            with col_btn:
                edit_key = f"edit_mode_{idx}"
                if edit_key not in st.session_state:
                    st.session_state[edit_key] = False
                
                if st.button("✏️ Renombrar", key=f"btn_edit_{idx}"):
                    st.session_state[edit_key] = not st.session_state[edit_key]
                    st.rerun()

            if st.session_state.get(edit_key, False):
                new_name = st.text_input("Nuevo nombre para el Stem", value=current_name, key=f"input_rename_{idx}")
                if new_name and new_name != current_name:
                    st.session_state["stem_names"][idx] = new_name
                    st.rerun()

            col_controls, col_dl = st.columns([3, 2])
            with col_controls:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.checkbox("Solo (S)", key=f"solo_{idx}")
                with c2:
                    st.checkbox("Mute (M)", key=f"mute_{idx}")
                with c3:
                    vol = st.slider("Volumen", 0.0, 1.0, 0.8, key=f"vol_{idx}")
            
            with col_dl:
                st.markdown("<br>", unsafe_allow_html=True)
                stem_io = io.BytesIO()
                sf.write(stem_io, stem_data, sample_rate, format='WAV')
                stem_bytes_dl = stem_io.getvalue()
                st.download_button(
                    label=f"Descargar {current_name}",
                    data=stem_bytes_dl,
                    file_name=f"{current_name.lower().replace(' ', '_')}.wav",
                    mime="audio/wav",
                    key=f"dl_{idx}"
                )

            processed_audio = stem_data * vol
            bytes_io = io.BytesIO()
            sf.write(bytes_io, processed_audio, sample_rate, format='WAV')
            wav_bytes = bytes_io.getvalue()
            b64_audio = base64.b64encode(wav_bytes).decode()

            wave_html = f"""
            <div style="background-color: #262730; padding: 22px 22px 30px 22px; border-radius: 12px; border: 1px solid #3d3e48; margin-top: 10px; margin-bottom: 25px; font-family: sans-serif;">
                <p style="color: #ffffff; font-weight: 600; margin-bottom: 4px; font-size: 15px;">🎚️ Reproductor y Waveform Dinámico en Vivo</p>
                <p style="color: #b0b0b0; font-size: 13px; margin-bottom: 15px;">Stem: <b>{current_name}</b> (Haz clic o arrastra sobre la onda para mover el cursor)</p>
                
                <div id="waveform-{idx}"></div>
                
                <div style="margin-top: 18px; display: flex; align-items: center; gap: 12px;">
                    <button id="playBtn-{idx}" style="background-color: #00ffcc; color: #1e1e24; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 14px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">▶ Play</button>
                    <span id="time-{idx}" style="color: #ffffff; font-size: 14px; font-family: monospace;">0:00 / 0:00</span>
                </div>
            </div>

            <script src="https://unpkg.com/wavesurfer.js@7"></script>
            <script>
                const wavesurfer_{idx} = WaveSurfer.create({{
                    container: '#waveform-{idx}',
                    waveColor: '#00ffcc',
                    progressColor: '#008866',
                    cursorColor: '#ffffff',
                    cursorWidth: 2,
                    barWidth: 2,
                    barGap: 2,
                    height: 90,
                    url: 'data:audio/wav;base64,{b64_audio}'
                }});

                const playBtn_{idx} = document.getElementById('playBtn-{idx}');
                const timeLabel_{idx} = document.getElementById('time-{idx}');

                playBtn_{idx}.addEventListener('click', () => {{
                    wavesurfer_{idx}.playPause();
                }});

                wavesurfer_{idx}.on('play', () => {{
                    playBtn_{idx}.textContent = '⏸ Pause';
                }});

                wavesurfer_{idx}.on('pause', () => {{
                    playBtn_{idx}.textContent = '▶ Play';
                }});

                wavesurfer_{idx}.on('timeupdate', (currentTime) => {{
                    const duration = wavesurfer_{idx}.getDuration() || 0;
                    const formatTime = (seconds) => {{
                        const mins = Math.floor(seconds / 60);
                        const secs = Math.floor(seconds % 60);
                        return mins + ':' + (secs < 10 ? '0' : '') + secs;
                    }};
                    timeLabel_{idx}.textContent = formatTime(currentTime) + ' / ' + formatTime(duration);
                }});
            </script>
            """

            components.html(wave_html, height=270)
            st.markdown("---")
