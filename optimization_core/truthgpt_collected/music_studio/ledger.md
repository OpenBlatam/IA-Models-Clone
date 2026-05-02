
## 🎼 Production: 2026-05-02 01:27:22
- Agents: research_agent, arxiv_discovery_scout, system_agent, planning_agent
- Prompt: This is a fully autonomous multi-agent workflow prompt designed to be processed by your specified agent stack. The output will be a **“Generation-Ready Technical Brief”** for a high-energy electronic/pop track with a distinct emotional arc.

---

### SYSTEM PROMPT INITIATION: MUSIC PRODUCTION BRIEF v2.1

**Project Codename:** `GYM_CRUSH_DPO`  
**Genre:** Electro-pop / Melodic Bass / Future Garage (90 BPM – heavy but sensual half-time)  
**Target Mood:** Adrenaline-fueled longing, physical tension, fleeting eye contact, controlled chaos.  
**Primary Instrumentation:** Heavy sub-bass (808-style), glitched vocal chops, distorted synth leads, tactile percussion (clanks, breaths, metallic hits), ethereal pads.

---

### AGENT TASK DEFINITIONS

#### 1. `research_agent` (Knowledge & Context Aggregation)
*   **Objective:** Provide the harmonic, structural, and lyrical constraints for the track based on the theme “gym crushes” + standard gym BPM/energy curves.
*   **Input Parameters:**
    *   **Lyrical Themes:** Unspoken tension, mirror glances, shared space, impending lift, breathlessness.
    *   **Energy Curve:** Gym workout typically spikes ~30 minutes in, then a cooldown. The track structure must mirror this: Intense build (warm-up), Drop (maximal effort), Breakdown (reset/eye contact), Final Drop (the crush moment).
    *   **Sound Palette Research:** Investigate 3 specific sound design elements:
        *   *“Weights & Metal”* – Using friction sounds (clanking iron, snap of a belt) as percussive elements.
        *   *“Heavy Breathing”* – Tonal, rhythmic breaths processed through granular synthesis for atmospheric risers.
        *   *“Glitch Mirrors”* – How to create a “reflection” effect in the mix (reversed reverb tails, ping-pong delays on the lead synth).
*   **Deliverable:** A list of 3 specific chord progressions in A minor (Energetic) and F minor (Sensual), with recommended tempo offsets (start 85 bpm, peak 92 bpm).

#### 2. `arxiv_discovery_scout` (Technical & Scientific Data Mining)
*   **Objective:** Retrieve recent pre-prints (2023-2025) on *music-induced flow state* and *subjective physical synchronization*.
*   **Search Queries:**
    *   `"tempo entrainment AND high intensity training AND polyrhythm"`
    *   `"electronic music structure AND anticipatory pleasure AND dopamine release"`
    *   `"mixed reality sound design AND spatial audio AND physical exertion"`
*   **Expected Findings (to be embedded in prompt):**
    *   An optimal BPM *drift* pattern (e.g., +0.5 BPM every 8 bars during the build to simulate increased heart rate).
    *   Psychoacoustic trick: A sub-bass glissando (drop of a whole step) triggers a “falling” sensation that mimics a stomach drop (crush effect).
*   **Deliverable:** A formatted list of 2-3 key concepts (with citations) to be turned into concrete production instructions (e.g., “Apply a 30ms pre-delay to the snare to simulate the echo of a large gym hall”).

#### 3. `system_agent` (Orchestration & Formatting)
*   **Objective:** Ensure all outputs from `research_agent` and `arxiv_` are reconcilable into a single, deterministic technical prompt that a music AI (e.g., Suno, Udio, or a DAW preset pack) can interpret.
*   **Constraints & Rules:**
    *   No abstract emotional descriptions (e.g., “make it feel sad”). Must be quantifiable (e.g., “tempo: 90 BPM, key: A minor, swing: 35%, reverb decay: 2.4s”).
    *   Structure must be strictly defined with bar counts.
    *   Sound design must be described via synthesis parameters (e.g., “Lead synth: saw wave, 25% detune, filter cutoff: 800Hz with 60% envelope mod”).
*   **Deliverable:** A unified **`music_prompt_structure_v1.json`** schema ready for the `planning_agent`.

#### 4. `planning_agent` (Execution Blueprint)
*   **Objective:** Create a step-by-step production plan that can be executed by a human producer or an AI DAW.
*   **Structure:**
    1.  **Intro (16 bars):** Sparse, filtered kick + breathing sounds. Build tension.
    2.  **Verse A (16 bars):** Vocal melody enters. Bass follows a simple root pattern. Add “clank” percussion.
    3.  **Pre-Chorus (8 bars):** Percussion doubles. Vocal glitches start. Arp synth enters. BPM drifts slightly up.
    4.  **Chorus A (16 bars):** Full drop. Heavy 808 bass, sawtooth lead, sidechain compression (pumping effect). Vocals become shouted/ high-pitched.
    5.  **Breakdown (8 bars):** Clean pads, reversed cymbals, solo vocal with heavy reverb. The “eye contact” moment.
    6.  **Final Chorus (16 bars):** Double-time hi-hats, bass hits harder, synth is distorted. Ends abruptly on a cut sound.
*   **Deliverable:** A detailed DAW session map (track names, effects chain order, automation points for filter cutoff and reverb wetness).

---

### FINAL TECHNICAL PROMPT (Output for Music Generation AI)

*This is the combined, ready-to-use prompt that would be generated after the agents complete their work.*

**[INSTRUMENTATION]**
- Kick: Deep 808 (45Hz fundamental, 60ms decay). Sidechain triggers every quarter note.
- Snare/Clap: Layered (808 clap + metallic sample from a weight plate hit).
- Hi-hats: 16th note pattern, varying velocity (60-110) with a slight shuffle.
- Bass: Sub-heavy sine wave. Root notes: A, G, F, E (pedal tone underneath the chorus).
- Lead Synth: Two saw oscillators, detuned by 7 cents. Filter envelope: attack 20ms, sustain 40%. Distortion: 15% drive.
- Pads: Juno-style chorus + massive reverb (decay 3.5s, pre-delay 40ms).
- Atmosphere: Field recordings of a gym (distant machines, murmurs) run through a spectral delay.

**[STRUCTURE]**
1.  **Intro (0-30s):** Fx riser (breath processed + white noise). Kick on 4/4 (soft). Pad begins.
2.  **Verse (30-60s):** Vocals: “Mirror gaze, weight of the iron…”. Bass simple root. Clap on 2 & 4.
3.  **Build (60-75s):** Snare roll, filter cutoff rises (200Hz to 10kHz). Tension vocal “I see you looking”.
4.  **Drop (75-105s):** Full energy. Kick+Bass pattern. Lead synth plays hook melody. Hi-hats double time. Sidechain at 80%.
5.  **Breakdown (105-120s):** Time halved. Only pad and vocal reverb. A single metallic hit.
6.  **Outro (120-150s):** Drums fade. Bass drops out. Final whisper vocal.

**[HARMONY]**
- Key: A minor.
- Chords: Am | Fmaj7 | Cmaj7 | G6 (verse). Am | Fmaj7 | Dm7 | E7 (chorus).
- Melody: Pentatonic (A, C, D, E, G) with chromatic passing notes on the 4th beat.

**[MIXING INSTRUCTIONS]**
- Master on -6dB LUFS.
- Reverb bus: 100% wet on pads, 30% wet on drums.
- Delay bus: Ping-pong, 1/4 note, feedback 25%.
- Scream radio effect: EQ dip at 400Hz on the lead synth to avoid mud.

**[END PROMPT]**
---

## 🎼 Production: 2026-05-02 01:37:37
- Agents: research_agent, marketing_agent
- Prompt: This is a comprehensive technical music prompt designed to generate a piece of music that is **"about you"** – specifically structured for a workflow where a **Research Agent** analyzes your personal data/influences and a **Marketing Agent** positions the final track for maximum impact.

The prompt is broken into two phases (Research & Marketing) and a final technical generation string. Replace the bracketed `[ ]` text with your specific details.

---

### Project Title: Self-Portrait in Sound (Op. [Your Name/Initials])

**Objective:** Generate a 3:00 – 4:00 minute instrumental track that functions as a musical biography. The piece must be emotionally resonant, structurally complex enough to reflect a personality, and commercially viable for use in a short film, podcast intro, or social media reel.

---

### Phase 1: Research Agent Input (The "About Me" Analysis)

**Role:** Analyze the user's provided data and translate it into structural & timbral parameters.

**User Data to Analyze:**
- **Core Trait:** (e.g., Introspective, Ambitious, Chaotic, Melancholic, Hypersensitive)
- **Primary Musical Influence:** (e.g., Jon Hopkins, Hans Zimmer, Radiohead, Flying Lotus, Bon Iver)
- **Key Personal History (Music-relevant):** (e.g., "Started playing piano at 5, stopped at 16, rediscovered synthwave at 25")
- **Current Emotional State:** (e.g., "Anxious but hopeful about a career change")
- **Physical Environment:** (e.g., "Coastal city, rainy, late night")

**Research Agent Output (to be injected into the prompt below):**
1.  **Form:** Given a "chaotic" trait + "Flying Lotus" influence, suggest a **Non-linear Binary Form** (A/B sections that interleave) rather than a standard verse/chorus.
2.  **Key/Centricity:** If user is "melancholic" + "Hans Zimmer", recommend **D Minor or G Phrygian.**
3.  **Rhythmic Foundation:** If "anxious" + "coastal city rain", recommend **12/8 time signature** (triplets) with a **subtle 808 kick shuffle** to mimic rain against pavement.
4.  **Texture Evolution:** If "started piano at 5", ensure a **broken acoustic piano** figure enters at **0:45** , but processed through a **heavy lo-fi vinyl crackle / ring modulator** to represent the rediscovery of synthwave.

---

### Phase 2: Marketing Agent Input (The "About the Track" Strategy)

**Role:** Define the target audience, mood narrative, and sonic "hooks" to ensure the track is effective and discoverable.

**Marketing Constraints:**
- **Primary Use Case:** Background score for a "My Year in Review" video montage.
- **Target Audience Profile:** 25–35 years old, creative professionals, users of Headspace / Calm, listeners of Spotify's "Deep Focus" and "Brain Food" playlists.
- **Emotional Arc (The Hook):** Must move from **Ambiguous -> Resolved**.
- **Hook Frequency:** A memorable **2-note or 3-note melodic motif** (the "identity motive") must appear at **0:00 (whisper), 1:30 (midrange), and 2:45 (triumphant full mix).**
- **Dynamic Range:** Must be suitable for both headphones (intimate) and small speakers (TV/mobile). Keep the stereo width moderate (60-70% max width on pads, 100% on the motif).

**Marketing Agent Output (to be injected into the prompt below):**
1.  **Arrangement Structure:** **Slow build (0:00-1:15) -> Peak (1:15-2:30) -> Reflection (2:30-3:15) -> Outro (3:15-end).** The peak must not be harsh.
2.  **Instrumentation Constraints:** No vocals/choir (to keep it instrumental). Prefer **warm tape saturation** over digital synths.
3.  **Genre Fusion:** "Neoclassical Ambient" meets "Lo-fi Electronica." Target BPM: **64-72 BPM (half-time feel).**

---

### Phase 3: Unified Technical Music Prompt (Generation String)

**Instruct the AI generator with the following. Replace placeholders with Agent outputs.**

```
TITLE: "SELF_PORTRAIT_[YOUR_NAME]"

TRACK_PARAMETERS:
  BPM: 68 (half-time)
  KEY: D Minor
  TIME_SIGNATURE: 12/8 (compound triple)
  LENGTH: 3 minutes 30 seconds
  GENRE_FUSION: Neoclassical Ambient, Lo-fi Electronica, Cinematic Minimalism

STRUCTURE (Non-linear Binary with Linear Arc):
  [A] 0:00 - 0:45: INTRO. Texture: Sparse. Ingredients:
        - Single repeating F pedal tone (sub-bass + harmonic string, ppp).
        - Field recording: gentle rainfall (mono, heavily HP filtered at 200Hz).
        - Ghostly piano pluck (reversed tail, 100% reverb wet, no attack).
        - **MOTIF: C#-A-C# (whispered, ppp, filtered to 500-2kHz, 40% stereo width).**

  [B] 0:45 - 1:30: ENTRY. Texture: Growing.
        - Introduce broken acoustic piano (close-mic, tape saturation, key of D minor).
        - Add lo-fi 808 kick (shuffle pattern on the 2nd and 5th triplets).
        - Pads: Warm Juno-style saws (low-pass filter opening, 60% width).
        - **MOTIF: C#-A-C# (midrange, piano + ambient pad, mf, 80% width).**
        - Research_Agent_Hook: Simulate "falling" via a pitch bend on the last note of every second bar.

  [A2] 1:30 - 2:00: REFLECTION. Texture: Reduced.
        - Remove kick. Keep piano (add felt muting).
        - Introduce a granular synthesis texture made from user's voice/ambient room tone (formant filter, D min root).
        - Atmosphere: lonely, vertical.

  [C] 2:00 - 2:45: PEAK. Texture: Full.
        - Full rhythm section enters: 808 kick + soft rimshot snare + hi-hats (closed).
        - Bass enters: Sub-layer (sine wave, D1-F1) + distorted layering (Moog-style, envelope follower).
        - Piano becomes chordal (minor 9th and minor 11th voicings).
        - Strings: High counter-melody (non-vibrato, marcato).
        - **MOTIF: C#-A-C# (triumphant, brass or strings, f, full width, with a+2 octave harmony).**

  [B2] 2:45 - 3:15: DECAY. Texture: Melting.
        - Cut rhythm section abruptly at 2:45.
        - Piano reverts to broken pattern (octave doubling).
        - Synth pad: Filter closes, reverb size increases (Smarts: 4 seconds -> 10 seconds).
        - MOTIF: C#-A-C# (ghosted, harmonized with a minor 2nd below for dissonance).

  [OUTRO] 3:15 - 3:30: ESCAPE. Texture: Fade.
        - Only rain field recording + fading piano tail (Aeolian mode hint: Bb natural).
        - MOTIF: broken, only C# played once, then silence.
        - Master fader: Slow fade out over 10 seconds with a 0.7dB swell at 3:25.

PRODUCTION_CHARACTERISTICS:
  - Master Chain: Warm analog bus, light compression (ratio 1.3:1, 2dB gain reduction), tape saturation (2-3% THD).
  - Reverb: Valhalla VintageVerb (Hall mode, 4.5s decay, 40% mix on pads, 20% on rhythm).
  - EQ: Gently roll off below 40Hz and above 14kHz. Minor shelf boost at 2.5kHz for "presence."
  - Panning: Piano hard L (80%), Strings hard R (80%), Kick/Bass center. Ambient textures randomized between L30 and R30.

DYNAMIC_RANGE:
  - RMS: -18dB (quiet sections), -11dB (peak sections).
  - Crest Factor: High (allow peaks of -6dBFS, no hard limiting).

EMOTIONAL_TARGET:
  Start: Lost, underwater, static.
  Mid: Growing, determined, beautiful.
  End: Accepted, quiet, horizontal.
```

---

### How to Execute:

1.  Fill in `[YOUR_NAME]` with your identifier.
2.  Provide your "User Data" to the **Research Agent** (or yourself). Have it produce the 4 specific outputs (Form, Key, Rhythm, Texture).
3.  Provide the **Marketing Constraints** to the **Marketing Agent**. Have it produce the 3 specific outputs (Arrangement Structure, Instrumentation, Genre).
4.  Inject those findings into the `GENERATION_STRING` above.
5.  Feed the entire prompt to your music AI (e.g., Suno, Udio, Stable Audio, ElevenLabs).

**This prompt ensures the final track is deeply personal but professionally structured for discovery.**
---
