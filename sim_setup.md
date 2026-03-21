# DonkeyCar Simulator Setup Guide

One-time setup instructions for installing sdsandbox and gym-donkeycar.
The agent does not touch simulator installation — this is a human task.

---

## 1. Install sdsandbox (Unity binary)

Download the pre-built simulator binary from the
[gym-donkeycar releases page](https://github.com/tawnkramer/gym-donkeycar/releases)
(Linux / macOS / Windows builds are provided). No Unity editor required.

```bash
# Example: Linux — check https://github.com/tawnkramer/gym-donkeycar/releases for the latest version
wget https://github.com/tawnkramer/gym-donkeycar/releases/latest/download/DonkeySimLinux.zip
unzip DonkeySimLinux.zip -d ~/donkey_sim
chmod +x ~/donkey_sim/donkey_sim.x86_64
```

For macOS or Windows, download the corresponding archive from the releases page and extract.

---

## 2. Install gym-donkeycar Python package

```bash
pip install git+https://github.com/tawnkramer/gym-donkeycar
```

Or install a specific version:

```bash
pip install gym-donkeycar==22.11.6   # replace with current release
```

---

## 3. Start the simulator in server mode (headless)

```bash
# Headless (training server, no display required)
~/donkey_sim/donkey_sim.x86_64 --headless --port 9091 &
```

> The simulator listens on TCP port 9091. `gym-donkeycar` connects as a client on the same port.
> Multiple Python processes can open separate connections on different ports.

For a headed (GUI) session — useful for debugging:

```bash
~/donkey_sim/donkey_sim.x86_64 --port 9091
```

---

## 4. Verify the connection

```python
import gymnasium as gym
import gym_donkeycar

conf = {"exe_path": "already_running", "port": 9091}  # attach to running sim
env = gym.make("donkey-generated-track-v0", conf=conf)
obs, info = env.reset()
print(obs.shape)   # (120, 160, 3)
env.close()
```

If this prints `(120, 160, 3)`, the simulator is correctly configured.

---

## 5. Generate training data

Once the simulator is running, generate a training tub:

```bash
python prepare_donkey.py --generate                    # default: 20k steps, port 9091
python prepare_donkey.py --generate --num-steps 40000  # more data
python prepare_donkey.py --generate --port 9092         # different port
```

This drives the simulated car using a PD lane-centering controller and saves
each frame to `~/donkeycar/data/sim_tub/`.

---

## 6. Available tracks

| Gym env id | Track name |
|---|---|
| `donkey-generated-track-v0` | Procedurally generated road (good default) |
| `donkey-warehouse-v0` | Indoor warehouse circuit |
| `donkey-mountain-track-v0` | Outdoor mountain road |
| `donkey-roboracingleague-track-v0` | RoboRacing League oval |

For AutoResearch experiments the **generated track** is recommended: it is procedurally varied,
so the agent cannot overfit to a fixed layout.

---

## 7. Troubleshooting

- **"Connection refused"**: Make sure the sdsandbox binary is running and the port matches.
- **Black screen / no frames**: Ensure you're using `--headless` mode on a server without a display,
  or set `DISPLAY` correctly if using a virtual framebuffer (e.g. `xvfb-run`).
- **Slow simulator**: The Unity sim requires a reasonable CPU. On a headless server, ensure
  GPU rendering is available (e.g. `nvidia-smi` shows a GPU). The sim uses GPU for rendering.
- **gym-donkeycar import errors**: Make sure `gymnasium` is installed alongside `gym-donkeycar`.
