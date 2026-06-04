# Matt Gift Pi 4 Display Bring-Up

This setup targets a Raspberry Pi 4 running GhostFM headlessly with a CUQI/LCDWiki-style 3.5 inch MHS35 SPI display at 480x320.

The first milestone is only screen rendering. Touchscreen input, UI buttons, RTL-SDR tuning controls, and the final enclosure workflow can come later.

## 1. Download Raspberry Pi Imager

Install Raspberry Pi Imager from the official Raspberry Pi software page:

https://www.raspberrypi.com/software/

Use it to flash the new microSD card.

## 2. Flash The SD Card

In Raspberry Pi Imager:

1. Choose device: `Raspberry Pi 4`
2. Choose OS: `Raspberry Pi OS (other)` -> `Raspberry Pi OS Lite (32-bit)`
3. Choose storage: the new microSD card
4. Open OS customization before writing.

Use these customization settings:

- Hostname: `pi`
- Username: `ghostfm`
- Enable SSH.
- Wi-Fi SSID: `kooka-net`
- Wi-Fi password: use the provided password.
- Set timezone/locale normally for your location.

Write the card, eject it, insert it into the Pi 4, attach the 3.5 inch display, and power on.

After a few minutes, connect over SSH:

```bash
ssh ghostfm@pi.local
```

If `.local` name resolution does not work, find the Pi's IP address from your router and SSH to that:

```bash
ssh ghostfm@<pi-ip-address>
```

## 3. Install The Screen Driver

The paper calls this Raspbian-only. That is not a blocker for this setup: Raspbian is the old/common name for Raspberry Pi OS, and the vendor LCD-show driver supports Raspberry Pi OS Bookworm paths.

On the Pi:

```bash
sudo apt update
sudo apt install -y git
sudo rm -rf LCD-show
git clone https://github.com/goodtft/LCD-show.git
chmod -R 755 LCD-show
cd LCD-show
sudo ./MHS35-show 90
```

The Pi should reboot after the driver install.

Reconnect:

```bash
ssh ghostfm@pi.local
```

Verify the LCD framebuffer exists:

```bash
ls /dev/fb*
test -e /dev/fb1 && echo "LCD framebuffer found"
```

If the screen is sideways or upside down, rotate it from the driver directory:

```bash
cd ~/LCD-show
sudo ./rotate.sh 0
sudo ./rotate.sh 90
sudo ./rotate.sh 180
sudo ./rotate.sh 270
```

Keep whichever rotation makes the display upright. Prefer the driver rotation over GhostFM's `--display-rotate` flag.

## 4. Install GhostFM

Clone the repo and switch to this branch:

```bash
cd ~
git clone <repo-url> the-digital-ear
cd the-digital-ear
git switch matt-gift
```

Install system packages:

```bash
sudo apt install -y python3-venv python3-pip python3-numpy python3-pil python3-psutil fonts-dejavu-core
```

Create the venv:

```bash
python3 -m venv --system-site-packages venv
```

Install runtime Python dependencies:

```bash
./venv/bin/pip install sounddevice gpiozero lgpio
```

## 5. Run The Display Test

Do this before starting the full FM/audio pipeline:

```bash
./venv/bin/python ghost_fm.py \
  --display-test \
  --display-backend fbdev \
  --fbdev /dev/fb1 \
  --display-width 480 \
  --display-height 320 \
  --display-byte-order little \
  --display-normal-assets \
  --display-ui-scale 2.5 \
  --display-asset-scale 1.5 \
  --no-joystick
```

Expected result:

- The 3.5 inch screen fills with the GhostFM display.
- The screen is not cropped, mirrored, offset, or upside down.
- The note/status fields animate.

Stop the test with `Ctrl-C`.

If nothing appears:

```bash
ls /dev/fb*
journalctl -b | grep -i -E 'fb|spi|ili|lcd'
```

Then retry the LCD-show driver install and rotation steps.

If orientation is correct but colors look wrong, try the alternate RGB565 byte order:

```bash
./venv/bin/python ghost_fm.py \
  --display-test \
  --display-backend fbdev \
  --fbdev /dev/fb1 \
  --display-width 480 \
  --display-height 320 \
  --display-byte-order big \
  --display-normal-assets \
  --display-ui-scale 2.5 \
  --display-asset-scale 1.5 \
  --no-joystick
```

## 6. Enable Autostart

Only do this after the display test works.

```bash
sudo cp ghostfm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ghostfm
sudo systemctl start ghostfm
```

Watch logs:

```bash
journalctl -u ghostfm -f
```

Stop/disable while debugging:

```bash
sudo systemctl stop ghostfm
sudo systemctl disable ghostfm
```

## Current Scope

This branch only adapts GhostFM rendering for the Pi 4 MHS35 framebuffer display. It intentionally does not implement touchscreen controls or new UI buttons yet.
