"""Generate merlin.ico from the wizard emoji."""

import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs("assets", exist_ok=True)

size = 256
img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Use Segoe UI Emoji on Windows for color emoji rendering
font = None
for font_name in ["seguiemj.ttf", "NotoColorEmoji.ttf"]:
    try:
        font = ImageFont.truetype(font_name, 200)
        print(f"Using font: {font_name}")
        break
    except (OSError, IOError):
        continue

if font is None:
    try:
        font = ImageFont.truetype("arial.ttf", 200)
        print("Fallback to arial.ttf")
    except (OSError, IOError):
        font = ImageFont.load_default()
        print("Fallback to default font")

# Draw the wizard emoji centered
text = "\U0001F9D9"
bbox = draw.textbbox((0, 0), text, font=font)
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]
x = (size - tw) // 2 - bbox[0]
y = (size - th) // 2 - bbox[1]
draw.text((x, y), text, font=font, embedded_color=True)

# Verify something was drawn
extrema = img.getextrema()
print(f"Alpha channel range: {extrema[3]}")

# Save as .ico with multiple sizes for crisp display at all resolutions
icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
img.save("assets/merlin.ico", format="ICO", sizes=icon_sizes)

file_size = os.path.getsize("assets/merlin.ico")
print(f"Created assets/merlin.ico ({file_size:,} bytes)")
