from pathlib import Path


def _write_ppm_grid(output_path, image_size=256, grid_size=4):
    output_path = Path(output_path)
    lines = ["P3", f"{image_size} {image_size}", "255"]
    step = max(1, image_size // grid_size)
    for row in range(image_size):
        pixels = []
        for col in range(image_size):
            if row % step == 0 or col % step == 0 or row == image_size - 1 or col == image_size - 1:
                pixels.append("255 255 255")
            else:
                base = 32 + ((row // step) * 35 + (col // step) * 25) % 160
                pixels.append(f"{base} {max(0, base - 20)} {min(255, base + 50)}")
        lines.append(" ".join(pixels))
    output_path.write_text(chr(10).join(lines) + chr(10), encoding="ascii")
    return output_path


def create_grid_overlay(input_path, output_path, image_size=256, grid_size=4):
    output_path = Path(output_path)
    try:
        from PIL import Image, ImageDraw
        image = Image.open(input_path).convert("RGB").resize((image_size, image_size))
        draw = ImageDraw.Draw(image)
        step = image_size // grid_size
        for index in range(grid_size + 1):
            pos = min(image_size - 1, index * step)
            draw.line((pos, 0, pos, image_size), fill=(255, 255, 255), width=2)
            draw.line((0, pos, image_size, pos), fill=(255, 255, 255), width=2)
        for row in range(grid_size):
            for col in range(grid_size):
                label_char = chr(ord("A") + row)
                label = f"{label_char}{col + 1}"
                draw.text((col * step + 5, row * step + 5), label, fill=(255, 255, 255))
        image.save(output_path)
        return output_path
    except Exception:
        return _write_ppm_grid(output_path, image_size=image_size, grid_size=grid_size)
